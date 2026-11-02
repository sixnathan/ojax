module U = Yojson.Safe.Util
module T = Ojax.Types
module C = Ojax.Core
module A = Ojax.Api
module G = Ojax.Numpy.Getitem
module TU = Ojax.Tree_util
module PP = Ojax.Pretty_printer
module UF = Ojax.Numpy.Ufuncs

let name = "apply_transforms"

let func = function
  | "negative" -> UF.negative
  | "sin" -> UF.sin
  | "square" -> fun v -> UF.multiply v v
  | other -> failwith ("apply_transforms: func " ^ other)

let base case names values =
  let params = U.member "params" case in
  let env = List.combine names values in
  let idx =
    List.map (Index_fuzz.index_of env) (U.to_list (U.member "index" params))
  in
  let mode =
    match U.member "mode" params with `String m -> Some m | _ -> None
  in
  G.at_apply ?mode (List.assoc "x" env) idx
    (func (U.to_string (U.member "func" params)))

let leaves t = TU.tree_leaves t
let leaf v = TU.Leaf v

let program case names values =
  let params = U.member "params" case in
  let in_axes =
    List.map
      (function `Null -> None | v -> Some (U.to_int v))
      (U.to_list (U.member "in_axes" params))
  in
  let xi =
    let rec find i = function
      | [] -> failwith "apply_transforms: no x"
      | "x" :: _ -> i
      | _ :: t -> find (i + 1) t
    in
    find 0 names
  in
  let with_x x = List.mapi (fun i v -> if i = xi then x else v) values in
  let vmapped vals =
    leaves
      (A.vmap
         (fun args -> leaf (base case names (List.concat_map leaves args)))
         in_axes (List.map leaf vals))
  in
  match U.to_string (U.member "transform" params) with
  | "vmap0" | "vmap1" | "vmap_idx" | "vmap_both" -> vmapped values
  | "jit_vmap0" -> A.jit_flat ~name:"f" vmapped values
  | "grad" ->
      leaves
        (A.grad
           (function
             | [ x ] ->
                 leaf
                   (Ojax.Numpy.Reductions.sum
                      (base case names (with_x (List.hd (leaves x)))))
             | _ -> failwith "apply_transforms: grad arity")
           [ leaf (List.nth values xi) ])
  | "jvp" ->
      let x = List.nth values xi in
      let y, dy =
        A.jvp
          (function
            | [ x ] -> leaf (base case names (with_x (List.hd (leaves x))))
            | _ -> failwith "apply_transforms: jvp arity")
          [ leaf x ]
          [ leaf (Ojax.Numpy.Array_creation.ones_like x) ]
      in
      leaves y @ leaves dy
  | other -> failwith ("apply_transforms: transform " ^ other)

let references = Hashtbl.create 256

let load_references root =
  if Hashtbl.length references = 0 then
    let path =
      Filename.concat
        (Filename.concat (Filename.concat root name) "text")
        "reference.jsonl"
    in
    In_channel.with_open_text path (fun ic ->
        let rec go () =
          match In_channel.input_line ic with
          | None -> ()
          | Some line ->
              let row = Yojson.Safe.from_string line in
              Hashtbl.replace references
                (U.to_string (U.member "set" row)
                ^ "/"
                ^ U.to_string (U.member "case" row))
                row;
              go ()
        in
        go ())

let guarded f = try Ok (f ()) with e -> Error e

let check_text ~suite ~set ~root case names avals =
  load_references root;
  let case_id = U.to_string (U.member "case_id" case) in
  match Hashtbl.find_opt references (name ^ "/" ^ set ^ "/" ^ case_id) with
  | None -> Fuzz.log_row ~suite (case_id ^ ":text") 0 (-7) 0.0 "no_reference"
  | Some reference ->
      let shape = ref TU.Null_def in
      let lowered =
        guarded (fun () ->
            Ojax.Jaxpr.with_trace_debug
              {
                T.func_src_info = "f at apply:0";
                traced_for = "jit";
                arg_names =
                  Some
                    (List.mapi (fun i _ -> Printf.sprintf "values[%d]" i) names);
              }
              (fun () ->
                Ojax.Jaxpr.make_jaxpr avals (fun args ->
                    let out = program case names args in
                    shape :=
                      TU.tree_structure
                        (match out with
                        | [ one ] -> leaf one
                        | many -> TU.Tuple (List.map leaf many));
                    out)))
      in
      let render f =
        match lowered with
        | Error e -> Error e
        | Ok closed -> guarded (fun () -> f closed)
      in
      Compose_fuzz.text_row ~suite ~row:case_id reference "jaxpr"
        (render PP.closed_jaxpr_to_jax_string);
      Compose_fuzz.text_row ~suite ~row:case_id reference "text"
        (render (fun closed ->
             Ojax.Stablehlo.Emit.emit_jax_text
               ~result_names:(Ojax.Stages.result_names_of_tree !shape)
               ~name:"f" closed))

let check_case ~suite ~root ~set ~set_dir ~x64 ~nd_of_npz ~string_of_dtype
    ~concrete case () =
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let case_id = U.to_string (U.member "case_id" case) in
  let args = U.to_list (U.member "args" case) in
  let names = List.map (fun a -> U.to_string (U.member "name" a)) args in
  let b, table =
    Fuzz.bundle (Filename.concat (Filename.concat set_dir "ops") "index.npz")
  in
  let input arg =
    let m = Hashtbl.find table (case_id ^ "/in_" ^ arg) in
    let nd = nd_of_npz (Npz.decode b m.Fuzz.start m.Fuzz.descr m.Fuzz.shape) in
    T.Concrete
      (Ojax.Ndarray.canonicalize
         (Ojax.Dtypes.canonicalize_dtype (Ojax.Ndarray.dtype nd))
         nd)
  in
  let inputs = List.map input names in
  let str key =
    match U.member key case with `String n -> Some n | _ -> None
  in
  let jax_error key = Option.value (str key) ~default:"unknown" in
  Fuzz.check_mode ?jax_message:(str "eager_message") ~suite ~case_id
    ~mode:"eager" ~string_of_dtype ~concrete
    ~jax_error:(jax_error "eager_error") b table
    (Fuzz.output_names case "eager") (fun () -> program case names inputs);
  Fuzz.check_mode ?jax_message:(str "jit_message") ~suite ~case_id ~mode:"jit"
    ~string_of_dtype ~concrete ~jax_error:(jax_error "jit_error") b table
    (Fuzz.output_names case "jit") (fun () ->
      A.jit_flat ~name:"f" (fun a -> program case names a) inputs);
  check_text ~suite ~set ~root case names
    (List.map (fun v -> C.get_aval v) inputs)

let suites ~goldens_root ~nd_of_npz ~string_of_dtype ~concrete =
  List.filter_map
    (fun (set, x64) ->
      let set_dir = Filename.concat (Filename.concat goldens_root name) set in
      let path = Filename.concat set_dir "manifest.json" in
      if not (Sys.file_exists path) then None
      else
        let suite = name ^ ":" ^ set in
        let cases =
          Yojson.Safe.from_file path |> U.member "cases" |> U.to_list
        in
        Some
          ( suite,
            List.map
              (fun case ->
                Alcotest.test_case
                  (U.to_string (U.member "case_id" case))
                  `Quick
                  (check_case ~suite ~root:goldens_root ~set ~set_dir ~x64
                     ~nd_of_npz ~string_of_dtype ~concrete case))
              cases ))
    [ ("x64_off", false); ("x64_on", true) ]
