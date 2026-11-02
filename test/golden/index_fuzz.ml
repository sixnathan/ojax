module U = Yojson.Safe.Util
module T = Ojax.Types
module C = Ojax.Core
module A = Ojax.Api
module G = Ojax.Numpy.Getitem
module JU = Ojax.Numpy.Jnp_util
module PP = Ojax.Pretty_printer

let name = "index_fuzz"
let opt_int = function `Null -> None | v -> Some (U.to_int v)

let scalar_value = function
  | `Int i -> JU.py_int i
  | `Float f -> JU.py_float f
  | `Intlit s -> JU.py_int (int_of_string s)
  | _ -> failwith "index_fuzz: scalar"

let host_array spec =
  let dt =
    Option.get (Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" spec)))
  in
  let rec flatten = function
    | `List l ->
        let parts = List.map flatten l in
        let shape =
          match parts with [] -> [ 0 ] | (s, _) :: _ -> List.length l :: s
        in
        (shape, List.concat_map snd parts)
    | `Bool b -> ([], [ (if b then 1.0 else 0.0) ])
    | `Int i -> ([], [ float_of_int i ])
    | `Float f -> ([], [ f ])
    | _ -> failwith "index_fuzz: const"
  in
  let data = U.member "data" spec in
  let shape, values = flatten data in
  let shape =
    match data with
    | `List [] -> [| 0 |]
    | _ -> Array.of_list (if values = [] then shape else shape)
  in
  T.Concrete (Ojax.Ndarray.of_floats dt shape (Array.of_list values))

let index_of env item =
  match U.to_string (U.member "k" item) with
  | "int" -> G.Int (U.to_int (U.member "v" item))
  | "slice" ->
      G.Slice
        {
          start = opt_int (U.member "start" item);
          stop = opt_int (U.member "stop" item);
          step = opt_int (U.member "step" item);
        }
  | "none" -> G.Newaxis
  | "ellipsis" -> G.Ellipsis
  | "bool" -> G.Bool (U.to_bool (U.member "v" item))
  | "arr" -> G.Arr (List.assoc (U.to_string (U.member "name" item)) env)
  | other -> failwith ("index_fuzz: index " ^ other)

let negative x = Ojax.Numpy.Ufuncs.negative x

let program case names values =
  let params = U.member "params" case in
  let env = List.combine names values in
  let env =
    match U.member "consts" params with
    | `Assoc l when U.member "const" params = `Bool true ->
        env @ List.map (fun (k, v) -> (k, host_array v)) l
    | _ -> env
  in
  let x = List.assoc "x" env in
  let idx = List.map (index_of env) (U.to_list (U.member "index" params)) in
  let flag key =
    match U.member key params with `Bool b -> Some b | _ -> None
  in
  let indices_are_sorted = flag "indices_are_sorted" in
  let unique_indices = flag "unique_indices" in
  let normalize_indices = flag "wrap_negative_indices" in
  let mode =
    match U.member "mode" params with `String m -> Some m | _ -> None
  in
  let y () =
    match U.member "y_scalar" params with
    | `Null -> List.assoc "y" env
    | v -> scalar_value v
  in
  let update f =
    f ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx (y ())
  in
  match U.to_string (U.member "op" params) with
  | "getitem" -> G.take x idx
  | "get" ->
      let fill_value =
        match U.member "fill_value" params with
        | `Float f -> Some (JU.py_float f)
        | `Int i -> Some (JU.py_int i)
        | _ -> None
      in
      G.take ?indices_are_sorted ?unique_indices ?mode ?fill_value
        ?normalize_indices x idx
  | "set" -> update G.at_set
  | "add" -> update G.at_add
  | "subtract" -> update G.at_subtract
  | "multiply" -> update G.at_multiply
  | "divide" -> update G.at_divide
  | "power" -> update G.at_power
  | "min" -> update G.at_min
  | "max" -> update G.at_max
  | ("ufunc_add_at" | "ufunc_multiply_at" | "ufunc_subtract_at") as op ->
      let inplace =
        match U.member "inplace" params with `Bool b -> b | _ -> false
      in
      let f =
        match op with
        | "ufunc_add_at" -> G.ufunc_add_at
        | "ufunc_multiply_at" -> G.ufunc_multiply_at
        | _ -> G.ufunc_subtract_at
      in
      f ~inplace x idx (y ())
  | "apply" ->
      G.at_apply ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x
        idx negative
  | other -> failwith ("index_fuzz: op " ^ other)

let references = Hashtbl.create 1024

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

let trace_debug names =
  {
    T.func_src_info = "f at index:0";
    traced_for = "jit";
    arg_names =
      Some (List.mapi (fun i _ -> Printf.sprintf "values[%d]" i) names);
  }

let guarded f = try Ok (f ()) with e -> Error e

let check_text ~suite ~set ~root case names avals =
  load_references root;
  let case_id = U.to_string (U.member "case_id" case) in
  match Hashtbl.find_opt references (name ^ "/" ^ set ^ "/" ^ case_id) with
  | None -> Fuzz.log_row ~suite (case_id ^ ":text") 0 (-7) 0.0 "no_reference"
  | Some reference ->
      let lowered =
        guarded (fun () ->
            Ojax.Jaxpr.with_trace_debug (trace_debug names) (fun () ->
                Ojax.Jaxpr.make_jaxpr avals (fun args ->
                    [ program case names args ])))
      in
      let x0 =
        List.hd avals |> fun _ ->
        T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [||] [| 0.0 |])
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
               ~result_names:
                 (Ojax.Stages.result_names_of_tree
                    (Ojax.Tree_util.tree_structure (Ojax.Tree_util.Leaf x0)))
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
    (Fuzz.output_names case "eager") (fun () -> [ program case names inputs ]);
  Fuzz.check_mode ?jax_message:(str "jit_message") ~suite ~case_id ~mode:"jit"
    ~string_of_dtype ~concrete ~jax_error:(jax_error "jit_error") b table
    (Fuzz.output_names case "jit") (fun () ->
      A.jit_flat (fun a -> [ program case names a ]) inputs);
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
