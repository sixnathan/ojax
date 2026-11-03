module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module S = Ojax.Sharding
module Tr = Ojax.Tree_util
module UF = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module Nd = Ojax.Ndarray

let name = "mesh_one"
let mesh = lazy (S.make_mesh ~axis_types:[ S.Auto ] [ 1 ] [ "x" ])
let row_spec = [ S.Named "x"; S.Unconstrained ]
let col_spec = [ S.Unconstrained; S.Named "x" ]
let sharding spec = S.named_sharding (Lazy.force mesh) spec
let scalar x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])
let base f = T.Concrete (Nd.of_floats D.F32 [| 4; 4 |] (Array.init 16 f))
let leaf = function Tr.Leaf v -> v | _ -> failwith "mesh_one: expected a leaf"
let placed spec v = leaf (S.device_put (Tr.Leaf v) (sharding spec))

let arg values =
  let v i = match values with Some a -> a.(i) | None -> float_of_int i in
  function
  | "row" -> placed row_spec (base v)
  | "col" ->
      placed col_spec
        (base (fun i -> v (((3 - (i / 4)) * 4) + (i mod 4)) *. 0.5))
  | "host" -> base (fun i -> v i +. 3.0)
  | k -> failwith ("mesh_one: arg " ^ k)

let body x = UF.add (UF.multiply (UF.sin x) (scalar 2.0)) (R.sum x)

let smap collective x =
  leaf
    (S.shard_map ~mesh:(Lazy.force mesh) ~in_specs:[ [ S.Named "x" ] ]
       ~out_specs:[ [ S.Named "x" ] ]
       (function
         | [ Tr.Leaf y ] -> Tr.Leaf (UF.add (UF.sin y) (collective y))
         | _ -> failwith "mesh_one: arity")
       [ Tr.Leaf x ])

let unary f = function [ x ] -> [ f x ] | _ -> failwith "mesh_one: arity"

let binary f = function
  | [ x; y ] -> [ f x y ]
  | _ -> failwith "mesh_one: arity"

let cases =
  [
    ("sin_sum", (unary body, None));
    ("out_sharded", (unary body, Some [ sharding row_spec ]));
    ("out_col", (unary body, Some [ sharding col_spec ]));
    ( "two_placed",
      (binary (fun x y -> UF.add (UF.multiply x y) (scalar 1.0)), None) );
    ( "one_placed",
      (binary (fun x y -> UF.add (UF.multiply x y) (scalar 1.0)), None) );
    ( "constraint",
      ( unary (fun x ->
            leaf
              (S.with_sharding_constraint
                 (Tr.Leaf (UF.multiply x (scalar 2.0)))
                 (sharding col_spec))),
        None ) );
    ( "shard_map_psum",
      (unary (smap (fun y -> Ojax.Parallel.psum y ~axis_name:"x")), None) );
    ( "shard_map_pmax",
      (unary (smap (fun y -> Ojax.Parallel.pmax y ~axis_name:"x")), None) );
    ( "shard_map_pmin",
      (unary (smap (fun y -> Ojax.Parallel.pmin y ~axis_name:"x")), None) );
  ]

let tree f args =
  match f (List.map leaf args) with
  | [ v ] -> Tr.Leaf v
  | vs -> Tr.Tuple (List.map (fun v -> Tr.Leaf v) vs)

let host v =
  match Ojax.Backend.to_host_value v with
  | T.Concrete nd -> nd
  | _ -> failwith "mesh_one: expected a host array"

let out_text v =
  let nd = host v in
  let raw = Npz.raw_of_nd nd in
  raw.Npz.raw_dtype ^ "["
  ^ String.concat "," (Array.to_list (Array.map string_of_int (Nd.shape nd)))
  ^ "]"
  ^ Audit_holes.hex raw.Npz.bytes

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log name jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(String.escaped jax ^ "\t" ^ String.escaped ojax)
      ~suite name 1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical else "differ")
  in
  let attempt f = try f () with e -> Audit_holes.error_detail e in
  let field f = Option.value (Audit_holes.str_opt row f) ~default:"" in
  match List.assoc_opt id cases with
  | None -> log id "binding" "no_binding"
  | Some (f, out_shardings) ->
      let values =
        match U.member "base" row with
        | `Null -> None
        | j -> Some (Array.of_list (List.map U.to_number (U.to_list j)))
      in
      let args =
        List.map
          (fun k -> arg values (U.to_string k))
          (U.to_list (U.member "args" row))
      in
      let trees = List.map (fun v -> Tr.Leaf v) args in
      log (id ^ ":hlo") (field "hlo")
        (attempt (fun () ->
             S.lower_text ?out_shardings ~name:"f" (tree f) trees));
      log (id ^ ":jaxpr") (field "jaxpr")
        (attempt (fun () ->
             Ojax.Pretty_printer.closed_jaxpr_to_jax_string
               (Ojax.Jaxpr.make_jaxpr (List.map Ojax.Core.get_aval args) f)));
      let jax =
        List.map Audit_holes.jax_out (U.to_list (U.member "outs" row))
      in
      let ours =
        try
          Ok
            (List.map out_text
               (Tr.tree_leaves (S.jit ?out_shardings (tree f) trees)))
        with e -> Error (Audit_holes.error_detail e)
      in
      List.iteri
        (fun i j ->
          log
            (Printf.sprintf "%s:out%d" id i)
            j
            (match ours with
            | Ok l -> Option.value (List.nth_opt l i) ~default:"missing"
            | Error e -> e))
        jax

let path ~goldens_root =
  Filename.concat (Filename.concat goldens_root name) "cases.jsonl"

let suites ~goldens_root =
  if not (Sys.file_exists (path ~goldens_root)) then []
  else
    let rows = Audit_holes.read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              let x64 = U.to_bool (U.member "x64" row) in
              if mode = "x64_on" = x64 then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
