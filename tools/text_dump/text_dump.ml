module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module C = Ojax.Core
module PP = Ojax.Pretty_printer

let dtype_of_name = function
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "bool" -> D.Bool
  | "uint32" -> D.Uint32
  | "complex64" -> D.Complex64
  | "complex128" -> D.Complex128
  | "int8" -> D.I8
  | "int16" -> D.I16
  | "uint8" -> D.Uint8
  | "uint16" -> D.Uint16
  | "uint64" -> D.Uint64
  | "float16" -> D.F16
  | "bfloat16" -> D.Bf16
  | other -> failwith ("text_dump: dtype " ^ other)

let aval_of_arg a : T.aval =
  {
    shape = Array.of_list (List.map U.to_int (U.to_list (U.member "shape" a)));
    dtype = dtype_of_name (U.to_string (U.member "dtype" a));
    weak_type = false;
  }

let guarded f =
  try `Ok (f ()) with
  | Stack_overflow -> `Err "Stack_overflow"
  | Out_of_memory -> `Err "Out_of_memory"
  | e -> (
      match Ojax.Errors.message_of e with
      | Some m -> `Err m
      | None -> `Err (Printexc.to_string e))

let field name = function
  | `Ok s -> [ (name, `String s) ]
  | `Err e -> [ (name ^ "_error", `String e) ]

let record oc ~set ~case traced =
  let lowered = guarded traced in
  let render f =
    match lowered with
    | `Err e -> `Err e
    | `Ok closed -> guarded (fun () -> f closed)
  in
  let fields =
    field "jaxpr" (render PP.closed_jaxpr_to_jax_string)
    @ field "structure" (render PP.primitive_structure)
    @ field "text" (render (Ojax.Stablehlo.Emit.emit_jax_text ~name:"f"))
  in
  Yojson.Safe.to_channel oc
    (`Assoc ([ ("set", `String set); ("case", `String case) ] @ fields));
  output_char oc '\n';
  flush oc

let traced_debug n =
  {
    T.func_src_info = Printf.sprintf "f at %s:%d" __FILE__ __LINE__;
    traced_for = "jit";
    arg_names = Some (List.init n (Printf.sprintf "xs[%d]"));
  }

let parity_case oc ~set case =
  let names =
    List.map
      (fun a -> U.to_string (U.member "name" a))
      (U.to_list (U.member "args" case))
  in
  let avals = List.map aval_of_arg (U.to_list (U.member "args" case)) in
  let op = U.to_string (U.member "op" case) in
  let params = U.member "params" case in
  let lookup args name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> failwith ("text_dump: missing input " ^ name)
    in
    go names args
  in
  record oc ~set
    ~case:(U.to_string (U.member "case_id" case))
    (fun () ->
      Ojax.Jaxpr.with_trace_debug
        (traced_debug (List.length avals))
        (fun () ->
          Ojax.Jaxpr.make_jaxpr avals (fun args ->
              Parity_cases.run op params (lookup args))))

let f32 shape : T.aval = { shape; dtype = D.F32; weak_type = false }
let of_dtype dtype shape : T.aval = { shape; dtype; weak_type = false }
let scalar_f32 v = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| v |])
let one prim args = List.hd (C.bind prim args)

let jaxpr_builders =
  [
    ("sin", [ f32 [| 3 |] ], fun a -> [ one T.Sin a ]);
    ( "sin_mul",
      [ f32 [| 2; 3 |]; f32 [| 2; 3 |] ],
      function
      | [ x; y ] -> [ one T.Mul [ one T.Sin [ x ]; y ] ]
      | _ -> assert false );
    ("chain", [ f32 [| 4 |] ], fun a -> [ one T.Exp [ one T.Neg a ] ]);
    ("reduce", [ f32 [| 2; 3 |] ], fun a -> [ one (T.Reduce_sum [| 0 |]) a ]);
    ( "dot",
      [ f32 [| 2; 3 |]; f32 [| 3; 4 |] ],
      fun a ->
        [
          one
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
                 untyped_dot = true;
                 dot_precision = None;
               })
            a;
        ] );
    ("reshape", [ f32 [| 2; 3 |] ], fun a -> [ one (T.Reshape [| 6 |]) a ]);
    ( "broadcast",
      [ f32 [| 3 |] ],
      fun a ->
        [ one (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) a ] );
    ( "convert",
      [ f32 [| 3 |] ],
      fun a -> [ one (T.Convert_element_type D.I32) a ] );
    ("compare", [ f32 [| 3 |]; f32 [| 3 |] ], fun a -> [ one T.Lt a ]);
    ( "select",
      [ of_dtype D.Bool [| 3 |]; f32 [| 3 |]; f32 [| 3 |] ],
      fun a -> [ one T.Select_n a ] );
    ("lit_mul", [], fun _ -> [ one T.Mul [ scalar_f32 2.0; scalar_f32 3.0 ] ]);
    ( "nested",
      [ f32 [| 3 |]; f32 [| 3 |] ],
      function
      | [ x; y ] ->
          let m = one T.Mul [ x; y ] in
          let d = one T.Sin [ x ] in
          [ one T.Add [ m; d ] ]
      | _ -> assert false );
  ]

let cases_of path = U.to_list (U.member "cases" (Yojson.Safe.from_file path))
let in_shard ~shard ~shards i = shards <= 1 || i mod shards = shard

let nd_of_npz (a : Npz.t) =
  let dt = dtype_of_name a.Npz.dtype in
  match a.Npz.data with
  | Npz.C c -> Ojax.Ndarray.of_complex dt a.Npz.shape c
  | Npz.F f -> Ojax.Ndarray.of_floats dt a.Npz.shape f
  | Npz.I i ->
      Ojax.Ndarray.of_floats dt a.Npz.shape (Array.map Int64.to_float i)

let dump_arrays goldens out =
  let oc = open_out out in
  let set_dir = Filename.concat (Filename.concat goldens "parity") "x64_off" in
  List.iter
    (fun case ->
      let id = U.to_string (U.member "case_id" case) in
      let path =
        Filename.concat (Filename.concat set_dir "inputs") (id ^ ".npz")
      in
      List.iter
        (fun (name, arr) ->
          let row =
            try
              let v = T.Concrete (nd_of_npz arr) in
              [
                ("repr", `String (Ojax.Numpy.Extras.array_repr v));
                ("str", `String (Ojax.Numpy.Extras.array_str_jax v));
              ]
            with e -> [ ("error", `String (Printexc.to_string e)) ]
          in
          Yojson.Safe.to_channel oc
            (`Assoc ([ ("case", `String id); ("arg", `String name) ] @ row));
          output_char oc '\n')
        (Npz.read path))
    (cases_of (Filename.concat set_dir "manifest.json"));
  close_out oc

let () =
  Ojax.Lax.install ();
  Ojax.Random.Prng.install ();
  Ojax.Random.Threefry.install ();
  let goldens = Sys.argv.(1) in
  let out = Sys.argv.(2) in
  if Sys.argv.(3) = "arrays" then begin
    dump_arrays goldens out;
    exit 0
  end;
  let shard = int_of_string Sys.argv.(3) in
  let shards = int_of_string Sys.argv.(4) in
  let oc = open_out out in
  List.iter
    (fun (set, x64) ->
      Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
          let manifest =
            Filename.concat
              (Filename.concat (Filename.concat goldens "parity") set)
              "manifest.json"
          in
          List.iteri
            (fun i case ->
              if in_shard ~shard ~shards i then
                parity_case oc ~set:("parity/" ^ set) case)
            (cases_of manifest);
          if shard = 0 then
            List.iter
              (fun (case, avals, f) ->
                record oc ~set:("jaxpr/" ^ set) ~case (fun () ->
                    Ojax.Jaxpr.make_jaxpr avals f))
              jaxpr_builders))
    [ ("x64_off", false); ("x64_on", true) ];
  close_out oc
