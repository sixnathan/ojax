module U = Yojson.Safe.Util

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type arg = { name : string; shape : int array; dtype : string }
type out = { oname : string; oshape : int array; odtype : string }

type case = {
  case_id : string;
  op : string;
  params : Yojson.Safe.t;
  compare : string;
  atol : float;
  rtol : float;
  args : arg list;
  outs : out list;
}

let to_shape j = Array.of_list (List.map U.to_int (U.to_list j))

let parse_arg j =
  {
    name = U.member "name" j |> U.to_string;
    shape = U.member "shape" j |> to_shape;
    dtype = U.member "dtype" j |> U.to_string;
  }

let parse_out j =
  {
    oname = U.member "name" j |> U.to_string;
    oshape = U.member "shape" j |> to_shape;
    odtype = U.member "dtype" j |> U.to_string;
  }

let parse_case j =
  let tol = U.member "tol" j in
  {
    case_id = U.member "case_id" j |> U.to_string;
    op = U.member "op" j |> U.to_string;
    params = U.member "params" j;
    compare = U.member "compare" j |> U.to_string;
    atol = U.member "atol" tol |> U.to_number;
    rtol = U.member "rtol" tol |> U.to_number;
    args = U.member "args" j |> U.to_list |> List.map parse_arg;
    outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
  }

let load_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_case )

let find_member members name =
  match List.assoc_opt name members with
  | Some a -> a
  | None -> failwith ("missing npz member " ^ name)

let check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.case_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.case_id ^ ".npz"))
  in
  List.iter
    (fun a ->
      let arr = find_member inputs a.name in
      if not (Compare.shapes_equal arr.Npz.shape a.shape) then
        Alcotest.failf "%s: input %s shape mismatch" c.case_id a.name;
      if canon arr.Npz.dtype <> canon a.dtype then
        Alcotest.failf "%s: input %s dtype %s != %s" c.case_id a.name
          arr.Npz.dtype a.dtype)
    c.args;
  List.iter
    (fun o ->
      let arr = find_member outputs o.oname in
      if not (Compare.shapes_equal arr.Npz.shape o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.case_id o.oname;
      if canon arr.Npz.dtype <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.case_id o.oname
          arr.Npz.dtype o.odtype;
      Compare.assert_tol o.odtype c.atol c.rtol;
      Compare.check
        ~name:(c.case_id ^ ":" ^ o.oname)
        ~compare:c.compare ~atol:c.atol ~rtol:c.rtol ~expected:arr ~actual:arr)
    c.outs

let dir_case_ids dir =
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun f -> Filename.check_suffix f ".npz")
  |> List.map Filename.remove_extension
  |> List.sort String.compare

let check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.case_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then begin
        let extra = List.filter (fun x -> not (List.mem x expected)) got in
        let missing = List.filter (fun x -> not (List.mem x got)) expected in
        Alcotest.failf "%s: coverage mismatch missing=[%s] extra=[%s]" sub
          (String.concat ";" missing)
          (String.concat ";" extra)
      end)
    [ "inputs"; "outputs" ]

let suite_for module_ set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root module_) set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  (module_ ^ ":" ^ set_name, coverage :: case_tests)

module Nd = Ojax.Ndarray
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype

let dtype_of_string = function
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "bool" -> D.Bool
  | s -> failwith ("lax golden: unsupported dtype " ^ s)

let string_of_dtype = function
  | D.F32 -> "float32"
  | D.F64 -> "float64"
  | D.I32 -> "int32"
  | D.I64 -> "int64"
  | D.Bool -> "bool"

let nd_of_npz (a : Npz.t) =
  let floats =
    match a.Npz.data with
    | Npz.F f -> f
    | Npz.I i -> Array.map Int64.to_float i
    | Npz.C _ -> failwith "lax golden: complex operand unsupported"
  in
  Nd.of_floats (dtype_of_string a.Npz.dtype) a.Npz.shape floats

let read_nd nd =
  let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
  let arr = Array.make n 0.0 in
  let _ =
    Nd.fold
      (fun i x ->
        arr.(i) <- x;
        i + 1)
      0 nd
  in
  arr

let ia j = Array.of_list (List.map U.to_int (U.to_list j))

let prim_of op params : T.primitive =
  let member name = U.member name params in
  match op with
  | "neg" -> T.Neg
  | "sin" -> T.Sin
  | "cos" -> T.Cos
  | "exp" -> T.Exp
  | "log" -> T.Log
  | "tanh" -> T.Tanh
  | "abs" -> T.Abs
  | "sign" -> T.Sign
  | "add" -> T.Add
  | "sub" -> T.Sub
  | "mul" -> T.Mul
  | "div" -> T.Div
  | "max" -> T.Max
  | "min" -> T.Min
  | "pow" -> T.Pow
  | "eq" -> T.Eq
  | "lt" -> T.Lt
  | "gt" -> T.Gt
  | "select_n" -> T.Select_n
  | "convert_element_type" ->
      T.Convert_element_type
        (dtype_of_string (U.to_string (member "new_dtype")))
  | "broadcast_in_dim" ->
      T.Broadcast_in_dim
        { shape = ia (member "shape"); dims = ia (member "dims") }
  | "reshape" -> T.Reshape (ia (member "new_sizes"))
  | "reduce_sum" -> T.Reduce_sum (ia (member "axes"))
  | "dot_general" ->
      let dn = U.to_list (member "dimension_numbers") in
      let contracting = U.to_list (List.nth dn 0) in
      let batch = U.to_list (List.nth dn 1) in
      T.Dot_general
        {
          lhs_contract = ia (List.nth contracting 0);
          rhs_contract = ia (List.nth contracting 1);
          lhs_batch = ia (List.nth batch 0);
          rhs_batch = ia (List.nth batch 1);
        }
  | _ -> failwith ("lax golden: unknown op " ^ op)

let concrete = function
  | T.Concrete n -> n
  | T.Tracer _ -> failwith "lax golden: expected concrete result"

let lax_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.case_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.case_id ^ ".npz"))
  in
  let operands =
    List.map
      (fun a -> T.Concrete (nd_of_npz (find_member inputs a.name)))
      c.args
  in
  let results = C.bind (prim_of c.op c.params) operands in
  let paired =
    try List.combine c.outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.case_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.case_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.case_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if c.compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol o.odtype c.atol c.rtol;
      Compare.check
        ~name:(c.case_id ^ ":" ^ o.oname)
        ~compare:c.compare ~atol:c.atol ~rtol:c.rtol ~expected:golden ~actual)
    paired

let lax_suite_for set_name =
  let set_dir = Filename.concat (Filename.concat goldens_root "lax") set_name in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (lax_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("lax:" ^ set_name, coverage :: case_tests)

module J = Ojax.Jaxpr
module PP = Ojax.Pretty_printer

let av shape dtype : T.aval = { shape; dtype; weak_type = false }
let jb1 = C.bind1
let scalar_f32 x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])

let jaxpr_builders :
    (string * T.aval list * (T.value list -> T.value list)) list =
  [
    ("sin", [ av [| 3 |] D.F32 ], fun a -> [ jb1 T.Sin a ]);
    ( "sin_mul",
      [ av [| 2; 3 |] D.F32; av [| 2; 3 |] D.F32 ],
      fun args ->
        match args with
        | [ x; y ] ->
            let c = jb1 T.Sin [ x ] in
            [ jb1 T.Mul [ c; y ] ]
        | _ -> assert false );
    ( "chain",
      [ av [| 4 |] D.F32 ],
      fun args ->
        match args with
        | [ x ] ->
            let n = jb1 T.Neg [ x ] in
            [ jb1 T.Exp [ n ] ]
        | _ -> assert false );
    ( "reduce",
      [ av [| 2; 3 |] D.F32 ],
      fun args -> [ jb1 (T.Reduce_sum [| 0 |]) args ] );
    ( "dot",
      [ av [| 2; 3 |] D.F32; av [| 3; 4 |] D.F32 ],
      fun args ->
        [
          jb1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            args;
        ] );
    ( "reshape",
      [ av [| 2; 3 |] D.F32 ],
      fun args -> [ jb1 (T.Reshape [| 6 |]) args ] );
    ( "broadcast",
      [ av [| 3 |] D.F32 ],
      fun args ->
        [ jb1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) args ]
    );
    ( "convert",
      [ av [| 3 |] D.F32 ],
      fun args -> [ jb1 (T.Convert_element_type D.I32) args ] );
    ( "compare",
      [ av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args -> [ jb1 T.Lt args ] );
    ( "select",
      [ av [| 3 |] D.Bool; av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args -> [ jb1 T.Select_n args ] );
    ("lit_mul", [], fun _ -> [ jb1 T.Mul [ scalar_f32 2.0; scalar_f32 3.0 ] ]);
    ( "nested",
      [ av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args ->
        match args with
        | [ x; y ] ->
            let m = jb1 T.Mul [ x; y ] in
            let s = jb1 T.Sin [ x ] in
            [ jb1 T.Add [ m; s ] ]
        | _ -> assert false );
  ]

let load_jaxpr_manifest path =
  let j = Yojson.Safe.from_file path in
  U.member "cases" j |> U.to_list
  |> List.map (fun c ->
      (U.member "case_id" c |> U.to_string, U.member "text" c |> U.to_string))

let jaxpr_check_case want (_case_id, avals, f) () =
  let got = PP.closed_jaxpr_to_string (J.make_jaxpr avals f) in
  Alcotest.(check string) "jaxpr text" want got

let jaxpr_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "jaxpr") set_name
  in
  let cases = load_jaxpr_manifest (Filename.concat set_dir "manifest.json") in
  let builder_ids =
    List.map (fun (id, _, _) -> id) jaxpr_builders |> List.sort String.compare
  in
  let manifest_ids = List.map fst cases |> List.sort String.compare in
  let coverage () =
    if builder_ids <> manifest_ids then
      Alcotest.failf "jaxpr:%s coverage mismatch" set_name
  in
  let case_tests =
    List.map
      (fun (case_id, want) ->
        let builder =
          List.find (fun (id, _, _) -> id = case_id) jaxpr_builders
        in
        Alcotest.test_case case_id `Quick (jaxpr_check_case want builder))
      cases
  in
  ( "jaxpr:" ^ set_name,
    Alcotest.test_case "coverage" `Quick coverage :: case_tests )

let must_fail msg f =
  match f () with
  | () -> Alcotest.failf "expected failure: %s" msg
  | exception Failure _ -> ()

let f32 xs =
  { Npz.dtype = "float32"; shape = [| Array.length xs |]; data = Npz.F xs }

let i32 xs =
  { Npz.dtype = "int32"; shape = [| Array.length xs |]; data = Npz.I xs }

let chk ~compare ~atol ~rtol a b =
  Compare.check ~name:"t" ~compare ~atol ~rtol ~expected:a ~actual:b

let compare_tests () =
  chk ~compare:"allclose" ~atol:1e-6 ~rtol:1e-6
    (f32 [| 1.0; 2.0 |])
    (f32 [| 1.0; 2.0000001 |]);
  must_fail "beyond tol" (fun () ->
      chk ~compare:"allclose" ~atol:1e-6 ~rtol:1e-6 (f32 [| 1.0 |])
        (f32 [| 1.1 |]));
  chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.nan |])
    (f32 [| Float.nan |]);
  must_fail "nan vs value" (fun () ->
      chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.nan |])
        (f32 [| 1.0 |]));
  chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.infinity |])
    (f32 [| Float.infinity |]);
  must_fail "inf sign" (fun () ->
      chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.infinity |])
        (f32 [| Float.neg_infinity |]));
  chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L; 2L |]) (i32 [| 1L; 2L |]);
  must_fail "int mismatch" (fun () ->
      chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L |]) (i32 [| 2L |]));
  must_fail "shape mismatch" (fun () ->
      chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L |]) (i32 [| 1L; 1L |]));
  must_fail "tol table" (fun () -> Compare.assert_tol "float32" 1e-3 1e-3)

let () =
  Ojax.Lax.install ();
  Alcotest.run "goldens"
    [
      suite_for "dtypes" "x64_off";
      suite_for "dtypes" "x64_on";
      lax_suite_for "x64_off";
      lax_suite_for "x64_on";
      jaxpr_suite_for "x64_off";
      jaxpr_suite_for "x64_on";
      ("compare", [ Alcotest.test_case "semantics" `Quick compare_tests ]);
    ]
