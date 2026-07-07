module U = Yojson.Safe.Util

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type arg = { name : string; shape : int array; dtype : string }
type out = { oname : string; oshape : int array; odtype : string }

type case = {
  case_id : string;
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
  Alcotest.run "goldens"
    [
      suite_for "dtypes" "x64_off";
      suite_for "dtypes" "x64_on";
      ("compare", [ Alcotest.test_case "semantics" `Quick compare_tests ]);
    ]
