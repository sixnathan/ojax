module U = Yojson.Safe.Util
module Backend = Ojax.Backend
module J = Ojax.Jaxpr
module Core = Ojax.Core
module T = Ojax.Types
module Dt = Ojax.Dtype
module Nd = Ojax.Ndarray
module A = Ojax.Pjrt.Abi
module D = Ojax.Pjrt.Discover

let () = Ojax.Lax.install ()

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

let e2e_dir = Filename.concat goldens_root "e2e"

let plugin_present =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> true
  | _ -> false

let () = if plugin_present then Unix.putenv "OJAX_BACKEND" "xla"

let dtype_of_name = function
  | "float32" -> Dt.F32
  | "float64" -> Dt.F64
  | "int32" -> Dt.I32
  | "int64" -> Dt.I64
  | "bool" -> Dt.Bool
  | "uint32" -> Dt.Uint32
  | other -> failwith ("e2e: unsupported dtype " ^ other)

let bind1 = Core.bind1
let mul a b = bind1 T.Mul [ a; b ]
let add a b = bind1 T.Add [ a; b ]
let sub a b = bind1 T.Sub [ a; b ]

let programs : (string * (T.value list -> T.value list)) list =
  [
    ( "cubic_f32_4",
      fun a ->
        let x = List.nth a 0 in
        [ sub (mul x (mul x x)) x ] );
    ( "i32_poly_4",
      fun a ->
        let x = List.nth a 0 in
        [ sub (mul x x) x ] );
    ( "i32_two_4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        [ sub (mul x y) x ] );
    ( "select_min_f32_4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        let p = bind1 T.Lt [ x; y ] in
        [ bind1 T.Select_n [ p; y; x ] ] );
    ( "abs_add_sign_f32_4",
      fun a ->
        let x = List.nth a 0 in
        [ add (bind1 T.Abs [ x ]) (bind1 T.Sign [ x ]) ] );
    ( "broadcast_mul_f32_2x3",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        let bx =
          bind1
            (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] })
            [ x ]
        in
        [ mul bx y ] );
    ( "convert_trunc_add_i32_4",
      fun a ->
        let x = List.nth a 0 in
        let t = bind1 (T.Convert_element_type Dt.I32) [ x ] in
        [ add t t ] );
    ( "reshape_reduce_f32_2x3",
      fun a ->
        let x = List.nth a 0 in
        let r = bind1 (T.Reshape [| 6 |]) [ x ] in
        [ bind1 (T.Reduce_sum [| 0 |]) [ r ] ] );
    ( "minmax_chain_f32_4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        [ sub (bind1 T.Max [ x; y ]) (bind1 T.Min [ x; y ]) ] );
    ( "exp_neg_f32_4",
      fun a ->
        let x = List.nth a 0 in
        [ bind1 T.Exp [ bind1 T.Neg [ x ] ] ] );
    ( "sin_mul_f32_4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        [ mul (bind1 T.Sin [ x ]) y ] );
    ( "tanh_exp_sin_f32_4",
      fun a ->
        let x = List.nth a 0 in
        [ bind1 T.Tanh [ bind1 T.Exp [ bind1 T.Sin [ x ] ] ] ] );
    ( "sum_sin_f32_5",
      fun a ->
        let x = List.nth a 0 in
        [ bind1 (T.Reduce_sum [| 0 |]) [ bind1 T.Sin [ x ] ] ] );
    ( "cos_log_f32_4",
      fun a ->
        let x = List.nth a 0 in
        [ bind1 T.Cos [ bind1 T.Log [ x ] ] ] );
    ( "pow_f32_4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        [ bind1 T.Pow [ x; y ] ] );
    ( "matmul_f32_2x3x4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        [
          bind1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            [ x; y ];
        ] );
    ( "matmul_add_f32_2x4",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 and c = List.nth a 2 in
        let d =
          bind1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            [ x; y ]
        in
        let bc =
          bind1
            (T.Broadcast_in_dim { shape = [| 2; 4 |]; dims = [| 1 |] })
            [ c ]
        in
        [ add d bc ] );
    ( "reduce_matmul_f32",
      fun a ->
        let x = List.nth a 0 and y = List.nth a 1 in
        let d =
          bind1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            [ x; y ]
        in
        [ bind1 (T.Reduce_sum [| 0; 1 |]) [ d ] ] );
  ]

let unravel shape flat =
  let r = Array.length shape in
  let idx = Array.make r 0 in
  let rem = ref flat in
  for i = r - 1 downto 0 do
    idx.(i) <- (!rem mod if shape.(i) = 0 then 1 else shape.(i));
    rem := !rem / if shape.(i) = 0 then 1 else shape.(i)
  done;
  idx

let ndarray_of_npz (a : Npz.t) =
  let dtype = dtype_of_name a.dtype in
  let floats =
    match a.data with
    | Npz.F fs -> fs
    | Npz.I is -> Array.map Int64.to_float is
    | Npz.C _ -> failwith "e2e: complex input unsupported"
  in
  Nd.canonicalize dtype (Nd.of_floats dtype a.shape floats)

let npz_of_ndarray nd : Npz.t =
  let shape = Nd.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  let dtype = Nd.dtype nd in
  let dtype_name =
    match dtype with
    | Dt.F32 -> "float32"
    | Dt.F64 -> "float64"
    | Dt.I32 -> "int32"
    | Dt.I64 -> "int64"
    | Dt.Bool -> "bool"
    | Dt.Uint32 -> "uint32"
    | Dt.Complex64 -> "complex64"
    | Dt.Complex128 -> "complex128"
  in
  let data =
    match dtype with
    | Dt.F32 | Dt.F64 ->
        Npz.F (Array.init n (fun i -> Nd.get_f nd (unravel shape i)))
    | _ -> Npz.I (Array.init n (fun i -> Nd.get_i64 nd (unravel shape i)))
  in
  { Npz.dtype = dtype_name; shape; data }

type case = {
  case_id : string;
  compare : string;
  atol : float;
  rtol : float;
  args : T.aval list;
}

let to_shape j = Array.of_list (List.map U.to_int (U.to_list j))

let parse_case j =
  let tol = U.member "tol" j in
  let args =
    U.member "args" j |> U.to_list
    |> List.map (fun a ->
        {
          T.shape = U.member "shape" a |> to_shape;
          dtype = U.member "dtype" a |> U.to_string |> dtype_of_name;
          weak_type = false;
        })
  in
  {
    case_id = U.member "case_id" j |> U.to_string;
    compare = U.member "compare" j |> U.to_string;
    atol = U.member "atol" tol |> U.to_number;
    rtol = U.member "rtol" tol |> U.to_number;
    args;
  }

let load_manifest () =
  let path = Filename.concat e2e_dir "manifest.json" in
  let j = Yojson.Safe.from_file path in
  U.member "cases" j |> U.to_list |> List.map parse_case

let builder case_id =
  match List.assoc_opt case_id programs with
  | Some f -> f
  | None -> failwith ("e2e: no program for " ^ case_id)

let run_case c () =
  if not plugin_present then ()
  else begin
    let inputs =
      Npz.read (Filename.concat e2e_dir ("inputs/" ^ c.case_id ^ ".npz"))
    in
    let inputs = List.sort (fun (a, _) (b, _) -> String.compare a b) inputs in
    let in_nds = List.map (fun (_, a) -> ndarray_of_npz a) inputs in
    let cj = J.make_jaxpr c.args (builder c.case_id) in
    let outs =
      Backend.executor cj (List.map (fun nd -> T.Concrete nd) in_nds)
    in
    let out_nd =
      match outs with
      | [ T.Concrete nd ] -> nd
      | _ -> Alcotest.fail (c.case_id ^ ": expected single concrete output")
    in
    let expected =
      List.assoc "out0"
        (Npz.read (Filename.concat e2e_dir ("outputs/" ^ c.case_id ^ ".npz")))
    in
    let actual = npz_of_ndarray out_nd in
    Compare.check ~name:c.case_id ~compare:c.compare ~atol:c.atol ~rtol:c.rtol
      ~expected ~actual
  end

let host_single label c in_nds =
  match Backend.executor (J.make_jaxpr c.args (builder c.case_id)) in_nds with
  | [ T.Concrete nd ] -> nd
  | _ -> Alcotest.fail (c.case_id ^ ": " ^ label ^ " expected single concrete")

let run_resident c () =
  if not plugin_present then ()
  else begin
    let inputs =
      Npz.read (Filename.concat e2e_dir ("inputs/" ^ c.case_id ^ ".npz"))
    in
    let inputs = List.sort (fun (a, _) (b, _) -> String.compare a b) inputs in
    let in_nds = List.map (fun (_, a) -> ndarray_of_npz a) inputs in
    let host_out =
      host_single "host" c (List.map (fun nd -> T.Concrete nd) in_nds)
    in
    let dev_in =
      List.map (fun nd -> Backend.of_host_value (T.Concrete nd)) in_nds
    in
    let cj = J.make_jaxpr c.args (builder c.case_id) in
    let dev_out =
      match List.map Backend.to_host_value (Backend.executor cj dev_in) with
      | [ T.Concrete nd ] -> nd
      | _ -> Alcotest.fail (c.case_id ^ ": resident expected single concrete")
    in
    Compare.check ~name:(c.case_id ^ ":resident") ~compare:c.compare
      ~atol:c.atol ~rtol:c.rtol ~expected:(npz_of_ndarray host_out)
      ~actual:(npz_of_ndarray dev_out)
  end

let leak_smoke cases () =
  if not plugin_present then ()
  else begin
    let c = List.find (fun c -> c.case_id = "matmul_add_f32_2x4") cases in
    let inputs =
      Npz.read (Filename.concat e2e_dir ("inputs/" ^ c.case_id ^ ".npz"))
    in
    let inputs = List.sort (fun (a, _) (b, _) -> String.compare a b) inputs in
    let args = List.map (fun (_, a) -> T.Concrete (ndarray_of_npz a)) inputs in
    let cj = J.make_jaxpr c.args (builder c.case_id) in
    let run = Backend.executor cj in
    ignore (run args);
    Gc.full_major ();
    let before = A.maxrss_bytes () in
    for i = 1 to 1000 do
      ignore (run args);
      if i mod 100 = 0 then Gc.full_major ()
    done;
    Gc.full_major ();
    let after = A.maxrss_bytes () in
    let growth = after - before in
    Alcotest.(check bool)
      (Printf.sprintf "rss growth %d bytes bounded" growth)
      true
      (growth < 64 * 1024 * 1024)
  end

let () =
  let cases = load_manifest () in
  let corpus =
    List.map (fun c -> Alcotest.test_case c.case_id `Quick (run_case c)) cases
  in
  let resident =
    List.map
      (fun c -> Alcotest.test_case c.case_id `Quick (run_resident c))
      cases
  in
  Alcotest.run "e2e"
    [
      ("corpus", corpus);
      ("resident", resident);
      ( "leak",
        [ Alcotest.test_case "matmul_add 1000x" `Slow (leak_smoke cases) ] );
    ]
