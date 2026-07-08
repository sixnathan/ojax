module J = Ojax.Jaxpr
module PP = Ojax.Pretty_printer
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let av shape dtype : T.aval = { shape; dtype; weak_type = false }
let b1 = C.bind1

let test_render_smoke () =
  let cj = J.make_jaxpr [ av [| 3 |] D.F32 ] (fun a -> [ b1 T.Sin a ]) in
  Alcotest.(check string)
    "sin" "{ lambda a:f32[3] . let b:f32[3] = sin a in ( b ) }"
    (PP.closed_jaxpr_to_string cj)

let test_encode_var () =
  let cases = [ (0, "a"); (25, "z"); (26, "aa"); (27, "ab"); (701, "zz") ] in
  List.iter
    (fun (n, s) ->
      Alcotest.(check string) (string_of_int n) s (PP.encode_var n))
    cases

let test_repr_float () =
  let cases =
    [
      (2.0, "2.0");
      (0.5, "0.5");
      (3.0, "3.0");
      (-1.5, "-1.5");
      (0.1, "0.1");
      (10.0, "10.0");
      (20.0, "20.0");
      (100.0, "100.0");
      (1000.0, "1000.0");
      (1e16, "1e+16");
      (1e-5, "1e-05");
    ]
  in
  List.iter
    (fun (x, s) -> Alcotest.(check string) s s (PP.python_repr_float x))
    cases

let test_eval () =
  let cj =
    J.make_jaxpr
      [ av [| 3 |] D.F32; av [| 3 |] D.F32 ]
      (fun args ->
        match args with
        | [ x; y ] ->
            let m = b1 T.Mul [ x; y ] in
            [ b1 T.Add [ m; x ] ]
        | _ -> assert false)
  in
  let x = Nd.of_floats D.F32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let y = Nd.of_floats D.F32 [| 3 |] [| 4.0; 5.0; 6.0 |] in
  let outs = J.eval_closed_jaxpr cj [ T.Concrete x; T.Concrete y ] in
  match outs with
  | [ T.Concrete out ] ->
      Array.iteri
        (fun i e ->
          Alcotest.(check (float 1e-6))
            (Printf.sprintf "eval[%d]" i)
            e (Nd.get_f out [| i |]))
        [| 5.0; 12.0; 21.0 |]
  | _ -> Alcotest.fail "eval: expected one concrete output"

let test_typecheck_ok () =
  let cj = J.make_jaxpr [ av [| 3 |] D.F32 ] (fun a -> [ b1 T.Sin a ]) in
  J.typecheck_jaxpr cj.T.jaxpr

let test_typecheck_bad () =
  let good = J.make_jaxpr [ av [| 3 |] D.F32 ] (fun a -> [ b1 T.Sin a ]) in
  let jx = good.T.jaxpr in
  let bad_eqn =
    {
      (List.hd jx.T.eqns) with
      T.outs = [ { T.vid = 999; vaval = av [| 5 |] D.F32 } ];
    }
  in
  let bad = { jx with T.eqns = [ bad_eqn ] } in
  match J.typecheck_jaxpr bad with
  | () -> Alcotest.fail "typecheck: expected failure"
  | exception Invalid_argument _ -> ()

let () =
  Ojax.Lax.install ();
  Alcotest.run "jaxpr"
    [
      ( "jaxpr",
        [
          Alcotest.test_case "render_smoke" `Quick test_render_smoke;
          Alcotest.test_case "encode_var" `Quick test_encode_var;
          Alcotest.test_case "repr_float" `Quick test_repr_float;
          Alcotest.test_case "eval" `Quick test_eval;
          Alcotest.test_case "typecheck_ok" `Quick test_typecheck_ok;
          Alcotest.test_case "typecheck_bad" `Quick test_typecheck_bad;
        ] );
    ]
