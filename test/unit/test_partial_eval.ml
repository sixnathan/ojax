module PE = Ojax.Interpreters.Partial_eval
module J = Ojax.Jaxpr
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let av shape dtype : T.aval = { shape; dtype; weak_type = false }
let b1 = C.bind1
let ndf = Nd.of_floats

let get v i =
  match v with
  | T.Concrete a -> Nd.get_f a [| i |]
  | _ -> Alcotest.fail "concrete"

let test_all_unknown () =
  let jaxpr, consts, pvals_out =
    PE.partial_eval_flat
      (fun args ->
        match args with
        | [ x ] -> [ b1 T.Sin [ b1 T.Cos [ x ] ] ]
        | _ -> assert false)
      [ PE.partial_val_unknown (av [| 3 |] D.F32) ]
  in
  Alcotest.(check bool) "unknown out" true (PE.is_unknown (List.hd pvals_out));
  Alcotest.(check int) "eqns" 2 (List.length jaxpr.T.eqns);
  Alcotest.(check int) "consts" 0 (List.length consts);
  let x = ndf D.F32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  match J.eval_jaxpr jaxpr (consts @ [ T.Concrete x ]) with
  | [ out ] ->
      List.iteri
        (fun i xi ->
          Alcotest.(check (float 1e-6))
            (Printf.sprintf "sin(cos)[%d]" i)
            (sin (cos xi))
            (get out i))
        [ 0.1; 0.2; 0.3 ]
  | _ -> Alcotest.fail "one output"

let test_mixed () =
  let y = ndf D.F32 [| 3 |] [| 2.0; 2.0; 2.0 |] in
  let jaxpr, consts, pvals_out =
    PE.partial_eval_flat
      (fun args ->
        match args with
        | [ x; y' ] -> [ b1 T.Add [ b1 T.Mul [ x; y' ]; b1 T.Sin [ y' ] ] ]
        | _ -> assert false)
      [
        PE.partial_val_unknown (av [| 3 |] D.F32);
        PE.partial_val_known (T.Concrete y);
      ]
  in
  Alcotest.(check bool) "unknown out" true (PE.is_unknown (List.hd pvals_out));
  let x = ndf D.F32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  match J.eval_jaxpr jaxpr (consts @ [ T.Concrete x ]) with
  | [ out ] ->
      List.iteri
        (fun i xi ->
          Alcotest.(check (float 1e-6))
            (Printf.sprintf "mixed[%d]" i)
            ((xi *. 2.0) +. sin 2.0)
            (get out i))
        [ 1.0; 2.0; 3.0 ]
  | _ -> Alcotest.fail "one output"

let test_all_known () =
  let x = ndf D.F32 [| 1 |] [| 0.5 |] in
  let jaxpr, consts, pvals_out =
    PE.partial_eval_flat
      (fun args ->
        match args with [ a ] -> [ b1 T.Sin [ a ] ] | _ -> assert false)
      [ PE.partial_val_known (T.Concrete x) ]
  in
  Alcotest.(check bool) "known out" true (PE.is_known (List.hd pvals_out));
  Alcotest.(check int) "no eqns" 0 (List.length jaxpr.T.eqns);
  Alcotest.(check int) "no consts" 0 (List.length consts);
  match (List.hd pvals_out).T.pv_const with
  | Some v -> Alcotest.(check (float 1e-6)) "sin 0.5" (sin 0.5) (get v 0)
  | None -> Alcotest.fail "expected const"

let test_shared_const () =
  let y = ndf D.F32 [| 2 |] [| 3.0; 4.0 |] in
  let jaxpr, consts, _ =
    PE.partial_eval_flat
      (fun args ->
        match args with
        | [ x; y' ] -> [ b1 T.Add [ b1 T.Mul [ x; y' ]; b1 T.Mul [ x; y' ] ] ]
        | _ -> assert false)
      [
        PE.partial_val_unknown (av [| 2 |] D.F32);
        PE.partial_val_known (T.Concrete y);
      ]
  in
  J.typecheck_jaxpr jaxpr;
  Alcotest.(check int) "dedup shared const" 1 (List.length consts);
  let x = ndf D.F32 [| 2 |] [| 1.0; 2.0 |] in
  match J.eval_jaxpr jaxpr (consts @ [ T.Concrete x ]) with
  | [ out ] ->
      Alcotest.(check (float 1e-6)) "s0" 6.0 (get out 0);
      Alcotest.(check (float 1e-6)) "s1" 16.0 (get out 1)
  | _ -> Alcotest.fail "one output"

let () =
  Alcotest.run "partial_eval"
    [
      ( "partial_eval",
        [
          Alcotest.test_case "all_unknown" `Quick test_all_unknown;
          Alcotest.test_case "mixed" `Quick test_mixed;
          Alcotest.test_case "all_known" `Quick test_all_known;
          Alcotest.test_case "shared_const" `Quick test_shared_const;
        ] );
    ]
