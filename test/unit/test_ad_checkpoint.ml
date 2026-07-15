module Ac = Ojax.Ad_checkpoint
module Ad = Ojax.Interpreters.Ad
module B = Ojax.Interpreters.Batching
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let b1 = C.bind1
let scalar x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])
let vector xs = T.Concrete (Nd.of_floats D.F32 [| Array.length xs |] xs)

let concrete = function
  | T.Concrete nd -> nd
  | _ -> Alcotest.fail "expected concrete value"

let get0 v = Nd.get_f (concrete v) [||]
let close a b = Float.abs (a -. b) <= 1e-5

let sin_sin args =
  match args with
  | [ x ] -> [ b1 T.Sin [ b1 T.Sin [ x ] ] ]
  | _ -> Alcotest.fail "sin_sin arity"

let cubic args =
  match args with
  | [ x ] -> [ b1 T.Sub [ b1 T.Mul [ x; b1 T.Mul [ x; x ] ]; x ] ]
  | _ -> Alcotest.fail "cubic arity"

let grad fn x = List.hd (Ad.grad (fun xs -> List.hd (fn xs)) [ scalar x ])

let test_forward_and_alias () =
  let x = scalar 0.7 in
  let want = get0 (List.hd (sin_sin [ x ])) in
  let got = get0 (List.hd (Ac.checkpoint sin_sin [ x ])) in
  let alias = get0 (List.hd (Ac.remat sin_sin [ x ])) in
  Alcotest.(check bool) "checkpoint" true (close got want);
  Alcotest.(check bool) "remat" true (close alias want)

let test_grad_policies () =
  let want = get0 (grad sin_sin 0.7) in
  let policies =
    [
      Ac.nothing_saveable;
      Ac.everything_saveable;
      Ac.checkpoint_dots;
      Ac.dots_with_no_batch_dims_saveable;
      Ac.save_only_these_names [ "mid" ];
    ]
  in
  List.iteri
    (fun i policy ->
      let fn = Ac.checkpoint ~policy sin_sin in
      Alcotest.(check bool)
        (Printf.sprintf "policy %d" i)
        true
        (close (get0 (grad fn 0.7)) want))
    policies

let test_direct_jvp () =
  let x = 0.7 in
  let po, to_ = Ad.jvp (Ac.checkpoint sin_sin) [ scalar x ] [ scalar 1.0 ] in
  let want = cos (sin x) *. cos x in
  Alcotest.(check bool) "primal" true (close (get0 (List.hd po)) (sin (sin x)));
  Alcotest.(check bool) "tangent" true (close (get0 (List.hd to_)) want)

let test_nested_grad () =
  let nested = Ac.checkpoint (Ac.checkpoint cubic) in
  let x = 0.4 in
  let want = (3.0 *. x *. x) -. 1.0 in
  Alcotest.(check bool) "nested" true (close (get0 (grad nested x)) want)

let test_vmap () =
  let x = vector [| 0.1; 0.2; 0.3 |] in
  let out = List.hd (B.vmap (Ac.checkpoint sin_sin) [ Some 0 ] [ x ]) in
  Array.iteri
    (fun i v ->
      Alcotest.(check bool)
        (Printf.sprintf "vmap %d" i)
        true
        (close (Nd.get_f (concrete out) [| i |]) (sin (sin v))))
    [| 0.1; 0.2; 0.3 |]

let test_jit_grad () =
  let fn args = Ad.grad (fun xs -> List.hd (Ac.checkpoint sin_sin xs)) args in
  let got = List.hd (Ojax.Api.jit_flat fn [ scalar 0.7 ]) in
  let want = cos (sin 0.7) *. cos 0.7 in
  Alcotest.(check bool) "jit grad" true (close (get0 got) want)

let test_saved_residuals () =
  let x = scalar 0.7 in
  let plain = Ac.saved_residuals sin_sin [ x ] in
  let remat = Ac.saved_residuals (Ac.checkpoint sin_sin) [ x ] in
  let saved =
    Ac.saved_residuals
      (Ac.checkpoint ~policy:Ac.everything_saveable sin_sin)
      [ x ]
  in
  Alcotest.(check int) "plain" 2 (List.length plain);
  Alcotest.(check int) "remat" 1 (List.length remat);
  Alcotest.(check int) "everything" 2 (List.length saved)

let test_policy_predicates () =
  let dims =
    {
      T.lhs_contract = [| 1 |];
      rhs_contract = [| 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  Alcotest.(check bool)
    "everything" true
    (Ac.policy_allows Ac.everything_saveable T.Sin);
  Alcotest.(check bool)
    "nothing" false
    (Ac.policy_allows Ac.nothing_saveable T.Sin);
  Alcotest.(check bool)
    "dot" true
    (Ac.policy_allows Ac.checkpoint_dots (T.Dot_general dims));
  Alcotest.(check bool)
    "dot no batch" true
    (Ac.policy_allows Ac.dots_with_no_batch_dims_saveable (T.Dot_general dims));
  Alcotest.(check bool)
    "named" true
    (Ac.policy_allows (Ac.save_only_these_names [ "mid" ]) ~name:"mid" T.Sin);
  Alcotest.(check bool)
    "unnamed" false
    (Ac.policy_allows (Ac.save_only_these_names [ "mid" ]) T.Sin);
  let named = Ac.checkpoint_name (scalar 2.0) "mid" in
  Alcotest.(check bool) "name identity" true (close (get0 named) 2.0)

let () =
  Alcotest.run "ad_checkpoint"
    [
      ( "checkpoint",
        [
          Alcotest.test_case "forward and alias" `Quick test_forward_and_alias;
          Alcotest.test_case "grad policies" `Quick test_grad_policies;
          Alcotest.test_case "direct jvp" `Quick test_direct_jvp;
          Alcotest.test_case "nested grad" `Quick test_nested_grad;
          Alcotest.test_case "vmap" `Quick test_vmap;
          Alcotest.test_case "jit grad" `Quick test_jit_grad;
        ] );
      ("residuals", [ Alcotest.test_case "counts" `Quick test_saved_residuals ]);
      ( "policies",
        [ Alcotest.test_case "predicates" `Quick test_policy_predicates ] );
    ]
