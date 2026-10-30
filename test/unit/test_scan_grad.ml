open Ojax
open Ojax.Types
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching
module Loops = Ojax.Lax

let () = Lax.install ()
let scalar x = Concrete (Ndarray.of_floats Dtype.F32 [||] [| x |])
let vec xs = Concrete (Ndarray.of_floats Dtype.F32 [| Array.length xs |] xs)

let to_nd = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected a concrete value"

let value v = Ndarray.get_f (to_nd v) [||]
let at v i = Ndarray.get_f (to_nd v) [| i |]

let check name expected actual =
  Alcotest.(check (float 1e-4)) name expected actual

let mul a b = Core.bind1 Mul [ a; b ]
let add a b = Core.bind1 Add [ a; b ]
let sin_ a = Core.bind1 Sin [ a ]
let sole = function [ x ] -> x | _ -> Alcotest.fail "expected one value"
let grad_of f args = Ad.grad f args

let product_scan args =
  match args with
  | [ c0; xs ] -> (
      match
        Loops.scan
          (function [ c; x ] -> [ mul c x ] | _ -> Alcotest.fail "arity")
          [ c0 ] [ xs ]
      with
      | [ c ] -> c
      | _ -> Alcotest.fail "arity")
  | _ -> Alcotest.fail "arity"

let grad_carry_and_xs () =
  match grad_of product_scan [ scalar 2.0; vec [| 3.0; 4.0 |] ] with
  | [ gc; gx ] ->
      check "d/dc" 12.0 (value gc);
      check "d/dx0" 8.0 (at gx 0);
      check "d/dx1" 6.0 (at gx 1)
  | _ -> Alcotest.fail "arity"

let nonlinear_with_ys args =
  let xs = sole args in
  match
    Loops.scan
      (function
        | [ c; x ] ->
            let c' = add c (sin_ x) in
            [ c'; mul c' x ]
        | _ -> Alcotest.fail "arity")
      [ scalar 1.0 ]
      [ xs ]
  with
  | [ _; ys ] -> Core.bind1 (Reduce_sum [| 0 |]) [ ys ]
  | _ -> Alcotest.fail "arity"

let grad_nonlinear_ys () =
  let xs = [| 0.5; 1.0; 1.5 |] in
  let g = sole (grad_of nonlinear_with_ys [ vec xs ]) in
  let partial i =
    let c = ref 1.0 in
    let c_i = ref 0.0 in
    let tail = ref 0.0 in
    Array.iteri
      (fun j x ->
        c := !c +. sin x;
        if j = i then c_i := !c;
        if j >= i then tail := !tail +. x)
      xs;
    !c_i +. (cos xs.(i) *. !tail)
  in
  Array.iteri
    (fun i _ -> check "matches analytic derivative" (partial i) (at g i))
    xs

let grad_reverse_scan () =
  let f args =
    let xs = sole args in
    match
      Loops.scan ~reverse:true
        (function
          | [ c; x ] -> [ add (mul c c) x ] | _ -> Alcotest.fail "arity")
        [ scalar 1.0 ]
        [ xs ]
    with
    | [ c ] -> c
    | _ -> Alcotest.fail "arity"
  in
  let xs = [| 0.5; 0.25 |] in
  let g = sole (grad_of f [ vec xs ]) in
  check "reverse x1 is applied first" (2.0 *. 1.25) (at g 1);
  check "reverse x0 is applied last" 1.0 (at g 0)

let second_order () =
  let f args =
    let x = sole args in
    let xs = Core.bind1 (Stack 0) [ x; x ] in
    match
      Loops.scan
        (function [ c; x ] -> [ mul c x ] | _ -> Alcotest.fail "arity")
        [ scalar 1.0 ]
        [ xs ]
    with
    | [ c ] -> c
    | _ -> Alcotest.fail "arity"
  in
  let g args = sole (grad_of f args) in
  check "first" 6.0 (value (g [ scalar 3.0 ]));
  check "second" 2.0 (value (sole (grad_of g [ scalar 3.0 ])))

let length_without_xs () =
  let run args =
    match
      Loops.scan ~length:3
        (function [ c ] -> [ mul c (sole args) ] | _ -> Alcotest.fail "arity")
        [ scalar 1.0 ]
        []
    with
    | [ c ] -> c
    | _ -> Alcotest.fail "arity"
  in
  check "value" 8.0 (value (run [ scalar 2.0 ]))

let closes_over_tracer () =
  let f args =
    let y = sole args in
    match
      Loops.scan
        (function
          | [ c; x ] -> [ mul (mul c x) y ] | _ -> Alcotest.fail "arity")
        [ scalar 1.0 ]
        [ vec [| 2.0; 3.0 |] ]
    with
    | [ c ] -> c
    | _ -> Alcotest.fail "arity"
  in
  check "grad" 24.0 (value (sole (grad_of f [ scalar 2.0 ])))

let vmap_of_grad () =
  let out =
    Batching.vmap_flat
      (fun a ->
        [ List.hd (grad_of product_scan [ sole a; vec [| 3.0; 4.0 |] ]) ])
      [ Some 0 ]
      [ vec [| 1.0; 2.0 |] ]
  in
  check "first" 12.0 (at (sole out) 0);
  check "second" 12.0 (at (sole out) 1)

let cond_closure () =
  let f args =
    let y = sole args in
    sole
      (Lax.cond
         (Concrete (Ndarray.of_floats Dtype.Bool [||] [| 1.0 |]))
         (fun ops -> [ mul (sole ops) y ])
         (fun ops -> ops)
         [ scalar 3.0 ])
  in
  check "grad" 3.0 (value (sole (grad_of f [ scalar 2.0 ])))

let while_closure () =
  let f args =
    let y = sole args in
    let pred = function
      | c :: _ -> Core.bind1 Lt [ c; scalar 10.0 ]
      | [] -> Alcotest.fail "arity"
    in
    sole (Lax.while_loop pred (fun c -> [ mul (sole c) y ]) [ scalar 1.0 ])
  in
  check "value" 16.0 (value (f [ scalar 2.0 ]));
  let po, to_ = Ad.jvp (fun a -> [ f a ]) [ scalar 2.0 ] [ scalar 1.0 ] in
  check "primal" 16.0 (value (sole po));
  check "tangent" 32.0 (value (sole to_))

let jit_of_grad () =
  let jitted = Api.jit_flat (fun args -> grad_of product_scan args) in
  match jitted [ scalar 2.0; vec [| 3.0; 4.0 |] ] with
  | [ gc; gx ] ->
      check "d/dc" 12.0 (value gc);
      check "d/dx1" 6.0 (at gx 1)
  | _ -> Alcotest.fail "arity"

let jit_cond_multi () =
  let f =
    Api.jit_flat (fun args ->
        match args with
        | [ p; x ] ->
            Lax.cond p
              (fun ops -> [ mul (sole ops) (scalar 2.0); sole ops ])
              (fun ops -> [ sole ops; mul (sole ops) (scalar 3.0) ])
              [ x ]
        | _ -> Alcotest.fail "arity")
  in
  let pred b = Concrete (Ndarray.of_floats Dtype.Bool [||] [| b |]) in
  match f [ pred 1.0; scalar 2.0 ] with
  | [ a; b ] ->
      check "true first" 4.0 (value a);
      check "true second" 2.0 (value b)
  | _ -> Alcotest.fail "arity"

let jit_scan_ys () =
  let f =
    Api.jit_flat (fun args ->
        Lax.scan ~reverse:true
          (function
            | [ c; x ] -> [ add c x; mul c x ] | _ -> Alcotest.fail "arity")
          [ scalar 1.0 ]
          args)
  in
  match f [ vec [| 2.0; 3.0 |] ] with
  | [ c; ys ] ->
      check "carry" 6.0 (value c);
      check "y1" 3.0 (at ys 1);
      check "y0" 8.0 (at ys 0)
  | _ -> Alcotest.fail "arity"

let contains text fragment =
  let n = String.length fragment in
  let rec go i =
    i + n <= String.length text && (String.sub text i n = fragment || go (i + 1))
  in
  go 0

let scan_lowers_to_while () =
  let body = function
    | [ c; x ] -> [ add c x; mul c x ]
    | _ -> Alcotest.fail "arity"
  in
  let aval = { shape = [| 2 |]; dtype = Dtype.F32; weak_type = false } in
  let closed =
    Jaxpr.make_jaxpr [ aval ] (fun args -> Lax.scan body [ scalar 1.0 ] args)
  in
  let text = Stablehlo.Emit.emit_closed_jaxpr closed in
  Alcotest.(check bool) "while" true (contains text "stablehlo.while")

let () =
  Alcotest.run "scan_grad"
    [
      ( "reverse",
        [
          Alcotest.test_case "carry and xs" `Quick grad_carry_and_xs;
          Alcotest.test_case "nonlinear ys" `Quick grad_nonlinear_ys;
          Alcotest.test_case "reverse scan" `Quick grad_reverse_scan;
          Alcotest.test_case "second order" `Quick second_order;
          Alcotest.test_case "length only" `Quick length_without_xs;
          Alcotest.test_case "closure over tracer" `Quick closes_over_tracer;
          Alcotest.test_case "vmap of grad" `Quick vmap_of_grad;
          Alcotest.test_case "jit of grad" `Quick jit_of_grad;
          Alcotest.test_case "cond closure" `Quick cond_closure;
          Alcotest.test_case "while closure" `Quick while_closure;
          Alcotest.test_case "jit cond multi" `Quick jit_cond_multi;
          Alcotest.test_case "jit scan ys" `Quick jit_scan_ys;
          Alcotest.test_case "lowers to while" `Quick scan_lowers_to_while;
        ] );
    ]
