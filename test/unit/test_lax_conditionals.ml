module T = Ojax.Types
module C = Ojax.Core
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Cond = Ojax.Lax
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = Ojax.Lax.install ()
let b1 = C.bind1
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

let bool_v b =
  T.Concrete (Nd.of_floats D.Bool [||] [| (if b then 1.0 else 0.0) |])

let read v =
  match v with
  | T.Concrete nd ->
      let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
      let a = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            a.(i) <- x;
            i + 1)
          0 nd
      in
      a
  | T.Tracer _ -> failwith "expected concrete"

let approx name a b =
  Array.iter2
    (fun x y ->
      if Float.abs (x -. y) > 1e-5 then Alcotest.failf "%s: %f <> %f" name x y)
    a b

let sin_fn a = [ b1 T.Sin a ]
let cos_fn a = [ b1 T.Cos a ]

let test_eval () =
  let x = f32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let ot = Cond.cond (bool_v true) sin_fn cos_fn [ x ] in
  let of_ = Cond.cond (bool_v false) sin_fn cos_fn [ x ] in
  approx "true=sin" (read (List.hd ot)) (read (b1 T.Sin [ x ]));
  approx "false=cos" (read (List.hd of_)) (read (b1 T.Cos [ x ]))

let test_jvp () =
  let x = f32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let tx = f32 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  let f p = fun ops -> Cond.cond p sin_fn cos_fn ops in
  let _, to_ = Ad.jvp (f (bool_v true)) [ x ] [ tx ] in
  approx "jvp true = cos*tx"
    (read (List.hd to_))
    (read (b1 T.Mul [ b1 T.Cos [ x ]; tx ]))

let test_grad () =
  let x = f32 [||] [| 0.5 |] in
  let g_true =
    Ad.grad
      (fun ops -> List.hd (Cond.cond (bool_v true) sin_fn cos_fn ops))
      [ x ]
  in
  approx "grad true = cos" (read (List.hd g_true)) (read (b1 T.Cos [ x ]));
  let g_false =
    Ad.grad
      (fun ops -> List.hd (Cond.cond (bool_v false) sin_fn cos_fn ops))
      [ x ]
  in
  approx "grad false = -sin"
    (read (List.hd g_false))
    (read (b1 T.Neg [ b1 T.Sin [ x ] ]))

let test_vmap () =
  let x = f32 [| 4 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let out =
    Batching.vmap
      (fun ops -> Cond.cond (bool_v true) sin_fn cos_fn ops)
      [ Some 0 ] [ x ]
  in
  approx "vmap true = sin" (read (List.hd out)) (read (b1 T.Sin [ x ]))

let test_platform_index () =
  let i0 =
    Cond.platform_index ~platforms:[| Some [| "cpu" |]; Some [| "tpu" |] |]
  in
  let i1 =
    Cond.platform_index ~platforms:[| Some [| "tpu" |]; Some [| "cpu" |] |]
  in
  let idef = Cond.platform_index ~platforms:[| Some [| "tpu" |]; None |] in
  approx "pi0" (read i0) [| 0.0 |];
  approx "pi1" (read i1) [| 1.0 |];
  approx "pidef" (read idef) [| 1.0 |]

let () =
  Alcotest.run "lax_conditionals"
    [
      ( "conditionals",
        [
          Alcotest.test_case "eval" `Quick test_eval;
          Alcotest.test_case "jvp" `Quick test_jvp;
          Alcotest.test_case "grad" `Quick test_grad;
          Alcotest.test_case "vmap" `Quick test_vmap;
          Alcotest.test_case "platform_index" `Quick test_platform_index;
        ] );
    ]
