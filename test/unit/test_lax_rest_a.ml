module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module J = Ojax.Jaxpr
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)
let f32 shape xs = cval D.F32 shape xs
let i32 shape xs = cval D.I32 shape xs

let out_floats v =
  match v with
  | T.Concrete a ->
      let n = Array.fold_left ( * ) 1 (Nd.shape a) in
      let arr = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            arr.(i) <- x;
            i + 1)
          0 a
      in
      arr
  | T.Tracer _ -> Alcotest.fail "not concrete"

let approx = Alcotest.(float 1e-6)
let arr = Alcotest.(array approx)

let test_clamp_impl () =
  let r =
    C.bind1 T.Clamp
      [
        f32 [| 3 |] [| 0.; 0.; 0. |];
        f32 [| 3 |] [| -1.; 5.; 2. |];
        f32 [| 3 |] [| 3.; 3.; 3. |];
      ]
  in
  Alcotest.check arr "clamp" [| 0.; 3.; 2. |] (out_floats r)

let test_clamp_impl_i32 () =
  let r =
    C.bind1 T.Clamp
      [
        i32 [| 3 |] [| 2.; 2.; 2. |];
        i32 [| 3 |] [| 0.; 4.; 8. |];
        i32 [| 3 |] [| 6.; 6.; 6. |];
      ]
  in
  Alcotest.check arr "clamp i32" [| 2.; 4.; 6. |] (out_floats r)

let test_bitcast_f32_i32 () =
  let r =
    C.bind1 (T.Bitcast_convert_type D.I32) [ f32 [| 2 |] [| 1.0; 2.0 |] ]
  in
  Alcotest.(check (array int64))
    "bitcast f32->i32"
    [| 1065353216L; 1073741824L |]
    (Array.map Int64.of_float (out_floats r));
  Alcotest.(check bool)
    "dtype i32" true
    (Nd.dtype (match r with T.Concrete a -> a | _ -> assert false) = D.I32)

let test_bitcast_i32_f32 () =
  let r =
    C.bind1 (T.Bitcast_convert_type D.F32)
      [ i32 [| 2 |] [| 1065353216.; 1073741824. |] ]
  in
  Alcotest.check arr "bitcast i32->f32" [| 1.0; 2.0 |] (out_floats r)

let test_iota_f32 () =
  let r =
    C.bind1 (T.Iota { dtype = D.F32; shape = [| 4 |]; dimension = 0 }) []
  in
  Alcotest.check arr "iota 1d" [| 0.; 1.; 2.; 3. |] (out_floats r)

let test_iota_dim0 () =
  let r =
    C.bind1 (T.Iota { dtype = D.I32; shape = [| 2; 3 |]; dimension = 0 }) []
  in
  Alcotest.check arr "iota dim0" [| 0.; 0.; 0.; 1.; 1.; 1. |] (out_floats r)

let test_iota_dim1 () =
  let r =
    C.bind1 (T.Iota { dtype = D.I32; shape = [| 2; 3 |]; dimension = 1 }) []
  in
  Alcotest.check arr "iota dim1" [| 0.; 1.; 2.; 0.; 1.; 2. |] (out_floats r)

let test_empty () =
  let r = C.bind1 (T.Empty { shape = [| 3 |]; dtype = D.F32 }) [] in
  Alcotest.check arr "empty" [| 0.; 0.; 0. |] (out_floats r)

let test_empty2 () =
  let r = C.bind1 (T.Empty2 D.F32) [] in
  Alcotest.check arr "empty2" [| 0. |] (out_floats r)

let test_dce_sink () =
  let r = C.bind T.Dce_sink [ f32 [| 2 |] [| 1.; 2. |] ] in
  Alcotest.(check int) "dce_sink no output" 0 (List.length r)

let test_composite () =
  let av = { T.shape = [| 3 |]; dtype = D.F32; weak_type = false } in
  let cj = J.make_jaxpr [ av ] (fun a -> [ C.bind1 T.Sin a ]) in
  let x = f32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let r = C.bind1 (T.Composite cj) [ x ] in
  Alcotest.check arr "composite=sin"
    [| sin 0.1; sin 0.2; sin 0.3 |]
    (out_floats r)

let test_clamp_jvp () =
  let mn = f32 [| 3 |] [| 0.; 0.; 0. |] in
  let x = f32 [| 3 |] [| 1.; -1.; 5. |] in
  let mx = f32 [| 3 |] [| 3.; 3.; 3. |] in
  let tmn = f32 [| 3 |] [| 0.; 0.; 0. |] in
  let tx = f32 [| 3 |] [| 1.; 1.; 1. |] in
  let tmx = f32 [| 3 |] [| 0.; 0.; 0. |] in
  let _, to_ =
    Ad.jvp (fun a -> [ C.bind1 T.Clamp a ]) [ mn; x; mx ] [ tmn; tx; tmx ]
  in
  Alcotest.check arr "clamp jvp" [| 1.; 0.; 0. |] (out_floats (List.hd to_))

let test_clamp_vmap () =
  let mn = f32 [| 2; 3 |] [| 0.; 0.; 0.; 0.; 0.; 0. |] in
  let x = f32 [| 2; 3 |] [| -1.; 1.; 5.; 2.; 4.; -2. |] in
  let mx = f32 [| 2; 3 |] [| 3.; 3.; 3.; 3.; 3.; 3. |] in
  let r =
    Batching.vmap
      (fun a -> [ C.bind1 T.Clamp a ])
      [ Some 0; Some 0; Some 0 ] [ mn; x; mx ]
  in
  Alcotest.check arr "clamp vmap"
    [| 0.; 1.; 3.; 2.; 3.; 0. |]
    (out_floats (List.hd r))

let () =
  Alcotest.run "lax_rest_a"
    [
      ( "impl",
        [
          Alcotest.test_case "clamp" `Quick test_clamp_impl;
          Alcotest.test_case "clamp_i32" `Quick test_clamp_impl_i32;
          Alcotest.test_case "bitcast_f32_i32" `Quick test_bitcast_f32_i32;
          Alcotest.test_case "bitcast_i32_f32" `Quick test_bitcast_i32_f32;
          Alcotest.test_case "iota_f32" `Quick test_iota_f32;
          Alcotest.test_case "iota_dim0" `Quick test_iota_dim0;
          Alcotest.test_case "iota_dim1" `Quick test_iota_dim1;
          Alcotest.test_case "empty" `Quick test_empty;
          Alcotest.test_case "empty2" `Quick test_empty2;
          Alcotest.test_case "dce_sink" `Quick test_dce_sink;
          Alcotest.test_case "composite" `Quick test_composite;
        ] );
      ( "transforms",
        [
          Alcotest.test_case "clamp_jvp" `Quick test_clamp_jvp;
          Alcotest.test_case "clamp_vmap" `Quick test_clamp_vmap;
        ] );
    ]
