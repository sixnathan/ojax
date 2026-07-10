module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ph = Ojax.Random.Philox
module Tf4 = Ojax.Random.Threefry4x32
module Rbg = Ojax.Random.Rbg
module St = Ojax.Random.Stateful

let () = Ojax.Random.Prng.install ()
let () = Ojax.Random.Threefry.install ()
let () = Ph.install ()
let () = Tf4.install ()
let seed64 v = T.Concrete (Nd.of_floats D.I64 [||] [| float_of_int v |])
let u32s v = T.Concrete (Nd.of_floats D.Uint32 [||] [| float_of_int v |])

let ints_of v =
  match v with
  | T.Concrete a ->
      let n = Array.fold_left ( * ) 1 (Nd.shape a) in
      List.init n (fun i ->
          let idx =
            let sh = Nd.shape a in
            let r = Array.length sh in
            let ix = Array.make r 0 in
            let rem = ref i in
            for k = r - 1 downto 0 do
              ix.(k) <- !rem mod sh.(k);
              rem := !rem / sh.(k)
            done;
            ix
          in
          Int64.to_int (Nd.get_i64 a idx))
  | _ -> Alcotest.fail "not concrete"

let check name expected v =
  Alcotest.(check (list int)) name expected (ints_of v)

let philox2x32 () =
  let k = Ph.philox2x32_seed (seed64 42) in
  check "p2 seed" [ 3991259024 ] k;
  check "p2 split"
    [ 2171499443; 413155073; 2864015503 ]
    (Ph.philox2x32_split k [| 3 |]);
  check "p2 bits"
    [ 944504651; 856563819; 3743454102; 2331796311 ]
    (Ph.philox2x32_random_bits k ~bit_width:32 ~shape:[| 4 |]);
  check "p2 fold_in" [ 684560128 ] (Ph.philox2x32_fold_in k (u32s 7))

let philox4x32 () =
  let k = Ph.philox4x32_seed (seed64 42) in
  check "p4 seed" [ 3717705800; 533567081 ] k;
  check "p4 split"
    [ 2181901948; 3341115038; 969239790; 3857553174 ]
    (Ph.philox4x32_split k [| 2 |]);
  check "p4 bits"
    [ 1373712901; 3223971023; 2207559167 ]
    (Ph.philox4x32_random_bits k ~bit_width:32 ~shape:[| 3 |]);
  check "p4 fold_in" [ 41828760; 81464021 ] (Ph.philox4x32_fold_in k (u32s 7))

let threefry4x32 () =
  let k = Tf4.threefry4x32_seed (seed64 42) in
  check "t4 seed" [ 1605953687; 3689221460; 1722219033; 3178796034 ] k;
  check "t4 split"
    [
      3188509147;
      2083947553;
      1469629592;
      921930771;
      361144456;
      1418825268;
      1547191614;
      2559243763;
    ]
    (Tf4.threefry4x32_split k [| 2 |]);
  check "t4 bits"
    [ 2739996017; 3649338001; 3087855958 ]
    (Tf4.threefry4x32_random_bits k ~bit_width:32 ~shape:[| 3 |]);
  check "t4 fold_in"
    [ 2445197787; 3874293641; 4221169969; 2984886860 ]
    (Tf4.threefry4x32_fold_in k (u32s 7))

let rbg () =
  let k = Rbg.rbg_seed (seed64 42) in
  check "rbg seed" [ 0; 42; 0; 42 ] k;
  check "rbg split"
    [
      1832780943;
      270669613;
      1832780943;
      270669613;
      64467757;
      2916123636;
      64467757;
      2916123636;
    ]
    (Rbg.rbg_split k [| 2 |]);
  check "rbg fold_in"
    [ 2547012911; 1371500959; 2547012911; 1371500959 ]
    (Rbg.rbg_fold_in k (u32s 7));
  Alcotest.check_raises "rbg random_bits bounded"
    (Failure
       "rbg: random_bits lowers to XLA lax.rng_bit_generator, whose output is \
        not guaranteed stable across backends/compiler versions (per jax) and \
        is not host-reproducible bit-exactly; device-only, not host-goldenable \
        (M5)") (fun () ->
      ignore
        (Rbg.rbg_prng_impl.random_bits
           (Nd.of_floats D.Uint32 [| 4 |] [| 0.; 0.; 0.; 0. |])
           32 [| 2 |]))

let primitive_live () =
  let a, b =
    match C.bind T.Philox2x32 [ u32s 0; u32s 0; u32s 42 ] with
    | [ a; b ] -> (a, b)
    | _ -> Alcotest.fail "arity"
  in
  check "philox2x32 prim out0" [ 3991259024 ] a;
  ignore b;
  let outs =
    C.bind T.Threefry4x32 (List.map u32s [ 0; 42; 0; 0; 0; 0; 0; 0 ])
  in
  check "threefry4x32 prim out0" [ 1605953687 ] (List.nth outs 0)

let stateful () =
  let g = St.stateful_rng 0 in
  check "stateful key1" [ 1797259609; 2579123966 ] (St.key g);
  check "stateful key2" [ 928981903; 3453687069 ] (St.key g)

let () =
  Alcotest.run "random_alt_prng"
    [
      ( "alt-prng",
        [
          Alcotest.test_case "philox2x32" `Quick philox2x32;
          Alcotest.test_case "philox4x32" `Quick philox4x32;
          Alcotest.test_case "threefry4x32" `Quick threefry4x32;
          Alcotest.test_case "rbg" `Quick rbg;
          Alcotest.test_case "primitive_live" `Quick primitive_live;
          Alcotest.test_case "stateful" `Quick stateful;
        ] );
    ]
