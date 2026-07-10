module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module NL = Ojax.Numpy.Lax_numpy

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let i32 shape xs = T.Concrete (Nd.of_floats D.I32 shape xs)

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
  | T.Device _ -> failwith "expected concrete"

let shape_of v = (Ojax.Core.get_aval v).T.shape
let dtype_of v = (Ojax.Core.get_aval v).T.dtype
let farr = Alcotest.(array (float 1e-6))

let test_shape_ops () =
  let x = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  Alcotest.(check (array int))
    "transpose" [| 3; 2 |]
    (shape_of (NL.transpose x));
  Alcotest.(check (array int))
    "reshape" [| 6 |]
    (shape_of (NL.reshape x [| 6 |]));
  Alcotest.(check (array int))
    "reshape neg1" [| 3; 2 |]
    (shape_of (NL.reshape x [| -1; 2 |]));
  Alcotest.(check (array int)) "ravel" [| 6 |] (shape_of (NL.ravel x));
  Alcotest.check farr "fliplr" [| 3.; 2.; 1.; 6.; 5.; 4. |] (read (NL.fliplr x));
  Alcotest.check farr "flipud" [| 4.; 5.; 6.; 1.; 2.; 3. |] (read (NL.flipud x))

let test_trunc () =
  let x = f32 [| 4 |] [| -1.7; 2.3; -0.4; 3.9 |] in
  Alcotest.check farr "trunc" [| -1.; 2.; -0.; 3. |] (read (NL.trunc x));
  let xi = i32 [| 3 |] [| 1.; 2.; 3. |] in
  Alcotest.check farr "trunc int passthrough" [| 1.; 2.; 3. |]
    (read (NL.trunc xi))

let test_diff () =
  let x = f32 [| 5 |] [| 1.; 3.; 6.; 10.; 15. |] in
  Alcotest.check farr "diff" [| 2.; 3.; 4.; 5. |] (read (NL.diff x));
  Alcotest.check farr "ediff1d" [| 2.; 3.; 4.; 5. |] (read (NL.ediff1d x))

let test_predicates () =
  let x = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let s = f32 [||] [| 1. |] in
  Alcotest.(check bool) "iscomplexobj" false (NL.iscomplexobj x);
  Alcotest.(check bool) "isrealobj" true (NL.isrealobj x);
  Alcotest.(check bool) "isscalar false" false (NL.isscalar x);
  Alcotest.(check bool) "isscalar true" true (NL.isscalar s);
  Alcotest.(check bool)
    "issubdtype f32 floating" true
    (NL.issubdtype (NL.Cdtype D.F32) NL.Floating);
  Alcotest.(check bool)
    "issubdtype f32 integer" false
    (NL.issubdtype (NL.Cdtype D.F32) NL.Integer);
  Alcotest.(check bool)
    "issubdtype i32 integer" true
    (NL.issubdtype (NL.Cdtype D.I32) NL.Integer);
  Alcotest.(check bool)
    "issubdtype i32 number" true
    (NL.issubdtype (NL.Cdtype D.I32) NL.Number);
  Alcotest.(check bool)
    "issubdtype bool generic" true
    (NL.issubdtype (NL.Cdtype D.Bool) NL.Generic)

let test_result_type () =
  let a = f32 [| 2 |] [| 1.; 2. |] in
  let b = i32 [| 2 |] [| 3.; 4. |] in
  Alcotest.(check bool)
    "result_type f32 i32 -> f32" true
    (NL.result_type [ a; b ] = D.F32)

let test_angle () =
  let x = f32 [| 2 |] [| 2.0; -3.0 |] in
  let r = read (NL.angle x) in
  Alcotest.(check bool) "angle pos = 0" true (Float.abs r.(0) < 1e-6);
  Alcotest.(check bool)
    "angle neg = pi" true
    (Float.abs (r.(1) -. Float.pi) < 1e-6);
  Alcotest.(check bool) "angle dtype" true (dtype_of (NL.angle x) = D.F32)

let bool_ shape xs = T.Concrete (Nd.of_floats D.Bool shape xs)

let test_nan_to_num () =
  let x =
    f32 [| 5 |] [| 0.0; Float.nan; 1.0; Float.infinity; Float.neg_infinity |]
  in
  let r = read (NL.nan_to_num x) in
  Alcotest.(check bool) "nan->0" true (r.(1) = 0.0);
  Alcotest.(check bool) "posinf->max" true (r.(3) > 3.0e38);
  Alcotest.(check bool) "neginf->min" true (r.(4) < -3.0e38);
  Alcotest.(check bool) "finite kept" true (r.(0) = 0.0 && r.(2) = 1.0)

let test_close () =
  let a = f32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = f32 [| 3 |] [| 1.0; 2.0; 3.0000001 |] in
  Alcotest.check farr "isclose equal" [| 1.; 1.; 1. |] (read (NL.isclose a b));
  Alcotest.check farr "allclose true" [| 1. |] (read (NL.allclose a b));
  let c = f32 [| 3 |] [| 1.0; 2.0; 9.0 |] in
  Alcotest.check farr "allclose false" [| 0. |] (read (NL.allclose a c))

let test_clip () =
  let x = f32 [| 5 |] [| -1.0; 0.5; 1.5; 2.5; 3.5 |] in
  Alcotest.check farr "clip" [| 0.; 0.5; 1.5; 2.; 2. |]
    (read (NL.clip ~min:0.0 ~max:2.0 x))

let test_where_select () =
  let c = bool_ [| 4 |] [| 1.; 0.; 1.; 0. |] in
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let y = f32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  Alcotest.check farr "where" [| 1.; 20.; 3.; 40. |] (read (NL.where_ c x y));
  let c0 = bool_ [| 4 |] [| 0.; 1.; 0.; 0. |] in
  let c1 = bool_ [| 4 |] [| 1.; 0.; 0.; 0. |] in
  let ch0 = i32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let ch1 = i32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  Alcotest.check farr "select" [| 10.; 2.; 0.; 0. |]
    (read (NL.select [ c0; c1 ] [ ch0; ch1 ]))

let test_moveaxis_swapaxes () =
  let x = f32 [| 2; 3; 4; 5 |] (Array.make 120 0.0) in
  Alcotest.(check (array int))
    "swapaxes" [| 5; 3; 4; 2 |]
    (shape_of (NL.swapaxes 0 3 x));
  Alcotest.(check (array int))
    "moveaxis" [| 4; 5; 3; 2 |]
    (shape_of (NL.moveaxis [| 0; 1 |] [| -1; -2 |] x));
  Alcotest.(check (array int))
    "expand_dims" [| 1; 2; 3; 4; 5; 1 |]
    (shape_of (NL.expand_dims x [| 0; 5 |]))

let test_squeeze () =
  let x = f32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  Alcotest.(check (array int)) "squeeze all" [| 3 |] (shape_of (NL.squeeze x));
  Alcotest.(check (array int))
    "squeeze axis" [| 1; 3 |]
    (shape_of (NL.squeeze ~axis:[| 2 |] x))

let test_broadcast () =
  Alcotest.(check (array int))
    "broadcast_shapes_n" [| 5; 3; 4 |]
    (NL.broadcast_shapes_n [ [| 3; 1 |]; [| 1; 4 |]; [| 5; 1; 1 |] ]);
  let a = f32 [| 1; 3 |] [| 1.; 2.; 3. |] in
  let b = f32 [| 2; 1 |] [| 10.; 20. |] in
  match NL.broadcast_arrays [ a; b ] with
  | [ a2; b2 ] ->
      Alcotest.(check (array int)) "ba shape a" [| 2; 3 |] (shape_of a2);
      Alcotest.(check (array int)) "ba shape b" [| 2; 3 |] (shape_of b2);
      Alcotest.check farr "ba b vals"
        [| 10.; 10.; 10.; 20.; 20.; 20. |]
        (read b2)
  | _ -> Alcotest.fail "broadcast_arrays arity"

let test_split_resize () =
  let x = f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  (match NL.split x (NL.Count 3) with
  | [ p; q; r ] ->
      Alcotest.check farr "split 0" [| 1.; 2. |] (read p);
      Alcotest.check farr "split 1" [| 3.; 4. |] (read q);
      Alcotest.check farr "split 2" [| 5.; 6. |] (read r)
  | _ -> Alcotest.fail "split arity");
  (match NL.array_split x (NL.Count 4) with
  | [ p; q; r; s ] ->
      Alcotest.(check int) "array_split 0 len" 2 (Array.length (read p));
      Alcotest.(check int) "array_split 1 len" 2 (Array.length (read q));
      Alcotest.(check int) "array_split 2 len" 1 (Array.length (read r));
      Alcotest.(check int) "array_split 3 len" 1 (Array.length (read s))
  | _ -> Alcotest.fail "array_split arity");
  Alcotest.check farr "resize"
    [| 1.; 2.; 3.; 4.; 5.; 6.; 1.; 2.; 3. |]
    (read (NL.resize x [| 3; 3 |]))

let test_unravel () =
  let idx = i32 [| 3 |] [| 1.; 3.; 5. |] in
  match NL.unravel_index idx [| 2; 3 |] with
  | [ r0; r1 ] ->
      Alcotest.check farr "unravel row" [| 0.; 1.; 1. |] (read r0);
      Alcotest.check farr "unravel col" [| 1.; 0.; 2. |] (read r1)
  | _ -> Alcotest.fail "unravel arity"

let test_round () =
  let x = f32 [| 4 |] [| 1.4; 2.6; -1.7; 3.2 |] in
  Alcotest.check farr "round" [| 1.; 3.; -2.; 3. |] (read (NL.round x))

let test_stack_concat () =
  let a = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = f32 [| 3 |] [| 4.; 5.; 6. |] in
  Alcotest.check farr "concatenate"
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (read (NL.concatenate [ a; b ]));
  Alcotest.(check (array int))
    "stack shape" [| 2; 3 |]
    (shape_of (NL.stack [ a; b ]));
  match NL.unstack (NL.stack [ a; b ]) with
  | [ p; q ] ->
      Alcotest.check farr "unstack0" [| 1.; 2.; 3. |] (read p);
      Alcotest.check farr "unstack1" [| 4.; 5.; 6. |] (read q)
  | _ -> Alcotest.fail "unstack arity"

let test_atleast () =
  let s = f32 [||] [| 5. |] in
  Alcotest.(check (array int)) "atleast_1d" [| 1 |] (shape_of (NL.atleast_1d s));
  Alcotest.(check (array int))
    "atleast_2d" [| 1; 1 |]
    (shape_of (NL.atleast_2d s));
  Alcotest.(check (array int))
    "atleast_3d" [| 1; 1; 1 |]
    (shape_of (NL.atleast_3d s));
  let v = f32 [| 3 |] [| 1.; 2.; 3. |] in
  Alcotest.(check (array int))
    "atleast_2d 1d" [| 1; 3 |]
    (shape_of (NL.atleast_2d v));
  Alcotest.(check (array int))
    "atleast_3d 1d" [| 1; 3; 1 |]
    (shape_of (NL.atleast_3d v))

let test_stacks () =
  let a = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = f32 [| 3 |] [| 4.; 5.; 6. |] in
  Alcotest.(check (array int))
    "vstack" [| 2; 3 |]
    (shape_of (NL.vstack [ a; b ]));
  Alcotest.(check (array int)) "hstack" [| 6 |] (shape_of (NL.hstack [ a; b ]));
  Alcotest.(check (array int))
    "dstack" [| 1; 3; 2 |]
    (shape_of (NL.dstack [ a; b ]));
  Alcotest.(check (array int))
    "column_stack" [| 3; 2 |]
    (shape_of (NL.column_stack [ a; b ]))

let test_tile_pad () =
  let a = f32 [| 2 |] [| 1.; 2. |] in
  Alcotest.check farr "tile"
    [| 1.; 2.; 1.; 2.; 1.; 2. |]
    (read (NL.tile a [| 3 |]));
  Alcotest.check farr "pad"
    [| 0.; 0.; 1.; 2.; 0.; 0. |]
    (read (NL.pad a [| (2, 2) |] 0.0))

let test_creation () =
  Alcotest.check farr "arange" [| 0.; 1.; 2.; 3.; 4. |]
    (read (NL.arange ~dtype:D.I32 5.0));
  Alcotest.check farr "arange step" [| 0.; 0.5; 1.; 1.5 |]
    (read (NL.arange ~step:0.5 ~dtype:D.F32 2.0));
  Alcotest.check farr "eye"
    [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
    (read (NL.eye ~dtype:D.F32 3));
  Alcotest.(check (array int))
    "eye shape" [| 3; 5 |]
    (shape_of (NL.eye ~m:5 ~k:1 ~dtype:D.F32 3));
  Alcotest.check farr "identity" [| 1.; 0.; 0.; 1. |]
    (read (NL.identity ~dtype:D.F32 2));
  Alcotest.(check (array int))
    "indices shape" [| 2; 2; 3 |]
    (shape_of (NL.indices ~dtype:D.I32 [| 2; 3 |]))

let test_i0_equal () =
  let x = f32 [| 3 |] [| 0.; 1.; 2. |] in
  let r = read (NL.i0 x) in
  Alcotest.(check bool) "i0(0)=1" true (Float.abs (r.(0) -. 1.0) < 1e-5);
  Alcotest.(check bool) "i0 monotone" true (r.(2) > r.(1) && r.(1) > r.(0));
  let a = i32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = i32 [| 3 |] [| 1.; 2.; 3. |] in
  let c = i32 [| 3 |] [| 1.; 2.; 9. |] in
  Alcotest.check farr "array_equal true" [| 1. |] (read (NL.array_equal a b));
  Alcotest.check farr "array_equal false" [| 0. |] (read (NL.array_equal a c));
  Alcotest.check farr "array_equiv true" [| 1. |] (read (NL.array_equiv a b))

let test_meshgrid_ix () =
  let x = i32 [| 2 |] [| 1.; 2. |] in
  let y = i32 [| 3 |] [| 10.; 20.; 30. |] in
  (match NL.meshgrid [ x; y ] with
  | [ gx; gy ] ->
      Alcotest.(check (array int)) "meshgrid xg" [| 3; 2 |] (shape_of gx);
      Alcotest.check farr "meshgrid xg vals"
        [| 1.; 2.; 1.; 2.; 1.; 2. |]
        (read gx);
      Alcotest.check farr "meshgrid yg vals"
        [| 10.; 10.; 20.; 20.; 30.; 30. |]
        (read gy)
  | _ -> Alcotest.fail "meshgrid arity");
  match NL.ix_ [ x; y ] with
  | [ ax; ay ] ->
      Alcotest.(check (array int)) "ix_ ax" [| 2; 1 |] (shape_of ax);
      Alcotest.(check (array int)) "ix_ ay" [| 1; 3 |] (shape_of ay)
  | _ -> Alcotest.fail "ix_ arity"

let test_g4 () =
  let x1 = i32 [| 3 |] [| 1.; 2.; 3. |] in
  Alcotest.check farr "diag construct"
    [| 1.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 3. |]
    (read (NL.diag x1));
  let m = i32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |] in
  Alcotest.check farr "diag extract" [| 1.; 5.; 9. |] (read (NL.diag m));
  Alcotest.check farr "diagonal offset1" [| 2.; 6. |]
    (read (NL.diagonal ~offset:1 m));
  Alcotest.check farr "diagonal offset-1" [| 4.; 8. |]
    (read (NL.diagonal ~offset:(-1) m));
  Alcotest.check farr "trace" [| 15. |] (read (NL.trace m));
  Alcotest.check farr "tril"
    [| 1.; 0.; 0.; 4.; 5.; 0.; 7.; 8.; 9. |]
    (read (NL.tril m));
  Alcotest.check farr "triu"
    [| 1.; 2.; 3.; 0.; 5.; 6.; 0.; 0.; 9. |]
    (read (NL.triu m));
  Alcotest.check farr "tri"
    [| 1.; 0.; 0.; 1.; 1.; 0.; 1.; 1.; 1. |]
    (read (NL.tri ~dtype:D.F32 3));
  let a = i32 [| 2 |] [| 1.; 2. |] and b = i32 [| 2 |] [| 3.; 4. |] in
  Alcotest.check farr "cross2d" [| -2. |] (read (NL.cross a b));
  let a3 = i32 [| 3 |] [| 1.; 2.; 3. |] and b3 = i32 [| 3 |] [| 4.; 5.; 6. |] in
  Alcotest.check farr "cross3d" [| -3.; 6.; -3. |] (read (NL.cross a3 b3));
  Alcotest.check farr "append flat" [| 1.; 2.; 3.; 4. |] (read (NL.append a b));
  Alcotest.(check (array int))
    "kron shape" [| 6; 6 |]
    (shape_of (NL.kron m (i32 [| 2; 2 |] [| 1.; 1.; 1.; 1. |])));
  Alcotest.(check (array int))
    "repeat shape" [| 3; 6 |]
    (shape_of (NL.repeat ~axis:1 m 2));
  let v = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Alcotest.check farr "vander inc"
    [| 1.; 1.; 1.; 1.; 2.; 4.; 1.; 3.; 9.; 1.; 4.; 16. |]
    (read (NL.vander ~n:3 ~increasing:true v));
  let y = f32 [| 5 |] [| 1.; 2.; 3.; 2.; 1. |] in
  Alcotest.check farr "trapezoid" [| 8. |] (read (NL.trapezoid y));
  Alcotest.(check int) "argmax" 2 (int_of_float (read (NL.argmax x1)).(0));
  (match NL.diag_indices 3 with
  | [ i; j ] ->
      Alcotest.check farr "diag_indices i" [| 0.; 1.; 2. |] (read i);
      Alcotest.check farr "diag_indices j" [| 0.; 1.; 2. |] (read j)
  | _ -> Alcotest.fail "diag_indices arity");
  Alcotest.(check bool)
    "diag_indices dtype" true
    (dtype_of (List.hd (NL.diag_indices 3)) = D.I32)

let test_g5 () =
  let nan = Float.nan in
  Alcotest.check farr "argmin" [| 0.; 2. |]
    (read (NL.argmin ~axis:1 (f32 [| 2; 3 |] [| 1.; 3.; 2.; 5.; 4.; 1. |])));
  Alcotest.check farr "nanargmax nan-skip" [| 2. |]
    (read (NL.nanargmax ~axis:0 (f32 [| 5 |] [| 1.; 3.; 5.; 4.; nan |])));
  Alcotest.check farr "nanargmax all-nan -> -1" [| -1. |]
    (read (NL.nanargmax ~axis:0 (f32 [| 3 |] [| nan; nan; nan |])));
  Alcotest.check farr "nanargmin nan-skip" [| 4. |]
    (read (NL.nanargmin ~axis:0 (f32 [| 5 |] [| nan; 3.; 5.; 4.; 2. |])));
  Alcotest.check farr "roll"
    [| 4.; 5.; 0.; 1.; 2.; 3. |]
    (read (NL.roll (f32 [| 6 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]) [| 2 |]));
  Alcotest.(check (array int))
    "rollaxis shape" [| 4; 2; 3 |]
    (shape_of (NL.rollaxis 2 (f32 [| 2; 3; 4 |] (Array.make 24 0.))));
  Alcotest.check farr "gcd" [| 1.; 2.; 3. |]
    (read
       (NL.gcd
          (i32 [| 3 |] [| 12.; 18.; 24. |])
          (i32 [| 3 |] [| 5.; 10.; 15. |])));
  Alcotest.check farr "lcm" [| 60.; 90.; 120. |]
    (read
       (NL.lcm
          (i32 [| 3 |] [| 12.; 18.; 24. |])
          (i32 [| 3 |] [| 5.; 10.; 15. |])));
  let a = f32 [| 7 |] [| 1.; 2.; 2.; 3.; 4.; 5.; 5. |] in
  Alcotest.check farr "searchsorted left" [| 1.; 5. |]
    (read (NL.searchsorted ~side:"left" a (f32 [| 2 |] [| 2.; 5. |])));
  Alcotest.check farr "searchsorted right" [| 3.; 7. |]
    (read (NL.searchsorted ~side:"right" a (f32 [| 2 |] [| 2.; 5. |])));
  Alcotest.check farr "digitize"
    [| 1.; 2.; 2.; 1.; 3.; 3. |]
    (read
       (NL.digitize
          (f32 [| 6 |] [| 1.; 2.; 2.5; 1.5; 3.; 3.5 |])
          (f32 [| 3 |] [| 1.; 2.; 3. |])));
  Alcotest.check farr "cov perfect-anticorr" [| 1.; -1.; -1.; 1. |]
    (read (NL.cov (f32 [| 2; 3 |] [| -1.; 0.; 1.; 1.; 0.; -1. |])));
  Alcotest.check farr "corrcoef" [| 1.; -1.; -1.; 1. |]
    (read (NL.corrcoef (f32 [| 2; 3 |] [| -1.; 0.; 1.; 1.; 0.; -1. |])))

let () =
  Alcotest.run "numpy_lax_numpy"
    [
      ( "lax_numpy",
        [
          Alcotest.test_case "shape_ops" `Quick test_shape_ops;
          Alcotest.test_case "trunc" `Quick test_trunc;
          Alcotest.test_case "diff" `Quick test_diff;
          Alcotest.test_case "predicates" `Quick test_predicates;
          Alcotest.test_case "result_type" `Quick test_result_type;
          Alcotest.test_case "angle" `Quick test_angle;
          Alcotest.test_case "nan_to_num" `Quick test_nan_to_num;
          Alcotest.test_case "close" `Quick test_close;
          Alcotest.test_case "clip" `Quick test_clip;
          Alcotest.test_case "where_select" `Quick test_where_select;
          Alcotest.test_case "moveaxis_swapaxes" `Quick test_moveaxis_swapaxes;
          Alcotest.test_case "squeeze" `Quick test_squeeze;
          Alcotest.test_case "broadcast" `Quick test_broadcast;
          Alcotest.test_case "split_resize" `Quick test_split_resize;
          Alcotest.test_case "unravel" `Quick test_unravel;
          Alcotest.test_case "round" `Quick test_round;
          Alcotest.test_case "stack_concat" `Quick test_stack_concat;
          Alcotest.test_case "atleast" `Quick test_atleast;
          Alcotest.test_case "stacks" `Quick test_stacks;
          Alcotest.test_case "tile_pad" `Quick test_tile_pad;
          Alcotest.test_case "creation" `Quick test_creation;
          Alcotest.test_case "i0_equal" `Quick test_i0_equal;
          Alcotest.test_case "meshgrid_ix" `Quick test_meshgrid_ix;
          Alcotest.test_case "g4" `Quick test_g4;
          Alcotest.test_case "g5" `Quick test_g5;
        ] );
    ]
