open Ojax
open Ojax.Types
module L = Lax_api
module C = Lax_config
module X = Lax_extras
module LL = Lax.Linalg
module NN = Nn.Functions
module E = Numpy.Einsum
module EP = Numpy.Einsum_paths
module IT = Numpy.Index_tricks
module TC = Numpy.Tensor_contractions
module P = Parallel
module B = Interpreters.Batching
module Tree = Tree_util

let () = Lax.install ()
let plugin_present () = Ojax.Pjrt.Discover.find () <> None
let arr dt sh xs = Concrete (Ndarray.of_floats dt sh xs)

let nd = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let prod sh = Array.fold_left ( * ) 1 sh

let unravel sh f =
  let n = Array.length sh in
  let idx = Array.make n 0 in
  let rank_from i =
    let r = ref i in
    for d = n - 1 downto 0 do
      idx.(d) <- !r mod sh.(d);
      r := !r / sh.(d)
    done
  in
  Array.init (prod sh) (fun i ->
      rank_from i;
      f (Array.copy idx))

let tolerance_of = function Dtype.F16 | Dtype.Bf16 -> 1e-3 | _ -> 1e-4

let near tol expected got =
  (Float.is_nan expected && Float.is_nan got)
  || expected = got
  || Float.abs (expected -. got) <= tol *. (1.0 +. Float.abs expected)

let same name dt sh expected v =
  let a = nd v in
  Alcotest.(check string)
    (name ^ " dtype") (Dtype.short_name dt)
    (Dtype.short_name (Ndarray.dtype a));
  Alcotest.(check (array int)) (name ^ " shape") sh (Ndarray.shape a);
  let got = unravel sh (Ndarray.get_f a) in
  Array.iteri
    (fun i e ->
      if not (near (tolerance_of dt) e got.(i)) then
        Alcotest.failf "%s[%d]: expected %.9g got %.9g" name i e got.(i))
    expected

let same_c name dt sh expected v =
  let a = nd v in
  Alcotest.(check string)
    (name ^ " dtype") (Dtype.short_name dt)
    (Dtype.short_name (Ndarray.dtype a));
  Alcotest.(check (array int)) (name ^ " shape") sh (Ndarray.shape a);
  let got = unravel sh (Ndarray.get_c a) in
  Array.iteri
    (fun i (re, im) ->
      let z = got.(i) in
      if not (near 1e-4 re z.Complex.re && near 1e-4 im z.Complex.im) then
        Alcotest.failf "%s[%d]: expected %g+%gi got %g+%gi" name i re im
          z.Complex.re z.Complex.im)
    expected

let same_uv name expected_u expected_vt u_shape vt_shape u vt =
  let u = nd u and vt = nd vt in
  Alcotest.(check (array int)) (name ^ " u shape") u_shape (Ndarray.shape u);
  Alcotest.(check (array int)) (name ^ " vt shape") vt_shape (Ndarray.shape vt);
  let rows = u_shape.(0) and cols = u_shape.(1) in
  let vrows = vt_shape.(0) and vcols = vt_shape.(1) in
  let paired = Stdlib.min cols vrows in
  for k = 0 to paired - 1 do
    let dot = ref 0.0 in
    for r = 0 to rows - 1 do
      dot := !dot +. (Ndarray.get_f u [| r; k |] *. expected_u.((r * cols) + k))
    done;
    let sign = if !dot < 0.0 then -1.0 else 1.0 in
    for r = 0 to rows - 1 do
      let e = expected_u.((r * cols) + k) in
      let g = sign *. Ndarray.get_f u [| r; k |] in
      if not (near 1e-3 e g) then
        Alcotest.failf "%s u[%d,%d]: expected %g got %g" name r k e g
    done;
    for c = 0 to vcols - 1 do
      let e = expected_vt.((k * vcols) + c) in
      let g = sign *. Ndarray.get_f vt [| k; c |] in
      if not (near 1e-3 e g) then
        Alcotest.failf "%s vt[%d,%d]: expected %g got %g" name k c e g
    done
  done

let reconstructs name m u s vt =
  let m = nd m and u = nd u and s = nd s and vt = nd vt in
  let sm = Ndarray.shape m in
  let k = (Ndarray.shape s).(0) in
  for r = 0 to sm.(0) - 1 do
    for c = 0 to sm.(1) - 1 do
      let total = ref Complex.zero in
      for j = 0 to k - 1 do
        let sv = { Complex.re = Ndarray.get_f s [| j |]; im = 0.0 } in
        total :=
          Complex.add !total
            (Complex.mul
               (Complex.mul (Ndarray.get_c u [| r; j |]) sv)
               (Ndarray.get_c vt [| j; c |]))
      done;
      let e = Ndarray.get_c m [| r; c |] in
      if Complex.norm (Complex.sub e !total) > 1e-4 then
        Alcotest.failf "%s reconstruct[%d,%d]" name r c
    done
  done

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let raises name sub f =
  match f () with
  | _ -> Alcotest.failf "%s: expected an error" name
  | exception
      ( Invalid_argument msg
      | Failure msg
      | Ojax.Py_exn.Value_error msg
      | Ojax.Py_exn.Type_error msg
      | Ojax.Py_exn.Not_implemented_error msg ) ->
      if not (contains msg sub) then
        Alcotest.failf "%s: unexpected message %s" name msg

let sole = function [ x ] -> x | _ -> Alcotest.fail "expected one value"
let jit1 f x = sole (Api.jit_flat (fun a -> [ f (sole a) ]) [ x ])
let jitn f xs = sole (Api.jit_flat (fun a -> [ f a ]) xs)
let vmap1 f x = sole (B.vmap_flat (fun a -> [ f (sole a) ]) [ Some 0 ] [ x ])
let unleaf = function Tree.Leaf v -> v | _ -> Alcotest.fail "expected leaf"

let lift_fn f = function
  | [ Tree.Leaf v ] -> Tree.Leaf (f v)
  | _ -> Alcotest.fail "expected one leaf"

let grad1 f x = unleaf (Api.grad (lift_fn f) [ Tree.Leaf x ])

let jvp1 f x t =
  let p, d = Api.jvp (lift_fn f) [ Tree.Leaf x ] [ Tree.Leaf t ] in
  (unleaf p, unleaf d)

let over_i f xs =
  sole
    (B.vmap_flat ~axis_name:"i"
       (fun a -> [ f a ])
       (List.map (fun _ -> Some 0) xs)
       xs)

let ints sh xs = arr Dtype.I32 sh xs

let ragged_doc_args () =
  [
    ints [| 2; 3 |] [| 1.; 2.; 2.; 3.; 4.; 0. |];
    ints [| 2; 4 |] (Array.make 8 0.);
    ints [| 2; 2 |] [| 0.; 1.; 0.; 1. |];
    ints [| 2; 2 |] [| 1.; 2.; 1.; 1. |];
    ints [| 2; 2 |] [| 0.; 0.; 1.; 2. |];
    ints [| 2; 2 |] [| 1.; 1.; 2.; 1. |];
  ]

let call_ragged = function
  | [ a; b; c; d; e; f ] -> P.ragged_all_to_all a b c d e f ~axis_name:"i"
  | _ -> Alcotest.fail "ragged arity"

let ragged_all_to_all_doc () =
  same "doc example" Dtype.I32 [| 2; 4 |]
    [| 1.; 3.; 0.; 0.; 2.; 2.; 4.; 0. |]
    (over_i call_ragged (ragged_doc_args ()));
  same "doc example jit" Dtype.I32 [| 2; 4 |]
    [| 1.; 3.; 0.; 0.; 2.; 2.; 4.; 0. |]
    (jitn (over_i call_ragged) (ragged_doc_args ()))

let ragged_all_to_all_rows () =
  let operand =
    arr Dtype.F32 [| 2; 2; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let output = arr Dtype.F32 [| 2; 2; 2 |] (Array.make 8 (-1.)) in
  let zero = ints [| 2; 2 |] [| 0.; 1.; 0.; 1. |] in
  let ones = ints [| 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  let out_off = ints [| 2; 2 |] [| 0.; 0.; 1.; 1. |] in
  same "row slices" Dtype.F32 [| 2; 2; 2 |]
    [| 1.; 2.; 5.; 6.; 3.; 4.; 7.; 8. |]
    (over_i call_ragged [ operand; output; zero; ones; out_off; ones ]);
  let keep = ints [| 2; 2 |] [| 0.; 0.; 0.; 0. |] in
  same "zero sizes keep output" Dtype.F32 [| 2; 2; 2 |] (Array.make 8 (-1.))
    (over_i call_ragged [ operand; output; zero; keep; out_off; keep ])

let ragged_all_to_all_errors () =
  let v = ints [| 2 |] [| 0.; 0. |] in
  raises "outside map" "mapped context" (fun () ->
      P.ragged_all_to_all v v v v v v ~axis_name:"i");
  raises "float offsets" "must be integer" (fun () ->
      let f = arr Dtype.F32 [| 1 |] [| 0. |] in
      P.ragged_all_to_all v v f f f f ~axis_name:"i");
  raises "rank" "must be rank 1" (fun () ->
      let s = ints [||] [| 0. |] in
      P.ragged_all_to_all v v s s s s ~axis_name:"i");
  raises "slots" "divisible" (fun () ->
      over_i
        (function
          | [ o; p; a ] -> P.ragged_all_to_all o p a a a a ~axis_name:"i"
          | _ -> Alcotest.fail "arity")
        [
          ints [| 2; 2 |] [| 0.; 0.; 0.; 0. |];
          ints [| 2; 2 |] [| 0.; 0.; 0.; 0. |];
          ints [| 2; 3 |] (Array.make 6 0.);
        ])

let rounding () =
  let x =
    arr Dtype.F32 [| 12 |]
      [| -2.5; -1.5; -1.0; -0.5; -0.25; 0.0; 0.25; 0.5; 1.0; 1.5; 2.5; 3.5 |]
  in
  same "even" Dtype.F32 [| 12 |]
    [| -2.0; -2.0; -1.0; -0.0; -0.0; 0.0; 0.0; 0.0; 1.0; 2.0; 2.0; 4.0 |]
    (L.round ~rounding_method:C.To_nearest_even x);
  same "afz" Dtype.F32 [| 12 |]
    [| -3.0; -2.0; -1.0; -1.0; -0.0; 0.0; 0.0; 1.0; 1.0; 2.0; 3.0; 4.0 |]
    (L.round ~rounding_method:C.Away_from_zero x);
  same "default" Dtype.F32 [| 12 |]
    [| -3.0; -2.0; -1.0; -1.0; -0.0; 0.0; 0.0; 1.0; 1.0; 2.0; 3.0; 4.0 |]
    (L.round x);
  same "even jit" Dtype.F32 [| 12 |]
    [| -2.0; -2.0; -1.0; -0.0; -0.0; 0.0; 0.0; 0.0; 1.0; 2.0; 2.0; 4.0 |]
    (jit1 (L.round ~rounding_method:C.To_nearest_even) x);
  same "even vmap" Dtype.F32 [| 12 |]
    [| -2.0; -2.0; -1.0; -0.0; -0.0; 0.0; 0.0; 0.0; 1.0; 2.0; 2.0; 4.0 |]
    (vmap1 (L.round ~rounding_method:C.To_nearest_even) x);
  same "even jvp" Dtype.F32 [| 12 |]
    [| 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |]
    (snd
       (jvp1 (L.round ~rounding_method:C.To_nearest_even) x (L.full_like x 1.0)));
  same "even grad" Dtype.F32 [| 12 |]
    [| 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0 |]
    (grad1
       (fun v ->
         L.reduce_sum (L.round ~rounding_method:C.To_nearest_even v) [| 0 |])
       x);
  Alcotest.(check bool)
    "negative zero" true
    (Float.sign_bit
       (Ndarray.get_f
          (nd
             (L.round ~rounding_method:C.To_nearest_even
                (arr Dtype.F32 [| 1 |] [| -0.5 |])))
          [| 0 |]));
  ()

let dot_precision () =
  let a =
    arr Dtype.F32 [| 2; 2 |]
      [|
        1.371337890625; 2.745342969894409; 4.120686054229736; 5.501372337341309;
      |]
  and b =
    arr Dtype.F32 [| 2; 2 |]
      [|
        3.246999979019165;
        0.5730000138282776;
        2.1010000705718994;
        2.482999801635742;
      |]
  and a16 =
    arr Dtype.F16 [| 2; 2 |] [| 3.302734375; 6.61328125; 9.921875; 13.25 |]
  and b16 =
    arr Dtype.F16 [| 2; 2 |] [| 13.09375; 2.310546875; 8.46875; 10.0078125 |]
  in
  let ab = L.convert_element_type a16 Dtype.Bf16
  and bb = L.convert_element_type b16 Dtype.Bf16 in
  let ps = C.precision_of_string in
  same "pref f16 from f32" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~preferred_element_type:Dtype.F16 a b);
  same "pref f32 from f16" Dtype.F32 [| 2; 2 |]
    [|
      99.25140380859375; 73.81559753417969; 242.12548828125; 155.52847290039062;
    |]
    (L.dot ~preferred_element_type:Dtype.F32 a16 b16);
  same "pref bf16 from f16" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~preferred_element_type:Dtype.Bf16 a16 b16);
  same "pref f32 from bf16" Dtype.F32 [| 2; 2 |]
    [| 99.583984375; 73.8740234375; 243.0546875; 155.48046875 |]
    (L.dot ~preferred_element_type:Dtype.F32 ab bb);
  same "mixed" Dtype.F32 [| 2; 2 |]
    [|
      24.61848258972168;
      18.313241958618164;
      60.0545768737793;
      38.584983825683594;
    |]
    (L.dot a16 b);
  same "pref f16 from bf16" Dtype.F16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~preferred_element_type:Dtype.F16 ab bb);
  same "pref f32 from i32" Dtype.F32 [| 2; 2 |]
    [| 7.0; 10.0; 15.0; 22.0 |]
    (L.dot ~preferred_element_type:Dtype.F32
       (arr Dtype.I32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |])
       (arr Dtype.I32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]));
  raises "narrow int" "must not be narrower" (fun () ->
      L.dot ~preferred_element_type:Dtype.I16
        (arr Dtype.I32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |])
        (arr Dtype.I32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]));
  same "level default" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(C.precision_of_level C.Precision_default) a b);
  same "string default" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "default") a b);
  same "level high" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(C.precision_of_level C.Precision_high) a b);
  same "string high" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "high") a b);
  same "level highest" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(C.precision_of_level C.Precision_highest) a b);
  same "string highest" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "highest") a b);
  same "pair" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(C.Precision (C.Precision_high, C.Precision_highest)) a b);
  same "f32 DEFAULT" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "DEFAULT") a b);
  same "f16 DEFAULT" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "DEFAULT") a16 b16);
  same "bf16 DEFAULT" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "DEFAULT") ab bb);
  same "pref f16 DEFAULT" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "DEFAULT") ~preferred_element_type:Dtype.F16 a b);
  same "jit DEFAULT" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "DEFAULT") u v | _ -> assert false)
       [ a; b ]);
  same "f32 F16_F16_F16" Dtype.F32 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "F16_F16_F16") a b);
  same "f16 F16_F16_F16" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "F16_F16_F16") a16 b16);
  same "bf16 F16_F16_F16" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 156.0 |]
    (L.dot ~precision:(ps "F16_F16_F16") ab bb);
  same "pref f16 F16_F16_F16" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "F16_F16_F16") ~preferred_element_type:Dtype.F16 a b);
  same "jit F16_F16_F16" Dtype.F32 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "F16_F16_F16") u v
         | _ -> assert false)
       [ a; b ]);
  same "f32 BF16_BF16_F32" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "BF16_BF16_F32") a b);
  same "f16 BF16_BF16_F32" Dtype.F16 [| 2; 2 |]
    [| 99.5625; 73.875; 243.0; 155.5 |]
    (L.dot ~precision:(ps "BF16_BF16_F32") a16 b16);
  same "bf16 BF16_BF16_F32" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "BF16_BF16_F32") ab bb);
  same "pref f16 BF16_BF16_F32" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "BF16_BF16_F32") ~preferred_element_type:Dtype.F16 a b);
  same "jit BF16_BF16_F32" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "BF16_BF16_F32") u v
         | _ -> assert false)
       [ a; b ]);
  same "f32 BF16_BF16_F32_X3" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X3") a b);
  same "f16 BF16_BF16_F32_X3" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X3") a16 b16);
  same "bf16 BF16_BF16_F32_X3" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X3") ab bb);
  same "pref f16 BF16_BF16_F32_X3" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X3") ~preferred_element_type:Dtype.F16
       a b);
  same "jit BF16_BF16_F32_X3" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "BF16_BF16_F32_X3") u v
         | _ -> assert false)
       [ a; b ]);
  same "f32 BF16_BF16_F32_X6" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X6") a b);
  same "f16 BF16_BF16_F32_X6" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X6") a16 b16);
  same "bf16 BF16_BF16_F32_X6" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X6") ab bb);
  same "pref f16 BF16_BF16_F32_X6" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "BF16_BF16_F32_X6") ~preferred_element_type:Dtype.F16
       a b);
  same "jit BF16_BF16_F32_X6" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "BF16_BF16_F32_X6") u v
         | _ -> assert false)
       [ a; b ]);
  same "f32 F32_F32_F32" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "F32_F32_F32") a b);
  same "f16 F32_F32_F32" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "F32_F32_F32") a16 b16);
  same "bf16 F32_F32_F32" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "F32_F32_F32") ab bb);
  same "pref f16 F32_F32_F32" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "F32_F32_F32") ~preferred_element_type:Dtype.F16 a b);
  same "jit F32_F32_F32" Dtype.F32 [| 2; 2 |]
    [|
      10.220699310302734;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "F32_F32_F32") u v
         | _ -> assert false)
       [ a; b ]);
  same "f32 F64_F64_F64" Dtype.F32 [| 2; 2 |]
    [|
      10.22070026397705;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (L.dot ~precision:(ps "F64_F64_F64") a b);
  same "f16 F64_F64_F64" Dtype.F16 [| 2; 2 |]
    [| 99.25; 73.8125; 242.125; 155.5 |]
    (L.dot ~precision:(ps "F64_F64_F64") a16 b16);
  same "bf16 F64_F64_F64" Dtype.Bf16 [| 2; 2 |]
    [| 99.5; 74.0; 243.0; 155.0 |]
    (L.dot ~precision:(ps "F64_F64_F64") ab bb);
  same "pref f16 F64_F64_F64" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot ~precision:(ps "F64_F64_F64") ~preferred_element_type:Dtype.F16 a b);
  same "jit F64_F64_F64" Dtype.F32 [| 2; 2 |]
    [|
      10.22070026397705;
      7.6024627685546875;
      24.938251495361328;
      16.021059036254883;
    |]
    (jitn
       (function
         | [ u; v ] -> L.dot ~precision:(ps "F64_F64_F64") u v
         | _ -> assert false)
       [ a; b ]);
  raises "cpu rejects ANY_F8_ANY_F8_F32" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "ANY_F8_ANY_F8_F32") a b);
  raises "cpu rejects ANY_F8_ANY_F8_F32_FAST_ACCUM"
    "not supported by dot_general on CPU" (fun () ->
      L.dot ~precision:(ps "ANY_F8_ANY_F8_F32_FAST_ACCUM") a b);
  raises "cpu rejects ANY_F8_ANY_F8_ANY" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "ANY_F8_ANY_F8_ANY") a b);
  raises "cpu rejects ANY_F8_ANY_F8_ANY_FAST_ACCUM"
    "not supported by dot_general on CPU" (fun () ->
      L.dot ~precision:(ps "ANY_F8_ANY_F8_ANY_FAST_ACCUM") a b);
  raises "cpu rejects F16_F16_F32" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "F16_F16_F32") a b);
  raises "cpu rejects BF16_BF16_BF16" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "BF16_BF16_BF16") a b);
  raises "cpu rejects BF16_BF16_F32_X9" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "BF16_BF16_F32_X9") a b);
  raises "cpu rejects TF32_TF32_F32" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "TF32_TF32_F32") a b);
  raises "cpu rejects TF32_TF32_F32_X3" "not supported by dot_general on CPU"
    (fun () -> L.dot ~precision:(ps "TF32_TF32_F32_X3") a b);
  raises "cpu rejects algorithm" "not supported by dot_general on CPU"
    (fun () ->
      L.dot
        ~precision:
          (C.Algorithm
             (C.dot_algorithm ~lhs_precision_type:Dtype.F32
                ~rhs_precision_type:Dtype.F32 ~accumulation_type:Dtype.F32 ()))
        a b);
  raises "bad string" "Precision argument" (fun () -> ps "nope");
  same "grad preset" Dtype.F32 [| 2; 2 |]
    [| 3.8203125; 4.5859375; 3.8203125; 4.5859375 |]
    (grad1
       (fun u ->
         L.reduce_sum (L.dot ~precision:(ps "F16_F16_F16") u b) [| 0; 1 |])
       a);
  same "dot_general" Dtype.F16 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (L.dot_general ~precision:(ps "F32_F32_F32")
       ~preferred_element_type:Dtype.F16 a b ~lhs_contract:[| 1 |]
       ~rhs_contract:[| 0 |] ~lhs_batch:[||] ~rhs_batch:[||]);
  same "jnp dot preset" Dtype.F32 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (TC.dot ~precision:(ps "F16_F16_F16") a b);
  same "jnp matmul preset" Dtype.F32 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (TC.matmul ~precision:(ps "F16_F16_F16") a b);
  same "jnp tensordot preset" Dtype.F32 [| 2; 2 |]
    [| 10.21875; 7.6015625; 24.9375; 16.015625 |]
    (TC.tensordot ~precision:(ps "F16_F16_F16") ~axes:(TC.Ax_int 1) a b);
  same "jnp inner preset" Dtype.F32 [| 2; 2 |]
    [| 6.0234375; 9.6953125; 16.53125; 22.3125 |]
    (TC.inner ~precision:(ps "F16_F16_F16") a b);
  same "jnp vdot preset" Dtype.F32 [||] [| 28.343284606933594 |]
    (TC.vdot ~precision:(ps "F32_F32_F32") a b);
  same "jnp vecdot preset" Dtype.F32 [| 2 |] [| 6.0234375; 22.3125 |]
    (TC.vecdot ~precision:(ps "F16_F16_F16") a b);
  raises "jnp dot rejects" "not supported by dot_general on CPU" (fun () ->
      TC.dot ~precision:(ps "TF32_TF32_F32") a b);
  ()

let accuracy () =
  let x = arr Dtype.F32 [| 4 |] [| 0.25; 0.5; 1.25; 2.5 |] in
  let tol = C.Tolerance (C.tolerance ~atol:1e-3 ())
  and hi = C.Mode C.Accuracy_highest
  and de = C.Mode C.Accuracy_default in
  same "exp tol" Dtype.F32 [| 4 |]
    [|
      1.2840254306793213;
      1.6487212181091309;
      3.490342855453491;
      12.182494163513184;
    |]
    (L.exp ~accuracy:tol x);
  same "exp highest" Dtype.F32 [| 4 |]
    [|
      1.2840254306793213;
      1.6487212181091309;
      3.490342855453491;
      12.182494163513184;
    |]
    (L.exp ~accuracy:hi x);
  same "exp default" Dtype.F32 [| 4 |]
    [|
      1.2840254306793213;
      1.6487212181091309;
      3.490342855453491;
      12.182494163513184;
    |]
    (L.exp ~accuracy:de x);
  same "exp jit" Dtype.F32 [| 4 |]
    [|
      1.2840254306793213;
      1.6487212181091309;
      3.490342855453491;
      12.182494163513184;
    |]
    (jit1 (L.exp ~accuracy:tol) x);
  same "exp2 tol" Dtype.F32 [| 4 |]
    [|
      1.1892070770263672;
      1.4142135381698608;
      2.3784141540527344;
      5.656854152679443;
    |]
    (L.exp2 ~accuracy:tol x);
  same "exp2 highest" Dtype.F32 [| 4 |]
    [|
      1.1892070770263672;
      1.4142135381698608;
      2.3784141540527344;
      5.656854152679443;
    |]
    (L.exp2 ~accuracy:hi x);
  same "exp2 default" Dtype.F32 [| 4 |]
    [|
      1.1892070770263672;
      1.4142135381698608;
      2.3784141540527344;
      5.656854152679443;
    |]
    (L.exp2 ~accuracy:de x);
  same "exp2 jit" Dtype.F32 [| 4 |]
    [|
      1.1892070770263672;
      1.4142135381698608;
      2.3784141540527344;
      5.656854152679443;
    |]
    (jit1 (L.exp2 ~accuracy:tol) x);
  same "expm1 tol" Dtype.F32 [| 4 |]
    [|
      0.2840254008769989;
      0.6487212777137756;
      2.490342855453491;
      11.182494163513184;
    |]
    (L.expm1 ~accuracy:tol x);
  same "expm1 highest" Dtype.F32 [| 4 |]
    [|
      0.2840254008769989;
      0.6487212777137756;
      2.490342855453491;
      11.182494163513184;
    |]
    (L.expm1 ~accuracy:hi x);
  same "expm1 default" Dtype.F32 [| 4 |]
    [|
      0.2840254008769989;
      0.6487212777137756;
      2.490342855453491;
      11.182494163513184;
    |]
    (L.expm1 ~accuracy:de x);
  same "expm1 jit" Dtype.F32 [| 4 |]
    [|
      0.2840254008769989;
      0.6487212777137756;
      2.490342855453491;
      11.182494163513184;
    |]
    (jit1 (L.expm1 ~accuracy:tol) x);
  same "log tol" Dtype.F32 [| 4 |]
    [|
      -1.3862943649291992;
      -0.6931471824645996;
      0.2231435477733612;
      0.9162907600402832;
    |]
    (L.log ~accuracy:tol x);
  same "log highest" Dtype.F32 [| 4 |]
    [|
      -1.3862943649291992;
      -0.6931471824645996;
      0.2231435477733612;
      0.9162907600402832;
    |]
    (L.log ~accuracy:hi x);
  same "log default" Dtype.F32 [| 4 |]
    [|
      -1.3862943649291992;
      -0.6931471824645996;
      0.2231435477733612;
      0.9162907600402832;
    |]
    (L.log ~accuracy:de x);
  same "log jit" Dtype.F32 [| 4 |]
    [|
      -1.3862943649291992;
      -0.6931471824645996;
      0.2231435477733612;
      0.9162907600402832;
    |]
    (jit1 (L.log ~accuracy:tol) x);
  same "log1p tol" Dtype.F32 [| 4 |]
    [|
      0.2231435477733612;
      0.40546509623527527;
      0.8109301924705505;
      1.252763032913208;
    |]
    (L.log1p ~accuracy:tol x);
  same "log1p highest" Dtype.F32 [| 4 |]
    [|
      0.2231435477733612;
      0.40546509623527527;
      0.8109301924705505;
      1.252763032913208;
    |]
    (L.log1p ~accuracy:hi x);
  same "log1p default" Dtype.F32 [| 4 |]
    [|
      0.2231435477733612;
      0.40546509623527527;
      0.8109301924705505;
      1.252763032913208;
    |]
    (L.log1p ~accuracy:de x);
  same "log1p jit" Dtype.F32 [| 4 |]
    [|
      0.2231435477733612;
      0.40546509623527527;
      0.8109301924705505;
      1.252763032913208;
    |]
    (jit1 (L.log1p ~accuracy:tol) x);
  same "tanh tol" Dtype.F32 [| 4 |]
    [|
      0.24491865932941437;
      0.46211716532707214;
      0.8482836484909058;
      0.9866143465042114;
    |]
    (L.tanh ~accuracy:tol x);
  same "tanh highest" Dtype.F32 [| 4 |]
    [|
      0.24491865932941437;
      0.46211716532707214;
      0.8482836484909058;
      0.9866143465042114;
    |]
    (L.tanh ~accuracy:hi x);
  same "tanh default" Dtype.F32 [| 4 |]
    [|
      0.24491865932941437;
      0.46211716532707214;
      0.8482836484909058;
      0.9866143465042114;
    |]
    (L.tanh ~accuracy:de x);
  same "tanh jit" Dtype.F32 [| 4 |]
    [|
      0.24491865932941437;
      0.46211716532707214;
      0.8482836484909058;
      0.9866143465042114;
    |]
    (jit1 (L.tanh ~accuracy:tol) x);
  same "logistic tol" Dtype.F32 [| 4 |]
    [|
      0.562176525592804;
      0.622459352016449;
      0.7772998809814453;
      0.9241418242454529;
    |]
    (L.logistic ~accuracy:tol x);
  same "logistic highest" Dtype.F32 [| 4 |]
    [|
      0.562176525592804;
      0.622459352016449;
      0.7772998809814453;
      0.9241418242454529;
    |]
    (L.logistic ~accuracy:hi x);
  same "logistic default" Dtype.F32 [| 4 |]
    [|
      0.562176525592804;
      0.622459352016449;
      0.7772998809814453;
      0.9241418242454529;
    |]
    (L.logistic ~accuracy:de x);
  same "logistic jit" Dtype.F32 [| 4 |]
    [|
      0.562176525592804;
      0.622459352016449;
      0.7772998809814453;
      0.9241418242454529;
    |]
    (jit1 (L.logistic ~accuracy:tol) x);
  same "sin tol" Dtype.F32 [| 4 |]
    [|
      0.24740396440029144;
      0.4794255495071411;
      0.9489846229553223;
      0.5984721183776855;
    |]
    (L.sin ~accuracy:tol x);
  same "sin highest" Dtype.F32 [| 4 |]
    [|
      0.24740396440029144;
      0.4794255495071411;
      0.9489846229553223;
      0.5984721183776855;
    |]
    (L.sin ~accuracy:hi x);
  same "sin default" Dtype.F32 [| 4 |]
    [|
      0.24740396440029144;
      0.4794255495071411;
      0.9489846229553223;
      0.5984721183776855;
    |]
    (L.sin ~accuracy:de x);
  same "sin jit" Dtype.F32 [| 4 |]
    [|
      0.24740396440029144;
      0.4794255495071411;
      0.9489846229553223;
      0.5984721183776855;
    |]
    (jit1 (L.sin ~accuracy:tol) x);
  same "cos tol" Dtype.F32 [| 4 |]
    [|
      0.9689124226570129;
      0.8775825500488281;
      0.3153223693370819;
      -0.8011435866355896;
    |]
    (L.cos ~accuracy:tol x);
  same "cos highest" Dtype.F32 [| 4 |]
    [|
      0.9689124226570129;
      0.8775825500488281;
      0.3153223693370819;
      -0.8011435866355896;
    |]
    (L.cos ~accuracy:hi x);
  same "cos default" Dtype.F32 [| 4 |]
    [|
      0.9689124226570129;
      0.8775825500488281;
      0.3153223693370819;
      -0.8011435866355896;
    |]
    (L.cos ~accuracy:de x);
  same "cos jit" Dtype.F32 [| 4 |]
    [|
      0.9689124226570129;
      0.8775825500488281;
      0.3153223693370819;
      -0.8011435866355896;
    |]
    (jit1 (L.cos ~accuracy:tol) x);
  same "tan tol" Dtype.F32 [| 4 |]
    [|
      0.25534191727638245;
      0.5463024973869324;
      3.0095696449279785;
      -0.7470223307609558;
    |]
    (L.tan ~accuracy:tol x);
  same "tan highest" Dtype.F32 [| 4 |]
    [|
      0.25534191727638245;
      0.5463024973869324;
      3.0095696449279785;
      -0.7470223307609558;
    |]
    (L.tan ~accuracy:hi x);
  same "tan default" Dtype.F32 [| 4 |]
    [|
      0.25534191727638245;
      0.5463024973869324;
      3.0095696449279785;
      -0.7470223307609558;
    |]
    (L.tan ~accuracy:de x);
  same "tan jit" Dtype.F32 [| 4 |]
    [|
      0.25534191727638245;
      0.5463024973869324;
      3.0095696449279785;
      -0.7470223307609558;
    |]
    (jit1 (L.tan ~accuracy:tol) x);
  same "sqrt tol" Dtype.F32 [| 4 |]
    [| 0.5; 0.7071067690849304; 1.1180340051651; 1.5811388492584229 |]
    (L.sqrt ~accuracy:tol x);
  same "sqrt highest" Dtype.F32 [| 4 |]
    [| 0.5; 0.7071067690849304; 1.1180340051651; 1.5811388492584229 |]
    (L.sqrt ~accuracy:hi x);
  same "sqrt default" Dtype.F32 [| 4 |]
    [| 0.5; 0.7071067690849304; 1.1180340051651; 1.5811388492584229 |]
    (L.sqrt ~accuracy:de x);
  same "sqrt jit" Dtype.F32 [| 4 |]
    [| 0.5; 0.7071067690849304; 1.1180340051651; 1.5811388492584229 |]
    (jit1 (L.sqrt ~accuracy:tol) x);
  same "rsqrt tol" Dtype.F32 [| 4 |]
    [| 2.0; 1.4142135381698608; 0.8944271802902222; 0.6324555277824402 |]
    (L.rsqrt ~accuracy:tol x);
  same "rsqrt highest" Dtype.F32 [| 4 |]
    [| 2.0; 1.4142135381698608; 0.8944271802902222; 0.6324555277824402 |]
    (L.rsqrt ~accuracy:hi x);
  same "rsqrt default" Dtype.F32 [| 4 |]
    [| 2.0; 1.4142135381698608; 0.8944271802902222; 0.6324555277824402 |]
    (L.rsqrt ~accuracy:de x);
  same "rsqrt jit" Dtype.F32 [| 4 |]
    [| 2.0; 1.4142135381698608; 0.8944271802902222; 0.6324555277824402 |]
    (jit1 (L.rsqrt ~accuracy:tol) x);
  same "cbrt tol" Dtype.F32 [| 4 |]
    [|
      0.6299605369567871;
      0.7937005162239075;
      1.0772173404693604;
      1.3572088479995728;
    |]
    (L.cbrt ~accuracy:tol x);
  same "cbrt highest" Dtype.F32 [| 4 |]
    [|
      0.6299605369567871;
      0.7937005162239075;
      1.0772173404693604;
      1.3572088479995728;
    |]
    (L.cbrt ~accuracy:hi x);
  same "cbrt default" Dtype.F32 [| 4 |]
    [|
      0.6299605369567871;
      0.7937005162239075;
      1.0772173404693604;
      1.3572088479995728;
    |]
    (L.cbrt ~accuracy:de x);
  same "cbrt jit" Dtype.F32 [| 4 |]
    [|
      0.6299605369567871;
      0.7937005162239075;
      1.0772173404693604;
      1.3572088479995728;
    |]
    (jit1 (L.cbrt ~accuracy:tol) x);
  raises "tol empty" "At least one" (fun () -> C.tolerance ());
  raises "tol neg" "non-negative" (fun () -> C.tolerance ~atol:(-1.0) ());
  raises "tol neg ulps" "non-negative" (fun () -> C.tolerance ~ulps:(-1) ());
  let t = C.tolerance ~rtol:0.5 ~ulps:2 () in
  Alcotest.(check (float 0.0)) "rtol" 0.5 t.C.rtol;
  Alcotest.(check int) "ulps" 2 t.C.ulps;
  ()

let gather_modes () =
  let x =
    arr Dtype.F32 [| 10 |]
      [| 10.0; 11.0; 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0 |]
  and idx = arr Dtype.I32 [| 5; 1 |] [| 0.0; 9.0; 10.0; -1.0; -5.0 |] in
  let dn =
    C.gather_dimension_numbers ~offset_dims:[||] ~collapsed_slice_dims:[| 0 |]
      ~start_index_map:[| 0 |] ()
  in
  same "gather clip" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (L.gather ~mode:C.Clip x idx dn [| 1 |]);
  same "gather string clip" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (L.gather ~mode:(C.gather_scatter_mode_of_string "clip") x idx dn [| 1 |]);
  same "gather jit clip" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (jit1 (fun v -> L.gather ~mode:C.Clip v idx dn [| 1 |]) x);
  same "gather vmap clip" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (vmap1 (fun i -> L.gather ~mode:C.Clip x i dn [| 1 |]) idx);
  same "gather fill" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; nan; nan; nan |]
    (L.gather ~mode:C.Fill_or_drop x idx dn [| 1 |]);
  same "gather string fill" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; nan; nan; nan |]
    (L.gather ~mode:(C.gather_scatter_mode_of_string "fill") x idx dn [| 1 |]);
  same "gather jit fill" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; nan; nan; nan |]
    (jit1 (fun v -> L.gather ~mode:C.Fill_or_drop v idx dn [| 1 |]) x);
  same "gather vmap fill" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; nan; nan; nan |]
    (vmap1 (fun i -> L.gather ~mode:C.Fill_or_drop x i dn [| 1 |]) idx);
  same "gather promise_in_bounds" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (L.gather ~mode:C.Promise_in_bounds x idx dn [| 1 |]);
  same "gather string promise_in_bounds" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (L.gather
       ~mode:(C.gather_scatter_mode_of_string "promise_in_bounds")
       x idx dn [| 1 |]);
  same "gather jit promise_in_bounds" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (jit1 (fun v -> L.gather ~mode:C.Promise_in_bounds v idx dn [| 1 |]) x);
  same "gather vmap promise_in_bounds" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (vmap1 (fun i -> L.gather ~mode:C.Promise_in_bounds x i dn [| 1 |]) idx);
  same "gather default" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; 19.0; 10.0; 10.0 |]
    (L.gather x idx dn [| 1 |]);
  same "gather drop string" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; nan; nan; nan |]
    (L.gather ~mode:(C.gather_scatter_mode_of_string "drop") x idx dn [| 1 |]);
  same "gather fill value" Dtype.F32 [| 5 |]
    [| 10.0; 19.0; -1.0; -1.0; -1.0 |]
    (L.gather ~mode:C.Fill_or_drop ~fill_value:(-1.0) x idx dn [| 1 |]);
  same "gather fill int" Dtype.I32 [| 5 |]
    [| 0.0; -2147483648.0; -2147483648.0; -2147483648.0; -2147483648.0 |]
    (L.gather ~mode:C.Fill_or_drop
       (arr Dtype.I32 [| 5 |] [| 0.0; 1.0; 2.0; 3.0; 4.0 |])
       idx dn [| 1 |]);
  same "gather fill bool" Dtype.Bool [| 5 |]
    [| 0.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.gather ~mode:C.Fill_or_drop
       (arr Dtype.Bool [| 5 |] [| 0.0; 0.0; 1.0; 1.0; 1.0 |])
       idx dn [| 1 |]);
  same "gather fill u8" Dtype.Uint8 [| 5 |]
    [| 0.0; 255.0; 255.0; 255.0; 255.0 |]
    (L.gather ~mode:C.Fill_or_drop
       (arr Dtype.Uint8 [| 5 |] [| 0.0; 1.0; 2.0; 3.0; 4.0 |])
       idx dn [| 1 |]);
  same "grad fill" Dtype.F32 [| 10 |]
    [| 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0 |]
    (grad1
       (fun v ->
         L.reduce_sum
           (L.gather ~mode:C.Fill_or_drop ~fill_value:0.0 v idx dn [| 1 |])
           [| 0 |])
       x);
  same "grad clip" Dtype.F32 [| 10 |]
    [| 3.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 2.0 |]
    (grad1
       (fun v -> L.reduce_sum (L.gather ~mode:C.Clip v idx dn [| 1 |]) [| 0 |])
       x);
  let win =
    C.gather_dimension_numbers ~offset_dims:[| 1 |] ~collapsed_slice_dims:[||]
      ~start_index_map:[| 0 |] ()
  in
  same "window fill" Dtype.F32 [| 3; 2 |]
    [| 10.0; 11.0; 18.0; 19.0; -1.0; -1.0 |]
    (L.gather ~mode:C.Fill_or_drop ~fill_value:(-1.0) x
       (arr Dtype.I32 [| 3; 1 |] [| 0.0; 8.0; 9.0 |])
       win [| 2 |]);
  same "window clip" Dtype.F32 [| 3; 2 |]
    [| 10.0; 11.0; 18.0; 19.0; 18.0; 19.0 |]
    (L.gather ~mode:C.Clip x
       (arr Dtype.I32 [| 3; 1 |] [| 0.0; 8.0; 9.0 |])
       win [| 2 |]);
  raises "one_hot gather" "ONE_HOT" (fun () ->
      L.gather ~mode:C.One_hot x idx dn [| 1 |]);
  raises "bad mode" "Unknown gather mode" (fun () ->
      C.gather_scatter_mode_of_string "wrong");
  ()

let scatter_modes () =
  let x = arr Dtype.F32 [| 5 |] [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
  and u = arr Dtype.F32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
  and si = arr Dtype.I32 [| 5; 1 |] [| 0.0; 4.0; 5.0; -1.0; 2.0 |] in
  let sd =
    C.scatter_dimension_numbers ~update_window_dims:[||]
      ~inserted_window_dims:[| 0 |] ~scatter_dims_to_operand_dims:[| 0 |] ()
  in
  same "scatter default" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter x si u sd);
  same "scatter clip" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 3.0 |]
    (L.scatter ~mode:C.Clip x si u sd);
  same "scatter fill" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter ~mode:C.Fill_or_drop x si u sd);
  same "scatter promise_in_bounds" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter ~mode:C.Promise_in_bounds x si u sd);
  same "scatter one_hot" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter ~mode:C.One_hot x si u sd);
  same "scatter clip jit" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 3.0 |]
    (jit1 (fun v -> L.scatter ~mode:C.Clip v si u sd) x);
  same "scatter fill jit" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (jit1 (fun v -> L.scatter ~mode:C.Fill_or_drop v si u sd) x);
  same "scatter_add default" Dtype.F32 [| 5 |]
    [| 2.0; 1.0; 6.0; 1.0; 3.0 |]
    (L.scatter_add x si u sd);
  same "scatter_add clip" Dtype.F32 [| 5 |]
    [| 6.0; 1.0; 6.0; 1.0; 6.0 |]
    (L.scatter_add ~mode:C.Clip x si u sd);
  same "scatter_add fill" Dtype.F32 [| 5 |]
    [| 2.0; 1.0; 6.0; 1.0; 3.0 |]
    (L.scatter_add ~mode:C.Fill_or_drop x si u sd);
  same "scatter_add promise_in_bounds" Dtype.F32 [| 5 |]
    [| 2.0; 1.0; 6.0; 1.0; 3.0 |]
    (L.scatter_add ~mode:C.Promise_in_bounds x si u sd);
  same "scatter_add one_hot" Dtype.F32 [| 5 |]
    [| 2.0; 1.0; 6.0; 1.0; 3.0 |]
    (L.scatter_add ~mode:C.One_hot x si u sd);
  same "scatter_add clip jit" Dtype.F32 [| 5 |]
    [| 6.0; 1.0; 6.0; 1.0; 6.0 |]
    (jit1 (fun v -> L.scatter_add ~mode:C.Clip v si u sd) x);
  same "scatter_add fill jit" Dtype.F32 [| 5 |]
    [| 2.0; 1.0; 6.0; 1.0; 3.0 |]
    (jit1 (fun v -> L.scatter_add ~mode:C.Fill_or_drop v si u sd) x);
  same "scatter_sub default" Dtype.F32 [| 5 |]
    [| 0.0; 1.0; -4.0; 1.0; -1.0 |]
    (L.scatter_sub x si u sd);
  same "scatter_sub clip" Dtype.F32 [| 5 |]
    [| -4.0; 1.0; -4.0; 1.0; -4.0 |]
    (L.scatter_sub ~mode:C.Clip x si u sd);
  same "scatter_sub fill" Dtype.F32 [| 5 |]
    [| 0.0; 1.0; -4.0; 1.0; -1.0 |]
    (L.scatter_sub ~mode:C.Fill_or_drop x si u sd);
  same "scatter_sub promise_in_bounds" Dtype.F32 [| 5 |]
    [| 0.0; 1.0; -4.0; 1.0; -1.0 |]
    (L.scatter_sub ~mode:C.Promise_in_bounds x si u sd);
  same "scatter_sub one_hot" Dtype.F32 [| 5 |]
    [| 0.0; 1.0; -4.0; 1.0; -1.0 |]
    (L.scatter_sub ~mode:C.One_hot x si u sd);
  same "scatter_sub clip jit" Dtype.F32 [| 5 |]
    [| -4.0; 1.0; -4.0; 1.0; -4.0 |]
    (jit1 (fun v -> L.scatter_sub ~mode:C.Clip v si u sd) x);
  same "scatter_sub fill jit" Dtype.F32 [| 5 |]
    [| 0.0; 1.0; -4.0; 1.0; -1.0 |]
    (jit1 (fun v -> L.scatter_sub ~mode:C.Fill_or_drop v si u sd) x);
  same "scatter_mul default" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_mul x si u sd);
  same "scatter_mul clip" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 6.0 |]
    (L.scatter_mul ~mode:C.Clip x si u sd);
  same "scatter_mul fill" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_mul ~mode:C.Fill_or_drop x si u sd);
  same "scatter_mul promise_in_bounds" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_mul ~mode:C.Promise_in_bounds x si u sd);
  same "scatter_mul one_hot" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_mul ~mode:C.One_hot x si u sd);
  same "scatter_mul clip jit" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 6.0 |]
    (jit1 (fun v -> L.scatter_mul ~mode:C.Clip v si u sd) x);
  same "scatter_mul fill jit" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (jit1 (fun v -> L.scatter_mul ~mode:C.Fill_or_drop v si u sd) x);
  same "scatter_min default" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.scatter_min x si u sd);
  same "scatter_min clip" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.scatter_min ~mode:C.Clip x si u sd);
  same "scatter_min fill" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.scatter_min ~mode:C.Fill_or_drop x si u sd);
  same "scatter_min promise_in_bounds" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.scatter_min ~mode:C.Promise_in_bounds x si u sd);
  same "scatter_min one_hot" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (L.scatter_min ~mode:C.One_hot x si u sd);
  same "scatter_min clip jit" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (jit1 (fun v -> L.scatter_min ~mode:C.Clip v si u sd) x);
  same "scatter_min fill jit" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (jit1 (fun v -> L.scatter_min ~mode:C.Fill_or_drop v si u sd) x);
  same "scatter_max default" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_max x si u sd);
  same "scatter_max clip" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 3.0 |]
    (L.scatter_max ~mode:C.Clip x si u sd);
  same "scatter_max fill" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_max ~mode:C.Fill_or_drop x si u sd);
  same "scatter_max promise_in_bounds" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_max ~mode:C.Promise_in_bounds x si u sd);
  same "scatter_max one_hot" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (L.scatter_max ~mode:C.One_hot x si u sd);
  same "scatter_max clip jit" Dtype.F32 [| 5 |]
    [| 4.0; 1.0; 5.0; 1.0; 3.0 |]
    (jit1 (fun v -> L.scatter_max ~mode:C.Clip v si u sd) x);
  same "scatter_max fill jit" Dtype.F32 [| 5 |]
    [| 1.0; 1.0; 5.0; 1.0; 2.0 |]
    (jit1 (fun v -> L.scatter_max ~mode:C.Fill_or_drop v si u sd) x);
  let sw =
    C.scatter_dimension_numbers ~update_window_dims:[| 1 |]
      ~inserted_window_dims:[||] ~scatter_dims_to_operand_dims:[| 0 |] ()
  in
  let z = arr Dtype.F32 [| 5 |] [| 0.0; 0.0; 0.0; 0.0; 0.0 |]
  and wi = arr Dtype.I32 [| 2; 1 |] [| 0.0; 4.0 |]
  and wu = arr Dtype.F32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  same "window clip" Dtype.F32 [| 5 |]
    [| 1.0; 2.0; 0.0; 3.0; 4.0 |]
    (L.scatter ~mode:C.Clip z wi wu sw);
  same "window fill" Dtype.F32 [| 5 |]
    [| 1.0; 2.0; 0.0; 0.0; 0.0 |]
    (L.scatter ~mode:C.Fill_or_drop z wi wu sw);
  ()

let fft_type () =
  let r = arr Dtype.F32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |]
  and ii = arr Dtype.I32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let c =
    Concrete
      (Ndarray.of_complex Dtype.Complex64 [| 3 |]
         [|
           Complex.one;
           { Complex.re = 2.0; im = 0.0 };
           { Complex.re = 3.0; im = 0.0 };
         |])
  in
  let ft = C.fft_type_of_string in
  same_c "fft r" Dtype.Complex64 [| 4 |]
    [| (10.0, 0.0); (-2.0, 2.0); (-2.0, 0.0); (-2.0, -2.0) |]
    (L.fft r ~fft_type:(ft "fft") ~fft_lengths:[| 4 |]);
  same_c "FFT ii" Dtype.Complex64 [| 4 |]
    [| (10.0, 0.0); (-2.0, 2.0); (-2.0, 0.0); (-2.0, -2.0) |]
    (L.fft ii ~fft_type:(ft "FFT") ~fft_lengths:[| 4 |]);
  same_c "ifft r" Dtype.Complex64 [| 4 |]
    [| (2.5, 0.0); (-0.5, -0.5); (-0.5, 0.0); (-0.5, 0.5) |]
    (L.fft r ~fft_type:(ft "ifft") ~fft_lengths:[| 4 |]);
  same_c "IFFT r" Dtype.Complex64 [| 4 |]
    [| (2.5, 0.0); (-0.5, -0.5); (-0.5, 0.0); (-0.5, 0.5) |]
    (L.fft r ~fft_type:(ft "IFFT") ~fft_lengths:[| 4 |]);
  same_c "rfft r" Dtype.Complex64 [| 3 |]
    [| (10.0, 0.0); (-2.0, 2.0); (-2.0, 0.0) |]
    (L.fft r ~fft_type:(ft "rfft") ~fft_lengths:[| 4 |]);
  same_c "RFFT ii" Dtype.Complex64 [| 3 |]
    [| (10.0, 0.0); (-2.0, 2.0); (-2.0, 0.0) |]
    (L.fft ii ~fft_type:(ft "RFFT") ~fft_lengths:[| 4 |]);
  same "irfft" Dtype.F32 [| 4 |] [| 2.0; -0.5; 0.0; -0.5 |]
    (L.fft c ~fft_type:C.Fft_inverse_real ~fft_lengths:[| 4 |]);
  same_c "empty lengths" Dtype.Complex64 [| 4 |]
    [| (1.0, 0.0); (2.0, 0.0); (3.0, 0.0); (4.0, 0.0) |]
    (L.fft ii ~fft_type:C.Fft_forward ~fft_lengths:[||]);
  raises "bad type" "Unknown FFT type" (fun () -> ft "nope");
  raises "rfft complex" "only real valued" (fun () ->
      L.fft c ~fft_type:C.Fft_real ~fft_lengths:[| 3 |]);
  ()

let random_algorithm () =
  if plugin_present () then begin
    let k = arr Dtype.Uint32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
    let nk, vals = X.rng_bit_generator ~algorithm:C.Rng_default k [| 3 |] in
    same "C.Rng_default key" Dtype.Uint32 [| 4 |] [| 1.0; 2.0; 4.0; 4.0 |] nk;
    same "C.Rng_default vals" Dtype.Uint32 [| 3 |]
      [| 512747620.0; 1298009047.0; 1267190206.0 |]
      vals;
    let nk, vals = X.rng_bit_generator ~algorithm:C.Three_fry k [| 3 |] in
    same "C.Three_fry key" Dtype.Uint32 [| 4 |] [| 1.0; 2.0; 5.0; 4.0 |] nk;
    same "C.Three_fry vals" Dtype.Uint32 [| 3 |]
      [| 1576285164.0; 2249660814.0; 740599976.0 |]
      vals;
    let nk, vals = X.rng_bit_generator ~algorithm:C.Philox k [| 3 |] in
    same "C.Philox key" Dtype.Uint32 [| 4 |] [| 1.0; 2.0; 4.0; 4.0 |] nk;
    same "C.Philox vals" Dtype.Uint32 [| 3 |]
      [| 512747620.0; 1298009047.0; 1267190206.0 |]
      vals;
    ignore (nk, vals)
  end;
  ()

let linalg_config () =
  let m =
    arr Dtype.F32 [| 5; 4 |]
      [|
        1.0;
        2.0;
        0.5;
        3.0;
        3.0;
        4.0;
        1.5;
        -1.0;
        5.0;
        7.0;
        2.5;
        2.0;
        0.10000000149011612;
        -3.0;
        9.0;
        1.0;
        2.0;
        2.0;
        2.0;
        2.0;
      |]
  in
  let triple = function
    | [ u; s; vt ] -> (u, s, vt)
    | _ -> Alcotest.fail "svd arity"
  in
  let u, s, vt = triple (LL.svd ~algorithm:C.Svd_qr ~full_matrices:true m) in
  same "Svd_qr full=True s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_qr full=True"
    [|
      -0.24661757051944733;
      -0.054555743932724;
      0.7151603102684021;
      -0.4963671565055847;
      0.42233720421791077;
      -0.411767840385437;
      -0.1275160163640976;
      -0.6231625080108643;
      -0.13623087108135223;
      0.6381979584693909;
      -0.7832533121109009;
      -0.20448562502861023;
      -0.0119100883603096;
      -0.09796877205371857;
      -0.578757643699646;
      -0.21374142169952393;
      0.9672366380691528;
      -0.06303086131811142;
      -0.11754480749368668;
      -0.03128417953848839;
      -0.33234935998916626;
      0.05833299830555916;
      0.3099989593029022;
      0.8435912728309631;
      0.27999234199523926;
    |]
    [|
      -0.5313106179237366;
      -0.667738676071167;
      -0.46173009276390076;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917682647705;
      0.06702674925327301;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660098552703857;
      0.8183994889259338;
      -0.5356988906860352;
      -0.19888824224472046;
      0.060766808688640594;
    |]
    [| 5; 5 |] [| 4; 4 |] u vt;
  let u, s, vt = triple (LL.svd ~algorithm:C.Svd_qr ~full_matrices:false m) in
  same "Svd_qr full=False s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_qr full=False"
    [|
      -0.24661757051944733;
      -0.054555743932724;
      0.7151603102684021;
      -0.4963671565055847;
      -0.411767840385437;
      -0.1275160163640976;
      -0.6231625080108643;
      -0.13623087108135223;
      -0.7832533121109009;
      -0.20448562502861023;
      -0.0119100883603096;
      -0.09796877205371857;
      -0.21374142169952393;
      0.9672366380691528;
      -0.06303086131811142;
      -0.11754480749368668;
      -0.33234935998916626;
      0.05833299830555916;
      0.3099989593029022;
      0.8435912728309631;
    |]
    [|
      -0.5313106179237366;
      -0.667738676071167;
      -0.46173009276390076;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917682647705;
      0.06702674925327301;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660098552703857;
      0.8183994889259338;
      -0.5356988906860352;
      -0.19888824224472046;
      0.060766808688640594;
    |]
    [| 5; 4 |] [| 4; 4 |] u vt;
  same "Svd_qr novec" Dtype.F32 [| 4 |]
    [|
      11.451416969299316;
      9.530213356018066;
      3.4179677963256836;
      0.34277141094207764;
    |]
    (List.hd (LL.svd ~algorithm:C.Svd_qr ~compute_uv:false m));
  let u, s, vt =
    triple (LL.svd ~algorithm:C.Svd_divide_and_conquer ~full_matrices:true m)
  in
  same "Svd_divide_and_conquer full=True s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_divide_and_conquer full=True"
    [|
      -0.24661767482757568;
      -0.054555684328079224;
      0.7151603698730469;
      -0.4963671863079071;
      0.42233720421791077;
      -0.411767840385437;
      -0.1275160312652588;
      -0.623162567615509;
      -0.13623087108135223;
      0.6381979584693909;
      -0.7832533717155457;
      -0.20448558032512665;
      -0.011910038068890572;
      -0.09796877205371857;
      -0.578757643699646;
      -0.21374142169952393;
      0.9672365784645081;
      -0.0630309134721756;
      -0.11754479259252548;
      -0.03128417953848839;
      -0.33234935998916626;
      0.05833302065730095;
      0.309999018907547;
      0.8435912132263184;
      0.27999234199523926;
    |]
    [|
      -0.5313106179237366;
      -0.6677385568618774;
      -0.46173006296157837;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917086601257;
      0.0670267716050148;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660099148750305;
      0.8183994889259338;
      -0.5356989502906799;
      -0.19888825714588165;
      0.06076677888631821;
    |]
    [| 5; 5 |] [| 4; 4 |] u vt;
  let u, s, vt =
    triple (LL.svd ~algorithm:C.Svd_divide_and_conquer ~full_matrices:false m)
  in
  same "Svd_divide_and_conquer full=False s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_divide_and_conquer full=False"
    [|
      -0.24661767482757568;
      -0.054555684328079224;
      0.7151603698730469;
      -0.4963671863079071;
      -0.411767840385437;
      -0.1275160312652588;
      -0.623162567615509;
      -0.13623087108135223;
      -0.7832533717155457;
      -0.20448558032512665;
      -0.011910038068890572;
      -0.09796877205371857;
      -0.21374142169952393;
      0.9672365784645081;
      -0.0630309134721756;
      -0.11754479259252548;
      -0.33234935998916626;
      0.05833302065730095;
      0.309999018907547;
      0.8435912132263184;
    |]
    [|
      -0.5313106179237366;
      -0.6677385568618774;
      -0.46173006296157837;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917086601257;
      0.0670267716050148;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660099148750305;
      0.8183994889259338;
      -0.5356989502906799;
      -0.19888825714588165;
      0.06076677888631821;
    |]
    [| 5; 4 |] [| 4; 4 |] u vt;
  same "Svd_divide_and_conquer novec" Dtype.F32 [| 4 |]
    [|
      11.451416969299316;
      9.530213356018066;
      3.4179677963256836;
      0.34277141094207764;
    |]
    (List.hd (LL.svd ~algorithm:C.Svd_divide_and_conquer ~compute_uv:false m));
  let u, s, vt =
    triple (LL.svd ~algorithm:C.Svd_default ~full_matrices:true m)
  in
  same "Svd_default full=True s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_default full=True"
    [|
      -0.24661767482757568;
      -0.054555684328079224;
      0.7151603698730469;
      -0.4963671863079071;
      0.42233720421791077;
      -0.411767840385437;
      -0.1275160312652588;
      -0.623162567615509;
      -0.13623087108135223;
      0.6381979584693909;
      -0.7832533717155457;
      -0.20448558032512665;
      -0.011910038068890572;
      -0.09796877205371857;
      -0.578757643699646;
      -0.21374142169952393;
      0.9672365784645081;
      -0.0630309134721756;
      -0.11754479259252548;
      -0.03128417953848839;
      -0.33234935998916626;
      0.05833302065730095;
      0.309999018907547;
      0.8435912132263184;
      0.27999234199523926;
    |]
    [|
      -0.5313106179237366;
      -0.6677385568618774;
      -0.46173006296157837;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917086601257;
      0.0670267716050148;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660099148750305;
      0.8183994889259338;
      -0.5356989502906799;
      -0.19888825714588165;
      0.06076677888631821;
    |]
    [| 5; 5 |] [| 4; 4 |] u vt;
  let u, s, vt =
    triple (LL.svd ~algorithm:C.Svd_default ~full_matrices:false m)
  in
  same "Svd_default full=False s" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179675579071045; 0.3427714705467224;
    |]
    s;
  same_uv "Svd_default full=False"
    [|
      -0.24661767482757568;
      -0.054555684328079224;
      0.7151603698730469;
      -0.4963671863079071;
      -0.411767840385437;
      -0.1275160312652588;
      -0.623162567615509;
      -0.13623087108135223;
      -0.7832533717155457;
      -0.20448558032512665;
      -0.011910038068890572;
      -0.09796877205371857;
      -0.21374142169952393;
      0.9672365784645081;
      -0.0630309134721756;
      -0.11754479259252548;
      -0.33234935998916626;
      0.05833302065730095;
      0.309999018907547;
      0.8435912132263184;
    |]
    [|
      -0.5313106179237366;
      -0.6677385568618774;
      -0.46173006296157837;
      -0.24215617775917053;
      -0.13075695931911469;
      -0.5073985457420349;
      0.8490917086601257;
      0.0670267716050148;
      -0.1755959689617157;
      -0.09848245978355408;
      -0.16214825212955475;
      0.9660099148750305;
      0.8183994889259338;
      -0.5356989502906799;
      -0.19888825714588165;
      0.06076677888631821;
    |]
    [| 5; 4 |] [| 4; 4 |] u vt;
  same "Svd_default novec" Dtype.F32 [| 4 |]
    [|
      11.451416969299316;
      9.530213356018066;
      3.4179677963256836;
      0.34277141094207764;
    |]
    (List.hd (LL.svd ~algorithm:C.Svd_default ~compute_uv:false m));
  let mc =
    Concrete
      (Ndarray.of_complex Dtype.Complex64 [| 5; 4 |]
         (Array.map
            (fun (re, im) -> { Complex.re; im })
            [|
              (1.0, 0.0);
              (1.0806045532226562, 1.6829419136047363);
              (-0.2080734223127365, 0.4546487033367157);
              (-2.969977378845215, 0.4233599901199341);
              (3.0, 0.0);
              (2.1612091064453125, 3.3658838272094727);
              (-0.6242202520370483, 1.3639460802078247);
              (0.9899924993515015, -0.14112000167369843);
              (5.0, 0.0);
              (3.782115936279297, 5.890296936035156);
              (-1.0403671264648438, 2.2732434272766113);
              (-1.979984998703003, 0.28224000334739685);
              (0.10000000149011612, 0.0);
              (-1.6209068298339844, -2.5244128704071045);
              (-3.74532151222229, 8.183676719665527);
              (-0.9899924993515015, 0.14112000167369843);
              (2.0, 0.0);
              (1.0806045532226562, 1.6829419136047363);
              (-0.832293689250946, 1.8185948133468628);
              (-1.979984998703003, 0.28224000334739685);
            |]))
  in
  same "Svd_qr complex s" Dtype.F32 [| 4 |]
    [|
      11.451416969299316;
      9.5302152633667;
      3.417966604232788;
      0.34277158975601196;
    |]
    (List.hd (LL.svd ~algorithm:C.Svd_qr ~compute_uv:false mc));
  let u, s, vt = triple (LL.svd ~algorithm:C.Svd_qr ~full_matrices:false mc) in
  ignore (u, vt);
  same "Svd_qr complex s2" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179673194885254; 0.3427715599536896;
    |]
    s;
  reconstructs "Svd_qr complex" mc u s vt;
  let mc =
    Concrete
      (Ndarray.of_complex Dtype.Complex64 [| 5; 4 |]
         (Array.map
            (fun (re, im) -> { Complex.re; im })
            [|
              (1.0, 0.0);
              (1.0806045532226562, 1.6829419136047363);
              (-0.2080734223127365, 0.4546487033367157);
              (-2.969977378845215, 0.4233599901199341);
              (3.0, 0.0);
              (2.1612091064453125, 3.3658838272094727);
              (-0.6242202520370483, 1.3639460802078247);
              (0.9899924993515015, -0.14112000167369843);
              (5.0, 0.0);
              (3.782115936279297, 5.890296936035156);
              (-1.0403671264648438, 2.2732434272766113);
              (-1.979984998703003, 0.28224000334739685);
              (0.10000000149011612, 0.0);
              (-1.6209068298339844, -2.5244128704071045);
              (-3.74532151222229, 8.183676719665527);
              (-0.9899924993515015, 0.14112000167369843);
              (2.0, 0.0);
              (1.0806045532226562, 1.6829419136047363);
              (-0.832293689250946, 1.8185948133468628);
              (-1.979984998703003, 0.28224000334739685);
            |]))
  in
  same "Svd_divide_and_conquer complex s" Dtype.F32 [| 4 |]
    [|
      11.451416969299316;
      9.5302152633667;
      3.417966604232788;
      0.34277158975601196;
    |]
    (List.hd (LL.svd ~algorithm:C.Svd_divide_and_conquer ~compute_uv:false mc));
  let u, s, vt =
    triple (LL.svd ~algorithm:C.Svd_divide_and_conquer ~full_matrices:false mc)
  in
  ignore (u, vt);
  same "Svd_divide_and_conquer complex s2" Dtype.F32 [| 4 |]
    [|
      11.451416015625; 9.530216217041016; 3.4179673194885254; 0.3427715599536896;
    |]
    s;
  reconstructs "Svd_divide_and_conquer complex" mc u s vt;
  same "svd qr vmap" Dtype.F32 [| 1; 4 |]
    [|
      11.451416969299316;
      9.530213356018066;
      3.4179677963256836;
      0.34277141094207764;
    |]
    (vmap1
       (fun v -> List.hd (LL.svd ~algorithm:C.Svd_qr ~compute_uv:false v))
       (L.expand_dims m [| 0 |]));
  let m64 = L.convert_element_type m Dtype.F64 in
  let qr64, _, _ =
    triple (LL.svd ~algorithm:C.Svd_qr ~full_matrices:false m64)
  in
  let dc64, _, _ =
    triple (LL.svd ~algorithm:C.Svd_divide_and_conquer ~full_matrices:false m64)
  in
  let qr64 = nd qr64 and dc64 = nd dc64 in
  let differs =
    Array.exists Fun.id
      (unravel [| 5; 4 |] (fun i ->
           Ndarray.get_f qr64 i <> Ndarray.get_f dc64 i))
  in
  Alcotest.(check bool) "gesvd and gesdd are distinct drivers" true differs;
  raises "svd jacobi" "not implemented on CPU" (fun () ->
      LL.svd ~algorithm:C.Svd_jacobi m);
  raises "svd polar" "not implemented on CPU" (fun () ->
      LL.svd ~algorithm:C.Svd_polar m);
  let e = arr Dtype.F32 [| 2; 2 |] [| 2.0; 1.0; 1.0; 3.0 |] in
  same_c "eig lapack w" Dtype.Complex64 [| 2 |]
    [| (1.3819659948349, 0.0); (3.6180338859558105, 0.0) |]
    (List.hd (LL.eig ~implementation:C.Lapack e));
  raises "eig cusolver" "Only the lapack" (fun () ->
      LL.eig ~implementation:C.Cusolver e);
  raises "eig magma" "Only the lapack" (fun () ->
      LL.eig ~implementation:C.Magma e);
  same "eigh qr w" Dtype.F32 [| 2 |]
    [| 1.3819661140441895; 3.6180338859558105 |]
    (snd (LL.eigh ~implementation:C.Eigh_qr e));
  same_uv "eigh qr v"
    [|
      -0.8506508469581604;
      0.5257311463356018;
      0.5257311463356018;
      0.8506508469581604;
    |]
    [|
      -0.8506508469581604;
      0.5257311463356018;
      0.5257311463356018;
      0.8506508469581604;
    |]
    [| 2; 2 |] [| 2; 2 |]
    (fst (LL.eigh ~implementation:C.Eigh_qr e))
    (L.transpose (fst (LL.eigh ~implementation:C.Eigh_qr e)) [| 1; 0 |]);
  raises "eigh jacobi" "Jacobi" (fun () ->
      LL.eigh ~implementation:C.Eigh_jacobi e);
  raises "eigh qdwh" "QDWH" (fun () -> LL.eigh ~implementation:C.Eigh_qdwh e);
  ()

let ragged_dot_general () =
  let dims lc rc lb rb ragged group =
    {
      C.dot_dimension_numbers =
        C.dot_dimension_numbers ~lhs_contract:lc ~rhs_contract:rc ~lhs_batch:lb
          ~rhs_batch:rb ();
      lhs_ragged_dimensions = ragged;
      rhs_group_dimensions = group;
    }
  in
  let l = arr Dtype.F32 [| 3; 2 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
  and r =
    arr Dtype.F32 [| 2; 2; 2 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0 |]
  and gs = arr Dtype.I32 [| 2 |] [| 1.0; 2.0 |] in
  let d1 = dims [| 1 |] [| 1 |] [||] [||] [| 0 |] [| 0 |] in
  same "ragged_dot" Dtype.F32 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    (X.ragged_dot l r gs);
  same "mode1" Dtype.F32 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    (X.ragged_dot_general l r gs d1);
  same "mode1 jit" Dtype.F32 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    (jitn
       (function
         | [ a; b; c ] -> X.ragged_dot_general a b c d1 | _ -> assert false)
       [ l; r; gs ]);
  same "pref f16" Dtype.F16 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    (X.ragged_dot ~preferred_element_type:Dtype.F16 l r gs);
  same "precision" Dtype.F32 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    (X.ragged_dot ~precision:(C.precision_of_level C.Precision_highest) l r gs);
  same "rows past groups" Dtype.F32 [| 4; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0; 0.0; 0.0 |]
    (X.ragged_dot
       (arr Dtype.F32 [| 4; 2 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0 |])
       r gs);
  same "grad lhs" Dtype.F32 [| 3; 2 |]
    [| 1.0; 5.0; 9.0; 13.0; 9.0; 13.0 |]
    (grad1 (fun a -> L.reduce_sum (X.ragged_dot a r gs) [| 0; 1 |]) l);
  same "grad rhs" Dtype.F32 [| 2; 2; 2 |]
    [| 0.0; 0.0; 1.0; 1.0; 6.0; 6.0; 8.0; 8.0 |]
    (grad1 (fun b -> L.reduce_sum (X.ragged_dot l b gs) [| 0; 1 |]) r);
  let po, to_ = jvp1 (fun a -> X.ragged_dot a r gs) l (L.full_like l 1.0) in
  same "jvp primal" Dtype.F32 [| 3; 2 |]
    [| 2.0; 3.0; 26.0; 31.0; 46.0; 55.0 |]
    po;
  same "jvp tangent" Dtype.F32 [| 3; 2 |]
    [| 2.0; 4.0; 10.0; 12.0; 10.0; 12.0 |]
    to_;
  same "mode2" Dtype.F32 [| 2; 2; 4 |]
    [|
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      3.0;
      6.0;
      9.0;
      20.0;
      23.0;
      26.0;
      29.0;
      56.0;
      65.0;
      74.0;
      83.0;
    |]
    (X.ragged_dot_general
       (arr Dtype.F32 [| 2; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |])
       (arr Dtype.F32 [| 3; 4 |]
          [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |])
       gs
       (dims [| 1 |] [| 0 |] [||] [||] [| 1 |] [||]));
  same "mode3" Dtype.F32 [| 3; 2; 2 |]
    [|
      2.0; 3.0; 6.0; 11.0; 46.0; 55.0; 66.0; 79.0; 154.0; 171.0; 190.0; 211.0;
    |]
    (X.ragged_dot_general
       (arr Dtype.F32 [| 3; 2; 2 |]
          [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |])
       (arr Dtype.F32 [| 3; 2; 2 |]
          [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |])
       gs
       (dims [| 2 |] [| 1 |] [| 0 |] [| 0 |] [| 0 |] [||]));
  let lb =
    arr Dtype.F32 [| 2; 3; 2 |]
      [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |]
  and rb =
    arr Dtype.F32 [| 2; 2; 2; 2 |]
      [|
        0.0;
        1.0;
        2.0;
        3.0;
        4.0;
        5.0;
        6.0;
        7.0;
        8.0;
        9.0;
        10.0;
        11.0;
        12.0;
        13.0;
        14.0;
        15.0;
      |]
  and d4 = dims [| 2 |] [| 2 |] [| 0 |] [| 1 |] [| 1 |] [| 0 |] in
  same "mode1 batched gs" Dtype.F32 [| 2; 3; 2 |]
    [|
      2.0; 3.0; 46.0; 51.0; 82.0; 91.0; 66.0; 79.0; 86.0; 103.0; 274.0; 295.0;
    |]
    (X.ragged_dot_general lb rb
       (arr Dtype.I32 [| 2; 2 |] [| 1.0; 2.0; 2.0; 1.0 |])
       d4);
  same "mode1 1d gs broadcast" Dtype.F32 [| 2; 3; 2 |]
    [|
      2.0; 3.0; 46.0; 51.0; 82.0; 91.0; 66.0; 79.0; 222.0; 239.0; 274.0; 295.0;
    |]
    (X.ragged_dot_general lb rb gs d4);
  same "mode2 batched" Dtype.F32 [| 2; 2; 3; 4 |]
    [|
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      4.0;
      8.0;
      12.0;
      0.0;
      8.0;
      16.0;
      24.0;
      452.0;
      477.0;
      502.0;
      527.0;
      596.0;
      629.0;
      662.0;
      695.0;
      740.0;
      781.0;
      822.0;
      863.0;
      56.0;
      62.0;
      68.0;
      74.0;
      152.0;
      170.0;
      188.0;
      206.0;
      248.0;
      278.0;
      308.0;
      338.0;
      756.0;
      785.0;
      814.0;
      843.0;
      964.0;
      1001.0;
      1038.0;
      1075.0;
      1172.0;
      1217.0;
      1262.0;
      1307.0;
    |]
    (X.ragged_dot_general
       (arr Dtype.F32 [| 2; 3; 4 |]
          [|
            0.0;
            1.0;
            2.0;
            3.0;
            4.0;
            5.0;
            6.0;
            7.0;
            8.0;
            9.0;
            10.0;
            11.0;
            12.0;
            13.0;
            14.0;
            15.0;
            16.0;
            17.0;
            18.0;
            19.0;
            20.0;
            21.0;
            22.0;
            23.0;
          |])
       (arr Dtype.F32 [| 2; 4; 4 |]
          [|
            0.0;
            1.0;
            2.0;
            3.0;
            4.0;
            5.0;
            6.0;
            7.0;
            8.0;
            9.0;
            10.0;
            11.0;
            12.0;
            13.0;
            14.0;
            15.0;
            16.0;
            17.0;
            18.0;
            19.0;
            20.0;
            21.0;
            22.0;
            23.0;
            24.0;
            25.0;
            26.0;
            27.0;
            28.0;
            29.0;
            30.0;
            31.0;
          |])
       (arr Dtype.I32 [| 2; 2 |] [| 1.0; 3.0; 2.0; 2.0 |])
       (dims [| 2 |] [| 1 |] [| 0 |] [| 0 |] [| 2 |] [||]));
  raises "bad gs shape" "group_sizes" (fun () ->
      X.ragged_dot_general lb rb
        (arr Dtype.I32 [| 3; 2 |] [| 1.0; 2.0; 2.0; 1.0; 0.0; 0.0 |])
        d4);
  raises "float gs" "np.integer" (fun () ->
      X.ragged_dot l r (arr Dtype.F32 [| 2 |] [| 1.0; 2.0 |]));
  raises "scalar gs" ">=1" (fun () ->
      X.ragged_dot l r (arr Dtype.I32 [||] [| 1.0 |]));
  raises "group offset" "group_offset" (fun () ->
      X.ragged_dot ~group_offset:gs l r gs);
  raises "two ragged" "exactly one lhs ragged" (fun () ->
      X.ragged_dot_general l r gs
        (dims [| 1 |] [| 1 |] [||] [||] [| 0; 1 |] [| 0 |]));
  raises "missing group" "exactly one rhs group" (fun () ->
      X.ragged_dot_general l r gs (dims [| 1 |] [| 1 |] [||] [||] [| 0 |] [||]));
  raises "group mismatch" "rhs group dimension size" (fun () ->
      X.ragged_dot l r (arr Dtype.I32 [| 3 |] [| 1.0; 1.0; 1.0 |]));
  ()

let scaled_dot () =
  let x =
    arr Dtype.F32 [| 2; 4 |]
      [|
        1.0010000467300415;
        2.003000020980835;
        3.006999969482422;
        4.008999824523926;
        5.011000156402588;
        6.013000011444092;
        7.019000053405762;
        8.020999908447266;
      |]
  and y =
    arr Dtype.F32 [| 4; 2 |]
      [|
        1.7000000476837158;
        0.30000001192092896;
        0.10999999940395355;
        1.2999999523162842;
        2.9000000953674316;
        2.0999999046325684;
        3.299999952316284;
        4.699999809265137;
      |]
  and ls = arr Dtype.F32 [| 2; 2 |] [| 2.0; 4.0; 0.5; 1.5 |]
  and rs = arr Dtype.F32 [| 2; 2 |] [| 1.5; 3.0; 0.25; 5.0 |] in
  same "default bf16" Dtype.Bf16 [| 2; 2 |]
    [| 23.875; 27.875; 56.0; 61.5 |]
    (X.scaled_dot x y);
  same "f32" Dtype.F32 [| 2; 2 |]
    [| 23.8291015625; 27.92578125; 55.984375; 61.5068359375 |]
    (X.scaled_dot ~preferred_element_type:Dtype.F32 x y);
  same "scales" Dtype.F32 [| 2; 2 |]
    [| 27.6748046875; 517.9921875; 24.4346923828125; 405.59033203125 |]
    (X.scaled_dot ~lhs_scale:ls ~rhs_scale:rs ~preferred_element_type:Dtype.F32
       x y);
  same "scales bf16" Dtype.Bf16 [| 2; 2 |]
    [| 27.75; 520.0; 24.5; 406.0 |]
    (X.scaled_dot ~lhs_scale:ls ~rhs_scale:rs x y);
  same "jit" Dtype.F32 [| 2; 2 |]
    [| 27.6748046875; 517.9921875; 24.4346923828125; 405.59033203125 |]
    (jitn
       (function
         | [ a; b; c; d ] ->
             X.scaled_dot ~lhs_scale:c ~rhs_scale:d
               ~preferred_element_type:Dtype.F32 a b
         | _ -> assert false)
       [ x; y; ls; rs ]);
  same "rank 3" Dtype.F32 [| 1; 2; 2 |]
    [| 23.8291015625; 27.92578125; 55.984375; 61.5068359375 |]
    (X.scaled_dot ~preferred_element_type:Dtype.F32 (L.expand_dims x [| 0 |])
       (L.expand_dims y [| 0 |]));
  same "vmap" Dtype.F32 [| 1; 2; 2 |]
    [| 23.8291015625; 27.92578125; 55.984375; 61.5068359375 |]
    (vmap1
       (fun v -> X.scaled_dot ~preferred_element_type:Dtype.F32 v y)
       (L.expand_dims x [| 0 |]));
  same "scales f16" Dtype.F16 [| 2; 2 |]
    [| 27.75; 520.0; 24.5; 406.0 |]
    (X.scaled_dot ~lhs_scale:ls ~rhs_scale:rs ~preferred_element_type:Dtype.F16
       x y);
  same "explicit dn" Dtype.F32 [| 2; 2 |]
    [| 23.8291015625; 27.92578125; 55.984375; 61.5068359375 |]
    (X.scaled_dot
       ~dimension_numbers:
         (C.dot_dimension_numbers ~lhs_contract:[| 1 |] ~rhs_contract:[| 0 |] ())
       ~preferred_element_type:Dtype.F32 x y);
  raises "rank 1" "0-rank and 1-rank" (fun () ->
      X.scaled_dot (L.index_in_dim ~axis:0 ~keepdims:false x 0) y);
  raises "not divisible" "must be divisible" (fun () ->
      X.scaled_dot ~lhs_scale:(L.full [| 2; 3 |] 1.0 Dtype.F32) x y);
  raises "ratio" "must be at least 2" (fun () ->
      X.scaled_dot ~lhs_scale:(L.full [| 2; 4 |] 1.0 Dtype.F32) x y);
  raises "non contract" "does not match scale dim" (fun () ->
      X.scaled_dot ~lhs_scale:(L.full [| 3; 2 |] 1.0 Dtype.F32) x y);
  ()

let nn_scaled () =
  same "doc example" Dtype.F32 [| 1; 1; 1 |] [| 8.0 |]
    (NN.scaled_matmul
       (arr Dtype.F32 [| 1; 1; 3 |] [| 1.0; 2.0; 3.0 |])
       (arr Dtype.F32 [| 1; 1; 3 |] [| 4.0; 5.0; 6.0 |])
       (arr Dtype.F32 [| 1; 1; 1 |] [| 0.5 |])
       (arr Dtype.F32 [| 1; 1; 1 |] [| 0.5 |]));
  let a =
    arr Dtype.F32 [| 2; 3; 4 |]
      [|
        0.0;
        0.3700000047683716;
        0.7400000095367432;
        1.1100000143051147;
        1.4800000190734863;
        1.850000023841858;
        2.2200000286102295;
        2.5900001525878906;
        2.9600000381469727;
        3.3299999237060547;
        3.700000047683716;
        4.070000171661377;
        4.440000057220459;
        4.809999942779541;
        5.180000305175781;
        5.550000190734863;
        5.920000076293945;
        6.289999961853027;
        6.659999847412109;
        7.03000020980835;
        7.400000095367432;
        7.769999980926514;
        8.140000343322754;
        8.510000228881836;
      |]
  and b =
    arr Dtype.F32 [| 2; 2; 4 |]
      [|
        0.0;
        1.1299999952316284;
        2.259999990463257;
        3.3899998664855957;
        4.519999980926514;
        5.650000095367432;
        6.779999732971191;
        7.909999847412109;
        9.039999961853027;
        10.170000076293945;
        11.300000190734863;
        12.430000305175781;
        13.559999465942383;
        14.6899995803833;
        15.819999694824219;
        16.950000762939453;
      |]
  and sa =
    arr Dtype.F32 [| 2; 3; 2 |]
      [| 0.25; 0.5; 0.75; 1.0; 1.25; 1.5; 1.75; 2.0; 2.25; 2.5; 2.75; 3.0 |]
  and sb =
    arr Dtype.F32 [| 2; 2; 2 |] [| 1.0; 1.5; 2.0; 2.5; 3.0; 3.5; 4.0; 4.5 |]
  in
  same "blocks" Dtype.F32 [| 2; 3; 2 |]
    [|
      4.180999755859375;
      18.291873931884766;
      22.263824462890625;
      114.55940246582031;
      54.56205368041992;
      295.28314208984375;
      1360.183837890625;
      2500.238037109375;
      2216.138916015625;
      4079.92431640625;
      3280.30810546875;
      6045.09912109375;
    |]
    (NN.scaled_matmul a b sa sb);
  same "blocks f16" Dtype.F16 [| 2; 3; 2 |]
    [|
      4.1796875;
      18.296875;
      22.265625;
      114.5625;
      54.5625;
      295.25;
      1360.0;
      2500.0;
      2216.0;
      4080.0;
      3280.0;
      6044.0;
    |]
    (NN.scaled_matmul ~preferred_element_type:Dtype.F16 a b sa sb);
  same "blocks jit" Dtype.F32 [| 2; 3; 2 |]
    [|
      4.180999755859375;
      18.291873931884766;
      22.263824462890625;
      114.55940246582031;
      54.56205368041992;
      295.28314208984375;
      1360.183837890625;
      2500.238037109375;
      2216.138916015625;
      4079.92431640625;
      3280.30810546875;
      6045.09912109375;
    |]
    (jitn
       (function
         | [ p; q; r; s ] -> NN.scaled_matmul p q r s | _ -> assert false)
       [ a; b; sa; sb ]);
  raises "rank" "3-dimensional" (fun () ->
      NN.scaled_matmul (L.index_in_dim ~axis:0 ~keepdims:false a 0) b sa sb);
  raises "contract" "matching batch" (fun () ->
      NN.scaled_matmul a (L.full [| 2; 2; 3 |] 1.0 Dtype.F32) sa sb);
  raises "scale contract" "scales to have matching" (fun () ->
      NN.scaled_matmul a b sa (L.full [| 2; 2; 1 |] 1.0 Dtype.F32));
  raises "scale rows" "non-contract" (fun () ->
      NN.scaled_matmul a b (L.full [| 2; 2; 2 |] 1.0 Dtype.F32) sb);
  let nv = NN.get_scaled_dot_general_config "nvfp4" in
  let mx = NN.get_scaled_dot_general_config "mxfp8" in
  let nv2 =
    NN.get_scaled_dot_general_config
      ~global_scale:(arr Dtype.F32 [| 1 |] [| 0.5 |])
      "nvfp4"
  in
  Alcotest.(check int) "nv block" 16 nv.NN.block_size;
  Alcotest.(check int) "mx block" 32 mx.NN.block_size;
  Alcotest.(check string) "nv mode" "nvfp4" nv.NN.mode;
  Alcotest.(check string) "mx mode" "mxfp8" mx.NN.mode;
  Alcotest.(check bool)
    "nv types" true
    (nv.NN.data_type = NN.Float4_e2m1fn && nv.NN.scale_type = NN.Float8_e4m3fn);
  Alcotest.(check bool)
    "mx types" true
    (mx.NN.data_type = NN.Float8_e4m3fn && mx.NN.scale_type = NN.Float8_e8m0fnu);
  Alcotest.(check bool) "mx scale" true (mx.NN.global_scale = None);
  Alcotest.(check bool) "infer only" false (nv.NN.infer_only || mx.NN.infer_only);
  same "nv scale" Dtype.F32 [| 1 |] [| 1.0 |] (Option.get nv.NN.global_scale);
  same "nv2 scale" Dtype.F32 [| 1 |] [| 0.5 |] (Option.get nv2.NN.global_scale);
  raises "bad mode" "Unsupported mode" (fun () ->
      NN.get_scaled_dot_general_config "x");
  let lhs =
    arr Dtype.F32 [| 1; 2; 8 |]
      [|
        0.0;
        0.30000001192092896;
        0.6000000238418579;
        0.9000000357627869;
        1.2000000476837158;
        1.5;
        1.8000000715255737;
        2.1000001430511475;
        2.4000000953674316;
        2.700000047683716;
        3.0;
        3.3000001907348633;
        3.6000001430511475;
        3.9000000953674316;
        4.200000286102295;
        4.5;
      |]
  in
  same "sdg default" Dtype.F32 [| 1; 2; 2 |]
    [| 12.600001335144043; 32.76000213623047; 32.76000213623047; 99.0 |]
    (NN.scaled_dot_general ~lhs_contract:[| 2 |] ~rhs_contract:[| 2 |]
       ~lhs_batch:[| 0 |] ~rhs_batch:[| 0 |] lhs lhs);
  same "sdg pref" Dtype.F16 [| 1; 2; 2 |]
    [| 12.6015625; 32.75; 32.75; 99.0 |]
    (NN.scaled_dot_general ~preferred_element_type:Dtype.F16
       ~lhs_contract:[| 2 |] ~rhs_contract:[| 2 |] ~lhs_batch:[| 0 |]
       ~rhs_batch:[| 0 |] lhs lhs);
  same "sdg int" Dtype.F32 [| 1; 2; 2 |] [| 1.0; 3.0; 3.0; 13.0 |]
    (NN.scaled_dot_general ~lhs_contract:[| 2 |] ~rhs_contract:[| 2 |]
       ~lhs_batch:[| 0 |] ~rhs_batch:[| 0 |]
       (arr Dtype.I32 [| 1; 2; 2 |] [| 0.0; 1.0; 2.0; 3.0 |])
       (arr Dtype.I32 [| 1; 2; 2 |] [| 0.0; 1.0; 2.0; 3.0 |]));
  raises "sdg configs" "scaled_matmul" (fun () ->
      NN.scaled_dot_general ~configs:[ mx; mx; mx ] ~lhs_contract:[| 2 |]
        ~rhs_contract:[| 2 |] ~lhs_batch:[| 0 |] ~rhs_batch:[| 0 |] lhs lhs);
  ()

let einsum_path_table () =
  let check_path s shapes opt expected =
    let ops = List.map (fun sh -> L.full sh 0.0 Dtype.F32) shapes in
    let path, _ = E.einsum_path ~optimize:opt s ops in
    Alcotest.(check (list (list int))) s expected path
  in
  let check_info s shapes expected =
    let ops = List.map (fun sh -> L.full sh 0.0 Dtype.F32) shapes in
    let _, info = E.einsum_path s ops in
    Alcotest.(check string) s expected (EP.path_info_to_string info)
  in
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Greedy
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Branch_1
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ij,jk,kl->il"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |] ]
    {xx|  Complete contraction:  ij,jk,kl->il
         Naive scaling:  4
     Optimized scaling:  3
      Naive FLOP count:  3.600e+2
  Optimized FLOP count:  1.280e+2
   Theoretical speedup:  2.812e+0
  Largest intermediate:  1.000e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              jk,ij->ki                             kl,ki->il
   3           GEMM              ki,kl->il                                il->il|xx};
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ij,jk,kl"
    [ [| 2; 100 |]; [| 100; 3 |]; [| 3; 4 |] ]
    {xx|  Complete contraction:  ij,jk,kl->il
         Naive scaling:  4
     Optimized scaling:  3
      Naive FLOP count:  7.200e+3
  Optimized FLOP count:  1.248e+3
   Theoretical speedup:  5.769e+0
  Largest intermediate:  8.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              jk,ij->ki                             kl,ki->il
   3           GEMM              ki,kl->il                                il->il|xx};
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Auto
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Optimal
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Greedy
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Branch_all
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Branch_2
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Branch_1
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Optimal
    [ [ 0; 2 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ea,fb,abcd,gc,hd->efgh"
    [
      [| 10; 10 |];
      [| 10; 10 |];
      [| 10; 10; 10; 10 |];
      [| 10; 10 |];
      [| 10; 10 |];
    ]
    {xx|  Complete contraction:  ea,fb,abcd,gc,hd->efgh
         Naive scaling:  8
     Optimized scaling:  5
      Naive FLOP count:  5.000e+8
  Optimized FLOP count:  8.000e+5
   Theoretical speedup:  6.250e+2
  Largest intermediate:  1.000e+4 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   5           GEMM          abcd,ea->bcde                   fb,gc,hd,bcde->efgh
   5           GEMM          bcde,fb->cdef                      gc,hd,cdef->efgh
   5           GEMM          cdef,gc->defg                         hd,defg->efgh
   5           GEMM          defg,hd->efgh                            efgh->efgh|xx};
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Auto
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Optimal
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Greedy
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Branch_all
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Branch_2
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Branch_1
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Optimal
    [ [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ab,bc,cd,de,ef,fa"
    [ [| 2; 3 |]; [| 3; 4 |]; [| 4; 5 |]; [| 5; 6 |]; [| 6; 7 |]; [| 7; 2 |] ]
    {xx|  Complete contraction:  ab,bc,cd,de,ef,fa->
         Naive scaling:  6
     Optimized scaling:  3
      Naive FLOP count:  3.024e+4
  Optimized FLOP count:  4.280e+2
   Theoretical speedup:  7.065e+1
  Largest intermediate:  1.200e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              fa,ef->ae                      ab,bc,cd,de,ae->
   3           GEMM              ae,de->ad                         ab,bc,cd,ad->
   3           GEMM              ad,cd->ac                            ab,bc,ac->
   3           GEMM              ac,bc->ab                               ab,ab->
   2            DOT                ab,ab->                                    ->|xx};
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Auto
    [ [ 1; 2 ]; [ 1; 5 ]; [ 1; 2 ]; [ 0; 2 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 5 ]; [ 0; 4 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Greedy
    [ [ 5; 6 ]; [ 4; 5 ]; [ 3; 4 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 5 ]; [ 0; 4 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Branch_2
    [ [ 1; 2 ]; [ 1; 5 ]; [ 1; 2 ]; [ 0; 2 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Branch_1
    [ [ 1; 2 ]; [ 1; 2 ]; [ 1; 2 ]; [ 2; 3 ]; [ 1; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 5 ]; [ 0; 4 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ab,bc,cd,de,ef,fg,gh"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
    ]
    {xx|  Complete contraction:  ab,bc,cd,de,ef,fg,gh->ah
         Naive scaling:  8
     Optimized scaling:  3
      Naive FLOP count:  2.540e+6
  Optimized FLOP count:  1.524e+3
   Theoretical speedup:  1.667e+3
  Largest intermediate:  4.800e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              cd,bc->db                 ab,de,ef,fg,gh,db->ah
   3           GEMM              db,de->be                    ab,ef,fg,gh,be->ah
   3           GEMM              fg,ef->ge                       ab,gh,be,ge->ah
   3           GEMM              be,ab->ea                          gh,ge,ea->ah
   3           GEMM              ea,ge->ag                             gh,ag->ah
   3           GEMM              ag,gh->ah                                ah->ah|xx};
  check_path "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    EP.Auto
    [
      [ 1; 2 ];
      [ 4; 5 ];
      [ 3; 6 ];
      [ 2; 5 ];
      [ 1; 4 ];
      [ 2; 3 ];
      [ 0; 2 ];
      [ 0; 1 ];
    ];
  check_path "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    EP.Greedy
    [
      [ 6; 7 ];
      [ 5; 7 ];
      [ 4; 6 ];
      [ 3; 5 ];
      [ 2; 4 ];
      [ 1; 3 ];
      [ 0; 2 ];
      [ 0; 1 ];
    ];
  check_path "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    EP.Branch_2
    [
      [ 0; 1 ];
      [ 4; 5 ];
      [ 3; 6 ];
      [ 2; 5 ];
      [ 1; 4 ];
      [ 0; 3 ];
      [ 1; 2 ];
      [ 0; 1 ];
    ];
  check_path "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    EP.Branch_1
    [
      [ 1; 2 ];
      [ 4; 5 ];
      [ 3; 6 ];
      [ 2; 5 ];
      [ 1; 4 ];
      [ 2; 3 ];
      [ 0; 2 ];
      [ 0; 1 ];
    ];
  check_path "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    EP.Unoptimized
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_info "ab,bc,cd,de,ef,fg,gh,hi,ij"
    [
      [| 2; 3 |];
      [| 3; 4 |];
      [| 4; 5 |];
      [| 5; 6 |];
      [| 6; 7 |];
      [| 7; 8 |];
      [| 8; 9 |];
      [| 9; 2 |];
      [| 2; 3 |];
    ]
    {xx|  Complete contraction:  ab,bc,cd,de,ef,fg,gh,hi,ij->aj
         Naive scaling:  10
     Optimized scaling:  3
      Naive FLOP count:  1.960e+7
  Optimized FLOP count:  1.028e+3
   Theoretical speedup:  1.906e+4
  Largest intermediate:  1.600e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              cd,bc->db           ab,de,ef,fg,gh,hi,ij,db->aj
   3           GEMM              hi,gh->ig              ab,de,ef,fg,ij,db,ig->aj
   3           GEMM              ig,fg->if                 ab,de,ef,ij,db,if->aj
   3           GEMM              if,ef->ie                    ab,de,ij,db,ie->aj
   3           GEMM              ie,de->id                       ab,ij,db,id->aj
   3           GEMM              id,db->ib                          ab,ij,ib->aj
   3           GEMM              ib,ab->ia                             ij,ia->aj
   3           GEMM              ia,ij->aj                                aj->aj|xx};
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ij,ij,ij"
    [ [| 3; 4 |]; [| 3; 4 |]; [| 3; 4 |] ]
    {xx|  Complete contraction:  ij,ij,ij->
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  3.600e+1
  Optimized FLOP count:  3.600e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  1.200e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2              0              ij,ij->ij                               ij,ij->
   2            DOT                ij,ij->                                    ->|xx};
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ii,jj" [ [| 3; 3 |]; [| 4; 4 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "ii,jj"
    [ [| 3; 3 |]; [| 4; 4 |] ]
    {xx|  Complete contraction:  ii,jj->
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  1.200e+1
  Optimized FLOP count:  2.400e+1
   Theoretical speedup:  5.000e-1
  Largest intermediate:  1.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2              0                jj,ii->                                    ->|xx};
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ij,kl" [ [| 2; 3 |]; [| 4; 5 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "ij,kl"
    [ [| 2; 3 |]; [| 4; 5 |] ]
    {xx|  Complete contraction:  ij,kl->ijkl
         Naive scaling:  4
     Optimized scaling:  4
      Naive FLOP count:  1.200e+2
  Optimized FLOP count:  1.200e+2
   Theoretical speedup:  1.000e+0
  Largest intermediate:  1.200e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4   OUTER/EINSUM            kl,ij->ijkl                            ijkl->ijkl|xx};
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ab,cd,ef"
    [ [| 2; 2 |]; [| 3; 3 |]; [| 4; 4 |] ]
    {xx|  Complete contraction:  ab,cd,ef->abcdef
         Naive scaling:  6
     Optimized scaling:  6
      Naive FLOP count:  1.152e+3
  Optimized FLOP count:  6.120e+2
   Theoretical speedup:  1.882e+0
  Largest intermediate:  5.760e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4   OUTER/EINSUM            cd,ab->cdab                       ef,cdab->abcdef
   6   OUTER/EINSUM        cdab,ef->abcdef                        abcdef->abcdef|xx};
  check_path "...ij,...jk" [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Greedy
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Branch_1
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ] ];
  check_path "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ] ];
  check_info "...ij,...jk"
    [ [| 5; 2; 3 |]; [| 5; 3; 4 |] ]
    {xx|  Complete contraction:  cij,cjk->cik
         Naive scaling:  4
     Optimized scaling:  4
      Naive FLOP count:  2.400e+2
  Optimized FLOP count:  2.400e+2
   Theoretical speedup:  1.000e+0
  Largest intermediate:  4.000e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4              0           cjk,cij->cik                              cik->cik|xx};
  check_path "...a,...b->..." [ [| 2; 3 |]; [| 2; 4 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "...a,...b->..." [ [| 2; 3 |]; [| 2; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "...a,...b->..." [ [| 2; 3 |]; [| 2; 4 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "...a,...b->..."
    [ [| 2; 3 |]; [| 2; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ] ];
  check_path "...a,...b->..."
    [ [| 2; 3 |]; [| 2; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ] ];
  check_path "...a,...b->..."
    [ [| 2; 3 |]; [| 2; 4 |] ]
    EP.Branch_1
    [ [ 0; 1 ] ];
  check_path "...a,...b->..." [ [| 2; 3 |]; [| 2; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "...a,...b->..."
    [ [| 2; 3 |]; [| 2; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ] ];
  check_info "...a,...b->..."
    [ [| 2; 3 |]; [| 2; 4 |] ]
    {xx|  Complete contraction:  da,db->d
         Naive scaling:  3
     Optimized scaling:  3
      Naive FLOP count:  4.800e+1
  Optimized FLOP count:  4.800e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  2.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3              0               db,da->d                                  d->d|xx};
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ijk,jkl,lm,mi"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |]; [| 5; 6 |]; [| 6; 2 |] ]
    {xx|  Complete contraction:  ijk,jkl,lm,mi->
         Naive scaling:  5
     Optimized scaling:  4
      Naive FLOP count:  2.880e+3
  Optimized FLOP count:  3.800e+2
   Theoretical speedup:  7.579e+0
  Largest intermediate:  1.000e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4           GEMM            jkl,ijk->li                            lm,mi,li->
   3           GEMM              mi,lm->il                               li,il->
   2     DOT/EINSUM                il,li->                                    ->|xx};
  check_path "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    EP.Auto
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 12 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_path "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    EP.Greedy
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 12 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_path "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    EP.Branch_2
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 12 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 2 ];
      [ 0; 1 ];
      [ 0; 2 ];
      [ 0; 2 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_path "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    EP.Branch_1
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 12 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_path "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    EP.Unoptimized
    [
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
      [ 0; 1 ];
    ];
  check_info "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p"
    [
      [| 2 |];
      [| 3 |];
      [| 4 |];
      [| 5 |];
      [| 6 |];
      [| 7 |];
      [| 8 |];
      [| 9 |];
      [| 10 |];
      [| 11 |];
      [| 12 |];
      [| 13 |];
      [| 14 |];
      [| 15 |];
      [| 16 |];
      [| 17 |];
    ]
    {xx|  Complete contraction:  a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p->abcdefghijklmnop
         Naive scaling:  16
     Optimized scaling:  16
      Naive FLOP count:  5.335e+15
  Optimized FLOP count:  3.557e+14
   Theoretical speedup:  1.500e+1
  Largest intermediate:  3.557e+14 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2   OUTER/EINSUM                b,a->ba    c,d,e,f,g,h,i,j,k,l,m,n,o,p,ba->abcdefghijklmnop
   2   OUTER/EINSUM                d,c->dc    e,f,g,h,i,j,k,l,m,n,o,p,ba,dc->abcdefghijklmnop
   3   OUTER/EINSUM              ba,e->bae    f,g,h,i,j,k,l,m,n,o,p,dc,bae->abcdefghijklmnop
   2   OUTER/EINSUM                g,f->gf    h,i,j,k,l,m,n,o,p,dc,bae,gf->abcdefghijklmnop
   2   OUTER/EINSUM                i,h->ih    j,k,l,m,n,o,p,dc,bae,gf,ih->abcdefghijklmnop
   2   OUTER/EINSUM                k,j->kj    l,m,n,o,p,dc,bae,gf,ih,kj->abcdefghijklmnop
   2   OUTER/EINSUM                m,l->ml    n,o,p,dc,bae,gf,ih,kj,ml->abcdefghijklmnop
   2   OUTER/EINSUM                o,n->on    p,dc,bae,gf,ih,kj,ml,on->abcdefghijklmnop
   3   OUTER/EINSUM              dc,p->dcp    bae,gf,ih,kj,ml,on,dcp->abcdefghijklmnop
   5   OUTER/EINSUM          gf,bae->gfbae    ih,kj,ml,on,dcp,gfbae->abcdefghijklmnop
   4   OUTER/EINSUM            kj,ih->kjih    ml,on,dcp,gfbae,kjih->abcdefghijklmnop
   4   OUTER/EINSUM            on,ml->onml    dcp,gfbae,kjih,onml->abcdefghijklmnop
   8   OUTER/EINSUM    gfbae,dcp->gfbaedcp    kjih,onml,gfbaedcp->abcdefghijklmnop
   8   OUTER/EINSUM    onml,kjih->onmlkjih    gfbaedcp,onmlkjih->abcdefghijklmnop
  16   OUTER/EINSUM onmlkjih,gfbaedcp->abcdefghijklmnop    abcdefghijklmnop->abcdefghijklmnop|xx};
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,ba" [ [| 3; 4 |]; [| 4; 3 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "ab,ba"
    [ [| 3; 4 |]; [| 4; 3 |] ]
    {xx|  Complete contraction:  ab,ba->
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  2.400e+1
  Optimized FLOP count:  2.400e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  1.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2     DOT/EINSUM                ba,ab->                                    ->|xx};
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,cb" [ [| 2; 3; 4 |]; [| 4; 3 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "abc,cb"
    [ [| 2; 3; 4 |]; [| 4; 3 |] ]
    {xx|  Complete contraction:  abc,cb->a
         Naive scaling:  3
     Optimized scaling:  3
      Naive FLOP count:  4.800e+1
  Optimized FLOP count:  4.800e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  2.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3    GEMV/EINSUM              cb,abc->a                                  a->a|xx};
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "abc,bcd"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ]
    EP.Branch_all
    [ [ 0; 1 ] ];
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "abc,bcd" [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,bcd"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ]
    EP.Unoptimized
    [ [ 0; 1 ] ];
  check_info "abc,bcd"
    [ [| 2; 3; 4 |]; [| 3; 4; 5 |] ]
    {xx|  Complete contraction:  abc,bcd->ad
         Naive scaling:  4
     Optimized scaling:  4
      Naive FLOP count:  2.400e+2
  Optimized FLOP count:  2.400e+2
   Theoretical speedup:  1.000e+0
  Largest intermediate:  1.000e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4           GEMM            bcd,abc->ad                                ad->ad|xx};
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,b" [ [| 2; 3 |]; [| 3 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "ab,b"
    [ [| 2; 3 |]; [| 3 |] ]
    {xx|  Complete contraction:  ab,b->a
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  1.200e+1
  Optimized FLOP count:  1.200e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  2.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2           GEMM                b,ab->a                                  a->a|xx};
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Branch_all [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "ab,ac" [ [| 2; 3 |]; [| 2; 4 |] ] EP.Unoptimized [ [ 0; 1 ] ];
  check_info "ab,ac"
    [ [| 2; 3 |]; [| 2; 4 |] ]
    {xx|  Complete contraction:  ab,ac->bc
         Naive scaling:  3
     Optimized scaling:  3
      Naive FLOP count:  4.800e+1
  Optimized FLOP count:  4.800e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  1.200e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              ac,ab->bc                                bc->bc|xx};
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Auto [ [ 0; 1 ] ];
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Greedy [ [ 0; 1 ] ];
  check_path "abc,dbe"
    [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ]
    EP.Branch_all
    [ [ 0; 1 ] ];
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Branch_2 [ [ 0; 1 ] ];
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Branch_1 [ [ 0; 1 ] ];
  check_path "abc,dbe" [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ] EP.Optimal [ [ 0; 1 ] ];
  check_path "abc,dbe"
    [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ]
    EP.Unoptimized
    [ [ 0; 1 ] ];
  check_info "abc,dbe"
    [ [| 2; 3; 4 |]; [| 5; 3; 6 |] ]
    {xx|  Complete contraction:  abc,dbe->acde
         Naive scaling:  5
     Optimized scaling:  5
      Naive FLOP count:  1.440e+3
  Optimized FLOP count:  1.440e+3
   Theoretical speedup:  1.000e+0
  Largest intermediate:  2.400e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   5           TDOT          dbe,abc->acde                            acde->acde|xx};
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Branch_1
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_path "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "a,ab,abc->abc"
    [ [| 2 |]; [| 2; 3 |]; [| 2; 3; 4 |] ]
    {xx|  Complete contraction:  a,ab,abc->abc
         Naive scaling:  3
     Optimized scaling:  3
      Naive FLOP count:  7.200e+1
  Optimized FLOP count:  3.000e+1
   Theoretical speedup:  2.400e+0
  Largest intermediate:  2.400e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2              0               ab,a->ab                           abc,ab->abc
   3              0            ab,abc->abc                              abc->abc|xx};
  check_path "ij" [ [| 3; 3 |] ] EP.Auto [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Optimal [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Greedy [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Branch_all [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Branch_2 [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Branch_1 [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Optimal [ [ 0 ] ];
  check_path "ij" [ [| 3; 3 |] ] EP.Unoptimized [ [ 0 ] ];
  check_info "ij"
    [ [| 3; 3 |] ]
    {xx|  Complete contraction:  ij->ij
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  9.000e+0
  Optimized FLOP count:  9.000e+0
   Theoretical speedup:  1.000e+0
  Largest intermediate:  9.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2              0                 ij->ij                                ij->ij|xx};
  check_path "ii" [ [| 3; 3 |] ] EP.Auto [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Optimal [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Greedy [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Branch_all [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Branch_2 [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Branch_1 [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Optimal [ [ 0 ] ];
  check_path "ii" [ [| 3; 3 |] ] EP.Unoptimized [ [ 0 ] ];
  check_info "ii"
    [ [| 3; 3 |] ]
    {xx|  Complete contraction:  ii->
         Naive scaling:  1
     Optimized scaling:  1
      Naive FLOP count:  3.000e+0
  Optimized FLOP count:  6.000e+0
   Theoretical speedup:  5.000e-1
  Largest intermediate:  1.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   1              0                   ii->                                    ->|xx};
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Auto
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Optimal
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Greedy
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Branch_all
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Branch_2
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Branch_1
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Optimal
    [ [ 1; 2 ]; [ 0; 1 ] ];
  check_path "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ] ];
  check_info "abcd,cdef,efab"
    [ [| 2; 3; 4; 5 |]; [| 4; 5; 6; 7 |]; [| 6; 7; 2; 3 |] ]
    {xx|  Complete contraction:  abcd,cdef,efab->
         Naive scaling:  6
     Optimized scaling:  6
      Naive FLOP count:  1.512e+4
  Optimized FLOP count:  1.032e+4
   Theoretical speedup:  1.465e+0
  Largest intermediate:  1.200e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   6           GEMM        efab,cdef->abcd                           abcd,abcd->
   4            DOT            abcd,abcd->                                    ->|xx};
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ij,jk,kl,lm,mi"
    [ [| 2; 3 |]; [| 3; 2 |]; [| 2; 3 |]; [| 3; 2 |]; [| 2; 2 |] ]
    {xx|  Complete contraction:  ij,jk,kl,lm,mi->
         Naive scaling:  5
     Optimized scaling:  3
      Naive FLOP count:  3.600e+2
  Optimized FLOP count:  7.200e+1
   Theoretical speedup:  5.000e+0
  Largest intermediate:  4.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              jk,ij->ki                         kl,lm,mi,ki->
   3           GEMM              lm,kl->mk                            mi,ki,mk->
   3           GEMM              ki,mi->km                               mk,km->
   2     DOT/EINSUM                km,mk->                                    ->|xx};
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Auto
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Greedy
    [ [ 0; 1 ]; [ 0; 3 ]; [ 0; 2 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Branch_all
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Branch_2
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Branch_1
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Optimal
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_path "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    EP.Unoptimized
    [ [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ]; [ 0; 1 ] ];
  check_info "ab,ab,ab,ab,ab"
    [ [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |]; [| 2; 3 |] ]
    {xx|  Complete contraction:  ab,ab,ab,ab,ab->
         Naive scaling:  2
     Optimized scaling:  2
      Naive FLOP count:  3.000e+1
  Optimized FLOP count:  3.000e+1
   Theoretical speedup:  1.000e+0
  Largest intermediate:  6.000e+0 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   2              0              ab,ab->ab                         ab,ab,ab,ab->
   2              0              ab,ab->ab                            ab,ab,ab->
   2              0              ab,ab->ab                               ab,ab->
   2            DOT                ab,ab->                                    ->|xx};
  raises "no such path" "not found" (fun () -> EP.optimize_of_string "dp");
  Alcotest.(check bool)
    "of string" true
    (EP.optimize_of_string "eager" = EP.Greedy
    && EP.optimize_of_string "BRANCH-2" = EP.Branch_2);
  let ops =
    [
      L.full [| 2; 3 |] 0.0 Dtype.F32;
      L.full [| 3; 4 |] 0.0 Dtype.F32;
      L.full [| 4; 5 |] 0.0 Dtype.F32;
    ]
  in
  let pth, inf =
    E.einsum_path ~optimize:(EP.Path [ [ 1; 2 ]; [ 0; 1 ] ]) "ij,jk,kl" ops
  in
  Alcotest.(check (list (list int))) "custom" [ [ 1; 2 ]; [ 0; 1 ] ] pth;
  Alcotest.(check string)
    "custom info"
    {xx|  Complete contraction:  ij,jk,kl->il
         Naive scaling:  4
     Optimized scaling:  3
      Naive FLOP count:  3.600e+2
  Optimized FLOP count:  1.800e+2
   Theoretical speedup:  2.000e+0
  Largest intermediate:  1.500e+1 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   3           GEMM              kl,jk->lj                             ij,lj->il
   3           GEMM              lj,ij->il                                il->il|xx}
    (EP.path_info_to_string inf);
  Alcotest.(check int) "naive" 360 inf.EP.naive_cost;
  Alcotest.(check int) "opt" 180 inf.EP.opt_cost;
  Alcotest.(check int) "largest" 15 inf.EP.largest_intermediate;
  Alcotest.(check (list int)) "scales" [ 3; 3 ] inf.EP.scale_list;
  Alcotest.(check (list int)) "sizes" [ 15; 10 ] inf.EP.size_list;
  Alcotest.(check string) "eq" "ij,jk,kl->il" inf.EP.eq;
  raises "bad label size" "does not match" (fun () ->
      E.einsum_path "ij,jk"
        [ L.full [| 2; 3 |] 0.0 Dtype.F32; L.full [| 4; 5 |] 0.0 Dtype.F32 ]);
  raises "wrong rank" "correct number" (fun () ->
      E.einsum_path "ijk,jk"
        [ L.full [| 2; 3 |] 0.0 Dtype.F32; L.full [| 3; 5 |] 0.0 Dtype.F32 ]);
  raises "wrong count" "must be equal" (fun () ->
      E.einsum_path "ij,jk" [ L.full [| 2; 3 |] 0.0 Dtype.F32 ]);
  raises "repeat out" "more than once" (fun () ->
      E.einsum_path "ij->ii" [ L.full [| 2; 3 |] 0.0 Dtype.F32 ]);
  raises "unknown out" "did not appear" (fun () ->
      E.einsum_path "ij->k" [ L.full [| 2; 3 |] 0.0 Dtype.F32 ]);
  raises "bad arrow" "only contain one" (fun () ->
      E.einsum_path "ij-k" [ L.full [| 2; 3 |] 0.0 Dtype.F32 ]);
  ()

let einsum_execution () =
  let ev =
    arr Dtype.F32 [| 2; 2 |]
      [|
        1.0010000467300415;
        2.003000020980835;
        3.006999969482422;
        4.008999824523926;
      |]
  and a =
    arr Dtype.F32 [| 2; 3 |]
      [|
        0.30000001192092896;
        1.4000000953674316;
        2.5;
        3.6000001430511475;
        4.700000286102295;
        5.800000190734863;
      |]
  and b =
    arr Dtype.F32 [| 3; 4 |]
      [|
        -1.0;
        -0.30000001192092896;
        0.3999999761581421;
        1.0999999046325684;
        1.7999999523162842;
        2.5;
        3.1999998092651367;
        3.9000000953674316;
        4.599999904632568;
        5.299999713897705;
        6.0;
        6.699999809265137;
      |]
  and c =
    arr Dtype.F32 [| 4; 5 |]
      [|
        0.8999999761581421;
        1.0299999713897705;
        1.159999966621399;
        1.2899999618530273;
        1.4199999570846558;
        1.5499999523162842;
        1.6799999475479126;
        1.809999942779541;
        1.9399999380111694;
        2.069999933242798;
        2.1999998092651367;
        2.3299999237060547;
        2.4600000381469727;
        2.5899999141693115;
        2.7199997901916504;
        2.8499999046325684;
        2.9800000190734863;
        3.1100001335144043;
        3.239999771118164;
        3.369999885559082;
      |]
  in
  same "three Auto" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.66497802734375;
      403.764404296875;
      427.8638000488281;
      451.9631652832031;
      476.0625915527344;
    |]
    (E.einsum ~optimize:EP.Auto "ij,jk,kl->il" [ a; b; c ]);
  same "three Optimal" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.66497802734375;
      403.764404296875;
      427.8638000488281;
      451.9631652832031;
      476.0625915527344;
    |]
    (E.einsum ~optimize:EP.Optimal "ij,jk,kl->il" [ a; b; c ]);
  same "three Greedy" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.6650085449219;
      403.764404296875;
      427.86383056640625;
      451.96319580078125;
      476.0625915527344;
    |]
    (E.einsum ~optimize:EP.Greedy "ij,jk,kl->il" [ a; b; c ]);
  same "three Branch_1" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.6650085449219;
      403.764404296875;
      427.86383056640625;
      451.96319580078125;
      476.0625915527344;
    |]
    (E.einsum ~optimize:EP.Branch_1 "ij,jk,kl->il" [ a; b; c ]);
  same "three Unoptimized" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.66497802734375;
      403.764404296875;
      427.8638000488281;
      451.9631652832031;
      476.0625915527344;
    |]
    (E.einsum ~optimize:EP.Unoptimized "ij,jk,kl->il" [ a; b; c ]);
  same "custom path" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.6650085449219;
      403.764404296875;
      427.86383056640625;
      451.96319580078125;
      476.0625915527344;
    |]
    (E.einsum
       ~optimize:(EP.Path [ [ 1; 2 ]; [ 0; 1 ] ])
       "ij,jk,kl->il" [ a; b; c ]);
  same "default" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.66497802734375;
      403.764404296875;
      427.8638000488281;
      451.9631652832031;
      476.0625915527344;
    |]
    (E.einsum "ij,jk,kl" [ a; b; c ]);
  same "jit" Dtype.F32 [| 2; 5 |]
    [|
      145.52999877929688;
      154.9575958251953;
      164.3852081298828;
      173.8127899169922;
      183.24038696289062;
      379.66497802734375;
      403.764404296875;
      427.8638000488281;
      451.9631652832031;
      476.0625915527344;
    |]
    (jitn (E.einsum "ij,jk,kl->il") [ a; b; c ]);
  same "ellipsis" Dtype.F32 [| 2; 4 |]
    [|
      13.719999313354492;
      16.65999984741211;
      19.600000381469727;
      22.540000915527344;
      31.540000915527344;
      41.40999984741211;
      51.279998779296875;
      61.150001525878906;
    |]
    (E.einsum "...j,jk->...k" [ a; b ]);
  same "trace" Dtype.F32 [||] [| 5.009999752044678 |] (E.einsum "ii" [ ev ]);
  same "prec highest" Dtype.F32 [| 2; 2 |]
    [|
      7.025022029876709;
      10.035030364990234;
      15.065069198608398;
      22.09510040283203;
    |]
    (E.einsum
       ~precision:(C.precision_of_level C.Precision_highest)
       "ij,jk->ik" [ ev; ev ]);
  same "pref f16" Dtype.F16 [| 2; 2 |]
    [| 7.0234375; 10.03125; 15.0625; 22.09375 |]
    (E.einsum ~preferred_element_type:Dtype.F16 "ij,jk->ik" [ ev; ev ]);
  same "preset" Dtype.F32 [| 2; 2 |]
    [| 7.03125; 10.0390625; 15.0625; 22.09375 |]
    (E.einsum
       ~precision:(C.precision_of_string "F16_F16_F16")
       "ij,jk->ik" [ ev; ev ]);
  raises "rejected preset" "not supported by dot_general" (fun () ->
      E.einsum
        ~precision:(C.precision_of_string "TF32_TF32_F32")
        "ij,jk->ik" [ ev; ev ]);
  raises "three at once" "simultaneous contraction" (fun () ->
      E.einsum ~optimize:(EP.Path [ [ 0; 1; 2 ] ]) "ij,jk,kl->il" [ a; b; c ]);
  same "grad" Dtype.F32 [| 2; 3 |]
    [|
      13.509997367858887;
      133.0699920654297;
      252.62998962402344;
      13.509997367858887;
      133.0699920654297;
      252.62998962402344;
    |]
    (grad1
       (fun v -> L.reduce_sum (E.einsum "ij,jk,kl->il" [ v; b; c ]) [| 0; 1 |])
       a);
  ()

let index_tricks () =
  let y =
    arr Dtype.I32 [| 2; 3; 4 |]
      [|
        0.0;
        1.0;
        2.0;
        3.0;
        4.0;
        5.0;
        6.0;
        7.0;
        8.0;
        9.0;
        10.0;
        11.0;
        12.0;
        13.0;
        14.0;
        15.0;
        16.0;
        17.0;
        18.0;
        19.0;
        20.0;
        21.0;
        22.0;
        23.0;
      |]
  in
  let sl = IT.slice in
  same "apply 1,::2,-1" Dtype.I32 [| 2 |] [| 15.0; 23.0 |]
    (IT.get y (IT.s_ [ IT.Int 1; sl ~step:2 (); IT.Int (-1) ]));
  same "index_exp 1,::2,-1" Dtype.I32 [| 2 |] [| 15.0; 23.0 |]
    (IT.apply y (IT.index_exp [ IT.Int 1; sl ~step:2 (); IT.Int (-1) ]));
  same "apply ...,1:3" Dtype.I32 [| 2; 3; 2 |]
    [| 1.0; 2.0; 5.0; 6.0; 9.0; 10.0; 13.0; 14.0; 17.0; 18.0; 21.0; 22.0 |]
    (IT.get y (IT.s_ [ IT.Ellipsis; sl ~start:1 ~stop:3 () ]));
  same "index_exp ...,1:3" Dtype.I32 [| 2; 3; 2 |]
    [| 1.0; 2.0; 5.0; 6.0; 9.0; 10.0; 13.0; 14.0; 17.0; 18.0; 21.0; 22.0 |]
    (IT.apply y (IT.index_exp [ IT.Ellipsis; sl ~start:1 ~stop:3 () ]));
  same "apply None,0,::-1" Dtype.I32 [| 1; 3; 4 |]
    [| 8.0; 9.0; 10.0; 11.0; 4.0; 5.0; 6.0; 7.0; 0.0; 1.0; 2.0; 3.0 |]
    (IT.get y (IT.s_ [ IT.Newaxis; IT.Int 0; sl ~step:(-1) () ]));
  same "index_exp None,0,::-1" Dtype.I32 [| 1; 3; 4 |]
    [| 8.0; 9.0; 10.0; 11.0; 4.0; 5.0; 6.0; 7.0; 0.0; 1.0; 2.0; 3.0 |]
    (IT.apply y (IT.index_exp [ IT.Newaxis; IT.Int 0; sl ~step:(-1) () ]));
  same "apply 1:" Dtype.I32 [| 1; 3; 4 |]
    [| 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0; 20.0; 21.0; 22.0; 23.0 |]
    (IT.get y (IT.s_ [ sl ~start:1 () ]));
  same "index_exp 1:" Dtype.I32 [| 1; 3; 4 |]
    [| 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0; 20.0; 21.0; 22.0; 23.0 |]
    (IT.apply y (IT.index_exp [ sl ~start:1 () ]));
  same "apply -1,-1,-1" Dtype.I32 [||] [| 23.0 |]
    (IT.get y (IT.s_ [ IT.Int (-1); IT.Int (-1); IT.Int (-1) ]));
  same "index_exp -1,-1,-1" Dtype.I32 [||] [| 23.0 |]
    (IT.apply y (IT.index_exp [ IT.Int (-1); IT.Int (-1); IT.Int (-1) ]));
  same "apply 0,5:1:-2" Dtype.I32 [| 1; 4 |] [| 8.0; 9.0; 10.0; 11.0 |]
    (IT.get y (IT.s_ [ IT.Int 0; sl ~start:5 ~stop:1 ~step:(-2) () ]));
  same "index_exp 0,5:1:-2" Dtype.I32 [| 1; 4 |] [| 8.0; 9.0; 10.0; 11.0 |]
    (IT.apply y (IT.index_exp [ IT.Int 0; sl ~start:5 ~stop:1 ~step:(-2) () ]));
  same "apply 1,...,None,2" Dtype.I32 [| 3; 1 |] [| 14.0; 18.0; 22.0 |]
    (IT.get y (IT.s_ [ IT.Int 1; IT.Ellipsis; IT.Newaxis; IT.Int 2 ]));
  same "index_exp 1,...,None,2" Dtype.I32 [| 3; 1 |] [| 14.0; 18.0; 22.0 |]
    (IT.apply y (IT.index_exp [ IT.Int 1; IT.Ellipsis; IT.Newaxis; IT.Int 2 ]));
  same "apply oob" Dtype.I32 [| 3; 4 |]
    [| 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0; 20.0; 21.0; 22.0; 23.0 |]
    (IT.get y (IT.s_ [ IT.Int 5 ]));
  same "index_exp oob" Dtype.I32 [| 3; 4 |]
    [| 12.0; 13.0; 14.0; 15.0; 16.0; 17.0; 18.0; 19.0; 20.0; 21.0; 22.0; 23.0 |]
    (IT.apply y (IT.index_exp [ IT.Int 5 ]));
  same "apply neg oob" Dtype.I32 [| 3; 4 |]
    [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |]
    (IT.get y (IT.s_ [ IT.Int (-9) ]));
  same "index_exp neg oob" Dtype.I32 [| 3; 4 |]
    [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0; 9.0; 10.0; 11.0 |]
    (IT.apply y (IT.index_exp [ IT.Int (-9) ]));
  same "apply ::-1,::-2,::3" Dtype.I32 [| 2; 2; 2 |]
    [| 20.0; 23.0; 12.0; 15.0; 8.0; 11.0; 0.0; 3.0 |]
    (IT.get y (IT.s_ [ sl ~step:(-1) (); sl ~step:(-2) (); sl ~step:3 () ]));
  same "index_exp ::-1,::-2,::3" Dtype.I32 [| 2; 2; 2 |]
    [| 20.0; 23.0; 12.0; 15.0; 8.0; 11.0; 0.0; 3.0 |]
    (IT.apply y
       (IT.index_exp [ sl ~step:(-1) (); sl ~step:(-2) (); sl ~step:3 () ]));
  same "apply -2:,-5:9" Dtype.I32 [| 2; 3; 4 |]
    [|
      0.0;
      1.0;
      2.0;
      3.0;
      4.0;
      5.0;
      6.0;
      7.0;
      8.0;
      9.0;
      10.0;
      11.0;
      12.0;
      13.0;
      14.0;
      15.0;
      16.0;
      17.0;
      18.0;
      19.0;
      20.0;
      21.0;
      22.0;
      23.0;
    |]
    (IT.get y (IT.s_ [ sl ~start:(-2) (); sl ~start:(-5) ~stop:9 () ]));
  same "index_exp -2:,-5:9" Dtype.I32 [| 2; 3; 4 |]
    [|
      0.0;
      1.0;
      2.0;
      3.0;
      4.0;
      5.0;
      6.0;
      7.0;
      8.0;
      9.0;
      10.0;
      11.0;
      12.0;
      13.0;
      14.0;
      15.0;
      16.0;
      17.0;
      18.0;
      19.0;
      20.0;
      21.0;
      22.0;
      23.0;
    |]
    (IT.apply y (IT.index_exp [ sl ~start:(-2) (); sl ~start:(-5) ~stop:9 () ]));
  same "apply empty" Dtype.I32 [| 0; 3; 4 |] [||]
    (IT.get y (IT.s_ [ sl ~start:0 ~stop:0 () ]));
  same "index_exp empty" Dtype.I32 [| 0; 3; 4 |] [||]
    (IT.apply y (IT.index_exp [ sl ~start:0 ~stop:0 () ]));
  same "apply empty neg" Dtype.I32 [| 0; 3; 4 |] [||]
    (IT.get y (IT.s_ [ sl ~start:1 ~stop:1 ~step:(-1) () ]));
  same "index_exp empty neg" Dtype.I32 [| 0; 3; 4 |] [||]
    (IT.apply y (IT.index_exp [ sl ~start:1 ~stop:1 ~step:(-1) () ]));
  same "apply all" Dtype.I32 [| 2; 3; 4 |]
    [|
      0.0;
      1.0;
      2.0;
      3.0;
      4.0;
      5.0;
      6.0;
      7.0;
      8.0;
      9.0;
      10.0;
      11.0;
      12.0;
      13.0;
      14.0;
      15.0;
      16.0;
      17.0;
      18.0;
      19.0;
      20.0;
      21.0;
      22.0;
      23.0;
    |]
    (IT.get y (IT.s_ [ IT.Ellipsis ]));
  same "index_exp all" Dtype.I32 [| 2; 3; 4 |]
    [|
      0.0;
      1.0;
      2.0;
      3.0;
      4.0;
      5.0;
      6.0;
      7.0;
      8.0;
      9.0;
      10.0;
      11.0;
      12.0;
      13.0;
      14.0;
      15.0;
      16.0;
      17.0;
      18.0;
      19.0;
      20.0;
      21.0;
      22.0;
      23.0;
    |]
    (IT.apply y (IT.index_exp [ IT.Ellipsis ]));
  same "jit" Dtype.I32 [| 2; 1; 3 |]
    [| 13.0; 14.0; 15.0; 21.0; 22.0; 23.0 |]
    (jit1
       (fun v ->
         IT.get v
           (IT.s_ [ IT.Int 1; sl ~step:2 (); IT.Newaxis; sl ~start:1 () ]))
       y);
  same "grad" Dtype.F32 [| 2; 3; 4 |]
    [|
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      0.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
      1.0;
    |]
    (grad1
       (fun v ->
         L.reduce_sum
           (IT.get v (IT.s_ [ IT.Int 1; sl ~step:(-1) () ]))
           [| 0; 1 |])
       (L.convert_element_type y Dtype.F32));
  Alcotest.(check bool)
    "s_ single" true
    (IT.s_ [ IT.Int 1 ] = IT.Item (IT.Int 1));
  Alcotest.(check bool)
    "s_ tuple" true
    (IT.s_ [ IT.Int 1; IT.Ellipsis ] = IT.Tuple [ IT.Int 1; IT.Ellipsis ]);
  Alcotest.(check bool)
    "s_ slice" true
    (IT.s_ [ sl ~start:1 ~stop:3 () ]
    = IT.Item (IT.Slice { start = Some 1; stop = Some 3; step = None }));
  Alcotest.(check bool)
    "index_exp single" true
    (IT.index_exp [ IT.Int 1 ] = [ IT.Int 1 ]);
  raises "too many" "Too many indices" (fun () ->
      IT.apply y [ IT.Int 0; IT.Int 0; IT.Int 0; IT.Int 0 ]);
  raises "two ellipsis" "single ellipsis" (fun () ->
      IT.apply y [ IT.Ellipsis; IT.Ellipsis ]);
  raises "zero step" "cannot be zero" (fun () -> IT.apply y [ sl ~step:0 () ]);
  ()

let stage () =
  let x = arr Dtype.F32 [| 2 |] [| 1.0; 2.0 |] in
  same "eager" Dtype.F32 [| 2 |] [| 1.0; 2.0 |] (L.stage x);
  same "grad" Dtype.F32 [| 2 |] [| 2.0; 2.0 |]
    (grad1
       (fun v -> L.reduce_sum (L.mul (L.stage v) (L.full_like v 2.0)) [| 0 |])
       x);
  let p, t =
    jvp1 (fun v -> L.mul (L.stage v) (L.full_like v 2.0)) x (L.full_like x 1.0)
  in
  same "jvp primal" Dtype.F32 [| 2 |] [| 2.0; 4.0 |] p;
  same "jvp tangent" Dtype.F32 [| 2 |] [| 2.0; 2.0 |] t;
  same "vmap" Dtype.F32 [| 2 |] [| 1.0; 2.0 |] (vmap1 L.stage x);
  let is_tracer = ref false in
  let out =
    jit1
      (fun v ->
        let c = L.stage (L.full [||] 3.0 Dtype.F32) in
        (match c with Tracer _ -> is_tracer := true | _ -> ());
        L.mul v (L.full_like v 1.0) |> L.add (L.broadcast c [| 2 |]))
      x
  in
  Alcotest.(check bool) "lifted in jit" true !is_tracer;
  same "jit value" Dtype.F32 [| 2 |] [| 4.0; 5.0 |] out;
  (match L.stage x with
  | Tracer _ -> Alcotest.fail "eager stage traced"
  | _ -> ());
  ()

let () =
  Alcotest.run "lax_nn_surface"
    [
      ( "surface",
        [
          Alcotest.test_case "rounding" `Quick rounding;
          Alcotest.test_case "dot_precision" `Quick dot_precision;
          Alcotest.test_case "accuracy" `Quick accuracy;
          Alcotest.test_case "gather_modes" `Quick gather_modes;
          Alcotest.test_case "scatter_modes" `Quick scatter_modes;
          Alcotest.test_case "fft_type" `Quick fft_type;
          Alcotest.test_case "random_algorithm" `Quick random_algorithm;
          Alcotest.test_case "linalg_config" `Quick linalg_config;
          Alcotest.test_case "ragged_dot_general" `Quick ragged_dot_general;
          Alcotest.test_case "scaled_dot" `Quick scaled_dot;
          Alcotest.test_case "nn_scaled" `Quick nn_scaled;
          Alcotest.test_case "einsum_path_table" `Quick einsum_path_table;
          Alcotest.test_case "einsum_execution" `Quick einsum_execution;
          Alcotest.test_case "index_tricks" `Quick index_tricks;
          Alcotest.test_case "stage" `Quick stage;
          Alcotest.test_case "ragged_all_to_all doc" `Quick
            ragged_all_to_all_doc;
          Alcotest.test_case "ragged_all_to_all rows" `Quick
            ragged_all_to_all_rows;
          Alcotest.test_case "ragged_all_to_all errors" `Quick
            ragged_all_to_all_errors;
        ] );
    ]
