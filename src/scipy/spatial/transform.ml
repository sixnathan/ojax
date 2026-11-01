module T = Types
module C = Core
module Nd = Ndarray
module D = Dtype
module V = Numpy.Vectorize
module LN = Numpy.Lax_numpy
module U = Numpy.Ufuncs
module TC = Numpy.Tensor_contractions
module AC = Numpy.Array_creation
module IDX = Numpy.Indexing
module NLIN = Numpy.Linalg

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)

let concrete = function
  | T.Concrete nd -> nd
  | _ -> failwith "spatial.transform: forward-eval concrete only"

let dt_of v = Nd.dtype (concrete v)

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let inexact_or_default dt =
  if is_inexact dt then dt else Dtypes.default_float_dtype ()

let farr v =
  let nd = concrete v in
  let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
  let a = Array.make n 0.0 in
  ignore
    (Nd.fold
       (fun i x ->
         a.(i) <- x;
         i + 1)
       0 nd);
  a

let mk dt sh a = T.Concrete (Nd.of_floats dt sh a)
let bool_at v = (farr v).(0) <> 0.0
let iarr v = Array.map int_of_float (farr v)
let bool_val b = mk D.Bool [||] [| (if b then 1.0 else 0.0) |]
let int_scalar i = mk D.I32 [||] [| float_of_int i |]

let int_vec ints =
  mk D.I32 [| Array.length ints |] (Array.map float_of_int ints)

let pi = Float.pi
let deg2rad_f x = x *. (pi /. 180.)
let rad2deg_f x = x *. (180. /. pi)

let pymod x m =
  let r = Float.rem x m in
  if r < 0.0 then r +. m else r

let norm3 a = sqrt ((a.(0) *. a.(0)) +. (a.(1) *. a.(1)) +. (a.(2) *. a.(2)))

let norm4 a =
  sqrt
    ((a.(0) *. a.(0))
    +. (a.(1) *. a.(1))
    +. (a.(2) *. a.(2))
    +. (a.(3) *. a.(3)))

let compose_quat_arr p q =
  let c0 = (p.(1) *. q.(2)) -. (p.(2) *. q.(1)) in
  let c1 = (p.(2) *. q.(0)) -. (p.(0) *. q.(2)) in
  let c2 = (p.(0) *. q.(1)) -. (p.(1) *. q.(0)) in
  [|
    (p.(3) *. q.(0)) +. (q.(3) *. p.(0)) +. c0;
    (p.(3) *. q.(1)) +. (q.(3) *. p.(1)) +. c1;
    (p.(3) *. q.(2)) +. (q.(3) *. p.(2)) +. c2;
    (p.(3) *. q.(3)) -. (p.(0) *. q.(0)) -. (p.(1) *. q.(1)) -. (p.(2) *. q.(2));
  |]

let make_elem_quat_arr axis angle =
  let q = Array.make 4 0.0 in
  q.(3) <- cos (angle /. 2.0);
  q.(axis) <- sin (angle /. 2.0);
  q

let argmax4 a =
  let best = ref 0 in
  for i = 1 to Array.length a - 1 do
    if a.(i) > a.(!best) then best := i
  done;
  !best

module L = Lax_api

let lit q x = L.full [||] x (C.get_aval q).T.dtype
let at q k = L.index_in_dim ~keepdims:false q k
let where = LN.where_
let stack1 xs = C.bind1 (T.Stack 0) xs
let ( *: ) = U.multiply
let ( +: ) = U.add
let ( -: ) = U.subtract
let ( /: ) = U.true_divide

let head3 q =
  C.bind1
    (T.Slice
       { start_indices = [| 0 |]; limit_indices = [| 3 |]; strides = None })
    [ q ]

let vector_norm v = U.sqrt (TC.dot v v)
let normalize_quat q = q /: vector_norm q

let normalize_quaternion_core = function
  | [ q ] -> normalize_quat q
  | _ -> assert false

let as_matrix_core = function
  | [ q ] ->
      let x = at q 0 in
      let y = at q 1 in
      let z = at q 2 in
      let w = at q 3 in
      let x2 = x *: x in
      let y2 = y *: y in
      let z2 = z *: z in
      let w2 = w *: w in
      let xy = x *: y in
      let zw = z *: w in
      let xz = x *: z in
      let yw = y *: w in
      let yz = y *: z in
      let xw = x *: w in
      let two = lit q 2.0 in
      let e00 = x2 -: y2 -: z2 +: w2 in
      let e01 = two *: (xy -: zw) in
      let e02 = two *: (xz +: yw) in
      let e10 = two *: (xy +: zw) in
      let e11 = U.negative x2 +: y2 -: z2 +: w2 in
      let e12 = two *: (yz -: xw) in
      let e20 = two *: (xz -: yw) in
      let e21 = two *: (yz +: xw) in
      let e22 = U.negative x2 -: y2 +: z2 +: w2 in
      stack1
        [
          stack1 [ e00; e01; e02 ];
          stack1 [ e10; e11; e12 ];
          stack1 [ e20; e21; e22 ];
        ]
  | _ -> assert false

let as_mrp_core = function
  | [ q ] ->
      let w = at q 3 in
      let sign = where (U.less w (lit q 0.0)) (lit q (-1.0)) (lit q 1.0) in
      let denominator = lit q 1.0 +: (sign *: at q 3) in
      sign *: head3 q /: denominator
  | _ -> assert false

let as_rotvec_core = function
  | [ q; degrees ] ->
      let q = where (U.less (at q 3) (lit q 0.0)) (U.negative q) q in
      let norm = vector_norm (head3 q) in
      let angle = lit q 2.0 *: U.arctan2 norm (at q 3) in
      let angle2 = angle *: angle in
      let small =
        lit q 2.0
        +: (angle2 /: lit q 12.0)
        +: (lit q 7.0 *: angle2 *: angle2 /: lit q 2880.0)
      in
      let large = angle /: U.sin (angle /: lit q 2.0) in
      let scale = where (U.less_equal angle (lit q 1e-3)) small large in
      let scale = where degrees (U.rad2deg scale) scale in
      scale *: head3 q
  | _ -> assert false

let from_rotvec_core = function
  | [ rotvec; degrees ] ->
      let rotvec = where degrees (U.deg2rad rotvec) rotvec in
      let angle = vector_norm rotvec in
      let angle2 = angle *: angle in
      let lit x = lit rotvec x in
      let small =
        lit 0.5 -: (angle2 /: lit 48.0) +: (angle2 *: angle2 /: lit 3840.0)
      in
      let large = U.sin (angle /: lit 2.0) /: angle in
      let scale = where (U.less_equal angle (lit 1e-3)) small large in
      let scaled = scale *: rotvec in
      let last = U.cos (angle /: lit 2.0) in
      LN.hstack [ scaled; last ]
  | _ -> assert false

let dyn2 m i j =
  let row = L.dynamic_index_in_dim ~keepdims:false m i in
  L.dynamic_index_in_dim ~keepdims:false row j

let set1 v k x = L.dynamic_update_index_in_dim v x k
let int_lit x = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int x |])
let cell m i j = at (at m i) j

let from_matrix_core = function
  | [ matrix ] ->
      let lit x = lit matrix x in
      let m00 = cell matrix 0 0 in
      let m11 = cell matrix 1 1 in
      let m22 = cell matrix 2 2 in
      let trace = m00 +: m11 +: m22 in
      let decision = stack1 [ m00; m11; m22; trace ] in
      let choice = LN.astype (LN.argmax decision) D.I32 in
      let i = choice in
      let j = U.remainder (i +: int_lit 1) (int_lit 3) in
      let k = U.remainder (j +: int_lit 1) (int_lit 3) in
      let blank = C.splat (C.get_aval matrix).T.dtype [| 4 |] 0.0 in
      let d3 = at decision 3 in
      let q = set1 blank i (lit 1.0 -: d3 +: (lit 2.0 *: dyn2 matrix i i)) in
      let q = set1 q j (dyn2 matrix j i +: dyn2 matrix i j) in
      let q = set1 q k (dyn2 matrix k i +: dyn2 matrix i k) in
      let q = set1 q (int_lit 3) (dyn2 matrix k j -: dyn2 matrix j k) in
      let r = set1 blank (int_lit 0) (cell matrix 2 1 -: cell matrix 1 2) in
      let r = set1 r (int_lit 1) (cell matrix 0 2 -: cell matrix 2 0) in
      let r = set1 r (int_lit 2) (cell matrix 1 0 -: cell matrix 0 1) in
      let r = set1 r (int_lit 3) (lit 1.0 +: d3) in
      normalize_quat (where (U.not_equal choice (int_lit 3)) q r)
  | _ -> assert false

let from_mrp_core = function
  | [ mrp ] ->
      let lit x = lit mrp x in
      let squared = TC.dot mrp mrp +: lit 1.0 in
      let doubled = lit 2.0 *: head3 mrp in
      let last = lit 2.0 -: squared in
      LN.hstack [ doubled; last ] /: squared
  | _ -> assert false

let inv_core = function
  | [ q ] ->
      let dt = (C.get_aval q).T.dtype in
      q *: mk dt [| 4 |] [| -1.0; -1.0; -1.0; 1.0 |]
  | _ -> assert false

let magnitude_core = function
  | [ q ] ->
      let norm = vector_norm (head3 q) in
      lit q 2.0 *: U.arctan2 norm (U.absolute (at q 3))
  | _ -> assert false

let make_canonical_core = function
  | [ q ] ->
      let zero = lit q 0.0 in
      let is_neg = U.less q zero in
      let is_zero = U.equal q zero in
      let n k = at is_neg k and z k = at is_zero k in
      let ( &: ) = U.bitwise_and and ( |: ) = U.bitwise_or in
      let neg =
        n 3 |: (z 3 &: n 0) |: (z 3 &: z 0 &: n 1) |: (z 3 &: z 0 &: z 1 &: n 2)
      in
      where neg (U.negative q) q
  | _ -> assert false

let compose_quat p q =
  let cross = LN.cross (head3 p) (head3 q) in
  let p3 = at p 3 and q3 = at q 3 in
  let part k = (p3 *: at q k) +: (q3 *: at p k) +: at cross k in
  let e0 = part 0 in
  let e1 = part 1 in
  let e2 = part 2 in
  let e3 =
    (p3 *: q3) -: (at p 0 *: at q 0) -: (at p 1 *: at q 1) -: (at p 2 *: at q 2)
  in
  stack1 [ e0; e1; e2; e3 ]

let compose_quat_core = function
  | [ p; q ] -> compose_quat p q
  | _ -> assert false

let make_elementary_quat axis angle =
  let half = angle /: lit angle 2.0 in
  let zeros = C.splat (C.get_aval angle).T.dtype [| 4 |] 0.0 in
  let quat = set1 zeros (int_lit 3) (U.cos half) in
  let half = angle /: lit angle 2.0 in
  set1 quat axis (U.sin half)

let elementary_quat_compose_core = function
  | [ angles; axes; intrinsic; degrees ] ->
      let angles = where degrees (U.deg2rad angles) angles in
      let n = (shape axes).(0) in
      let elem k = make_elementary_quat (at axes k) (at angles k) in
      let result = ref (elem 0) in
      for k = 1 to n - 1 do
        let quat = elem k in
        let forward = compose_quat !result quat in
        let backward = compose_quat quat !result in
        result := where intrinsic forward backward
      done;
      !result
  | _ -> assert false

let compute_euler_from_quat_core = function
  | [ quat; axes; extrinsic; degrees ] ->
      let dt = (C.get_aval quat).T.dtype in
      let lit x = L.full [||] x dt in
      let angle_first = where extrinsic (int_lit 0) (int_lit 2) in
      let angle_third = where extrinsic (int_lit 2) (int_lit 0) in
      let axes = where extrinsic axes (L.rev axes [| 0 |]) in
      let i = at axes 0 in
      let j = at axes 1 in
      let k = at axes 2 in
      let symmetric = U.equal i k in
      let k = where symmetric (int_lit 3 -: i -: j) k in
      let sign =
        LN.astype
          (U.floor_divide ((i -: j) *: (j -: k) *: (k -: i)) (int_lit 2))
          dt
      in
      let pick idx = L.dynamic_index_in_dim ~keepdims:false quat idx in
      let q3 = at quat 3 in
      let a = where symmetric q3 (q3 -: pick j) in
      let b = where symmetric (pick i) (pick i +: (pick k *: sign)) in
      let c = where symmetric (pick j) (pick j +: q3) in
      let d = where symmetric (pick k *: sign) ((pick k *: sign) -: pick i) in
      let angles = C.splat dt [| 3 |] 0.0 in
      let middle = lit 2.0 *: U.arctan2 (U.hypot c d) (U.hypot a b) in
      let angles = set1 angles (int_lit 1) middle in
      let eps = lit 1e-7 and pi_v = lit Float.pi in
      let a1 = at angles 1 in
      let case =
        where
          (U.less_equal (U.absolute (a1 -: pi_v)) eps)
          (int_lit 2) (int_lit 0)
      in
      let case = where (U.less_equal (U.absolute a1) eps) (int_lit 1) case in
      let half_sum = U.arctan2 b a in
      let half_diff = U.arctan2 d c in
      let sign_diff = where extrinsic (lit (-1.0)) (lit 1.0) in
      let first =
        where
          (U.equal case (int_lit 1))
          (lit 2.0 *: half_sum)
          (lit 2.0 *: half_diff *: sign_diff)
      in
      let angles = set1 angles (int_lit 0) first in
      let current idx = L.dynamic_index_in_dim ~keepdims:false angles idx in
      let is_regular = U.equal case (int_lit 0) in
      let angles =
        set1 angles angle_first
          (where is_regular (half_sum -: half_diff) (current angle_first))
      in
      let current idx = L.dynamic_index_in_dim ~keepdims:false angles idx in
      let angles =
        set1 angles angle_third
          (where is_regular (half_sum +: half_diff) (current angle_third))
      in
      let current idx = L.dynamic_index_in_dim ~keepdims:false angles idx in
      let angles =
        set1 angles angle_third
          (where symmetric (current angle_third) (current angle_third *: sign))
      in
      let a1 = at angles 1 in
      let half_pi = lit (Float.pi /. 2.0) in
      let angles =
        set1 angles (int_lit 1) (where symmetric a1 (a1 -: half_pi))
      in
      let wrapped =
        U.remainder (angles +: pi_v) (lit (2.0 *. Float.pi)) -: pi_v
      in
      where degrees (U.rad2deg wrapped) wrapped
  | _ -> assert false

let apply_core = function
  | [ matrix; vector; inverse ] ->
      let chosen = where inverse (LN.matrix_transpose matrix) matrix in
      TC.matmul chosen vector
  | _ -> assert false

let normalize_quaternion q =
  V.vectorize ~signature:"(n)->(n)" normalize_quaternion_core [ q ]

let as_matrix_v q = V.vectorize ~signature:"(m)->(n,n)" as_matrix_core [ q ]
let as_mrp_v q = V.vectorize ~signature:"(m)->(n)" as_mrp_core [ q ]

let as_rotvec_v q degrees =
  V.vectorize ~signature:"(m),()->(n)" as_rotvec_core [ q; degrees ]

let from_rotvec_v rotvec degrees =
  V.vectorize ~signature:"(m),()->(n)" from_rotvec_core [ rotvec; degrees ]

let from_matrix_v m = V.vectorize ~signature:"(m,m)->(n)" from_matrix_core [ m ]
let from_mrp_v v = V.vectorize ~signature:"(m)->(n)" from_mrp_core [ v ]
let inv_v q = V.vectorize ~signature:"(n)->(n)" inv_core [ q ]
let magnitude_v q = V.vectorize ~signature:"(n)->()" magnitude_core [ q ]

let make_canonical_v q =
  V.vectorize ~signature:"(n)->(n)" make_canonical_core [ q ]

let compose_quat_v p q =
  V.vectorize ~signature:"(n),(n)->(n)" compose_quat_core [ p; q ]

let elementary_quat_compose_v angles axes intrinsic degrees =
  V.vectorize ~signature:"(m),(m),(),()->(n)" elementary_quat_compose_core
    [ angles; axes; intrinsic; degrees ]

let compute_euler_from_quat_v quat axes extrinsic degrees =
  V.vectorize ~signature:"(m),(l),(),()->(n)" compute_euler_from_quat_core
    [ quat; axes; extrinsic; degrees ]

let apply_v matrix vector inverse =
  V.vectorize ~signature:"(m,m),(m),()->(m)" apply_core
    [ matrix; vector; inverse ]

let basis_index c =
  match c with
  | 'x' -> 0
  | 'y' -> 1
  | 'z' -> 2
  | _ ->
      invalid_arg
        (Printf.sprintf "Expected axis to be from ['x', 'y', 'z'], got %c" c)

let seq_matches_upper seq =
  String.length seq >= 1
  && String.for_all (fun c -> c = 'X' || c = 'Y' || c = 'Z') seq

let seq_matches_lower seq =
  String.length seq >= 1
  && String.for_all (fun c -> c = 'x' || c = 'y' || c = 'z') seq

let seq_axes seq =
  int_vec
    (Array.init (String.length seq) (fun i ->
         basis_index (Char.lowercase_ascii seq.[i])))

module Rotation = struct
  type t = { quat : T.value }

  let of_quat quat = { quat }
  let quat r = r.quat
  let single r = ndim r.quat = 1

  let len r =
    if single r then failwith "Single rotation has no len()."
    else (shape r.quat).(0)

  let from_quat q = { quat = normalize_quaternion q }
  let from_matrix m = { quat = from_matrix_v m }
  let from_mrp v = { quat = from_mrp_v v }

  let from_rotvec ?(degrees = false) v =
    { quat = from_rotvec_v v (bool_val degrees) }

  let identity ?(dtype = Dtypes.default_float_dtype ()) () =
    { quat = mk dtype [| 4 |] [| 0.0; 0.0; 0.0; 1.0 |] }

  let concatenate rotations =
    { quat = LN.concatenate (List.map (fun r -> r.quat) rotations) }

  let from_euler seq angles ~degrees =
    let num_axes = String.length seq in
    if num_axes < 1 || num_axes > 3 then
      invalid_arg
        (Printf.sprintf
           "Expected axis specification to be a non-empty string of upto 3 \
            characters, got %s"
           seq);
    let intrinsic = seq_matches_upper seq in
    let extrinsic = seq_matches_lower seq in
    if not (intrinsic || extrinsic) then
      invalid_arg
        (Printf.sprintf
           "Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', \
            'Z'], got %s"
           seq);
    for idx = 0 to num_axes - 2 do
      if seq.[idx] = seq.[idx + 1] then
        invalid_arg
          (Printf.sprintf "Expected consecutive axes to be different, got %s"
             seq)
    done;
    let angles = LN.atleast_1d angles in
    let axes = seq_axes seq in
    {
      quat =
        elementary_quat_compose_v angles axes (bool_val intrinsic)
          (bool_val degrees);
    }

  let as_matrix r = as_matrix_v r.quat
  let as_mrp r = as_mrp_v r.quat
  let as_rotvec ?(degrees = false) r = as_rotvec_v r.quat (bool_val degrees)
  let magnitude r = magnitude_v r.quat
  let inv r = { quat = inv_v r.quat }

  let as_quat ?(canonical = false) ?(scalar_first = false) r =
    let q = if canonical then make_canonical_v r.quat else r.quat in
    if scalar_first then LN.roll ~axis:[| -1 |] q [| 1 |] else q

  let as_euler ?(degrees = false) seq r =
    if String.length seq <> 3 then
      invalid_arg (Printf.sprintf "Expected 3 axes, got %s." seq);
    let intrinsic = seq_matches_upper seq in
    let extrinsic = seq_matches_lower seq in
    if not (intrinsic || extrinsic) then
      invalid_arg
        (Printf.sprintf
           "Expected axes from `seq` to be from ['x', 'y', 'z'] or ['X', 'Y', \
            'Z'], got %s"
           seq);
    for idx = 0 to 1 do
      if seq.[idx] = seq.[idx + 1] then
        invalid_arg
          (Printf.sprintf "Expected consecutive axes to be different, got %s"
             seq)
    done;
    let axes = seq_axes seq in
    compute_euler_from_quat_v r.quat axes (bool_val extrinsic)
      (bool_val degrees)

  let apply ?(inverse = false) r vectors =
    apply_v (as_matrix r) vectors (bool_val inverse)

  let compose a b =
    { quat = normalize_quaternion (compose_quat_v a.quat b.quat) }

  let getitem r ind = { quat = IDX.take_index ~axis:0 r.quat ind }

  let getrow r i =
    let nd = concrete r.quat in
    let sh = Nd.shape nd in
    let cols = sh.(1) in
    {
      quat =
        mk (Nd.dtype nd) [| cols |]
          (Array.init cols (fun j -> Nd.get_f nd [| i; j |]));
    }

  let mean ?weights r =
    let dt = dt_of r.quat in
    let w =
      match weights with
      | None -> AC.ones ~dtype:dt [| len r |]
      | Some ww -> LN.astype ww dt
    in
    if ndim w <> 1 then
      invalid_arg
        (Printf.sprintf "Expected `weights` to be 1 dimensional, got shape %s."
           (String.concat ","
              (Array.to_list (Array.map string_of_int (shape w)))));
    if (shape w).(0) <> len r then
      invalid_arg
        "Expected `weights` to have number of values equal to number of \
         rotations.";
    let qt = LN.transpose r.quat in
    let wq = U.multiply (LN.expand_dims w [| 0 |]) qt in
    let k = TC.matmul wq r.quat in
    let _, v = NLIN.eigh k in
    let vnd = concrete v in
    let rows = (Nd.shape vnd).(0) in
    let cols = (Nd.shape vnd).(1) in
    {
      quat =
        mk (Nd.dtype vnd) [| rows |]
          (Array.init rows (fun i -> Nd.get_f vnd [| i; cols - 1 |]));
    }
end

module Slerp = struct
  type t = {
    times : T.value;
    timedelta : T.value;
    rotations : Rotation.t;
    rotvecs : T.value;
  }

  let take_rows v lo hi =
    IDX.take_index ~axis:0 v
      (LN.arange ~start:(float_of_int lo) ~step:1.0 ~dtype:D.I32
         (float_of_int hi))

  let init times rotations =
    if Rotation.single rotations || Rotation.len rotations = 1 then
      invalid_arg "`rotations` must be a sequence of at least 2 rotations.";
    let dt = dt_of rotations.Rotation.quat in
    let times = LN.astype times dt in
    if ndim times <> 1 then
      invalid_arg "Expected times to be specified in a 1 dimensional array.";
    if (shape times).(0) <> Rotation.len rotations then
      invalid_arg
        "Expected number of rotations to be equal to number of timestamps \
         given.";
    let timedelta = LN.diff times in
    let n = Rotation.len rotations in
    let all_quat = Rotation.as_quat rotations in
    let new_rotations = Rotation.of_quat (take_rows all_quat 0 (n - 1)) in
    let rotvecs =
      Rotation.as_rotvec
        (Rotation.compose
           (Rotation.inv new_rotations)
           (Rotation.of_quat (take_rows all_quat 1 n)))
    in
    { times; timedelta; rotations = new_rotations; rotvecs }

  let apply s times =
    let compute_times = LN.astype times (dt_of s.times) in
    if ndim compute_times > 1 then
      invalid_arg "`times` must be at most 1-dimensional.";
    let single_time = ndim compute_times = 0 in
    let compute_times = LN.atleast_1d compute_times in
    let ind =
      U.maximum
        (U.subtract (LN.searchsorted s.times compute_times) (int_scalar 1))
        (int_scalar 0)
    in
    let alpha =
      U.divide
        (U.subtract compute_times (IDX.take_index s.times ind))
        (IDX.take_index s.timedelta ind)
    in
    let rv = IDX.take_index ~axis:0 s.rotvecs ind in
    let scaled = U.multiply rv (LN.expand_dims alpha [| 1 |]) in
    let result =
      Rotation.compose
        (Rotation.getitem s.rotations ind)
        (Rotation.from_rotvec scaled)
    in
    if single_time then Rotation.getrow result 0 else result
end
