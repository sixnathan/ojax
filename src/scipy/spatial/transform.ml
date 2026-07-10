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

let normalize_quaternion_core = function
  | [ q ] ->
      let a = farr q in
      let n = norm4 a in
      mk (inexact_or_default (dt_of q)) [| 4 |] (Array.map (fun x -> x /. n) a)
  | _ -> assert false

let as_matrix_core = function
  | [ q ] ->
      let a = farr q in
      let x = a.(0) and y = a.(1) and z = a.(2) and w = a.(3) in
      let x2 = x *. x and y2 = y *. y and z2 = z *. z and w2 = w *. w in
      let xy = x *. y and zw = z *. w and xz = x *. z and yw = y *. w in
      let yz = y *. z and xw = x *. w in
      mk
        (inexact_or_default (dt_of q))
        [| 3; 3 |]
        [|
          x2 -. y2 -. z2 +. w2;
          2.0 *. (xy -. zw);
          2.0 *. (xz +. yw);
          2.0 *. (xy +. zw);
          -.x2 +. y2 -. z2 +. w2;
          2.0 *. (yz -. xw);
          2.0 *. (xz -. yw);
          2.0 *. (yz +. xw);
          -.x2 -. y2 +. z2 +. w2;
        |]
  | _ -> assert false

let as_mrp_core = function
  | [ q ] ->
      let a = farr q in
      let sign = if a.(3) < 0.0 then -1.0 else 1.0 in
      let denom = 1.0 +. (sign *. a.(3)) in
      mk
        (inexact_or_default (dt_of q))
        [| 3 |]
        [|
          sign *. a.(0) /. denom; sign *. a.(1) /. denom; sign *. a.(2) /. denom;
        |]
  | _ -> assert false

let as_rotvec_core = function
  | [ q; degrees ] ->
      let a0 = farr q in
      let a = if a0.(3) < 0.0 then Array.map (fun x -> -.x) a0 else a0 in
      let angle = 2.0 *. atan2 (norm3 a) a.(3) in
      let angle2 = angle *. angle in
      let small =
        2.0 +. (angle2 /. 12.0) +. (7.0 *. angle2 *. angle2 /. 2880.0)
      in
      let large = angle /. sin (angle /. 2.0) in
      let scale = if angle <= 1e-3 then small else large in
      let scale = if bool_at degrees then rad2deg_f scale else scale in
      mk
        (inexact_or_default (dt_of q))
        [| 3 |]
        [| scale *. a.(0); scale *. a.(1); scale *. a.(2) |]
  | _ -> assert false

let from_rotvec_core = function
  | [ rotvec; degrees ] ->
      let r0 = farr rotvec in
      let r = if bool_at degrees then Array.map deg2rad_f r0 else r0 in
      let angle = norm3 r in
      let angle2 = angle *. angle in
      let small = 0.5 -. (angle2 /. 48.0) +. (angle2 *. angle2 /. 3840.0) in
      let large = sin (angle /. 2.0) /. angle in
      let scale = if angle <= 1e-3 then small else large in
      mk
        (inexact_or_default (dt_of rotvec))
        [| 4 |]
        [| scale *. r.(0); scale *. r.(1); scale *. r.(2); cos (angle /. 2.0) |]
  | _ -> assert false

let from_matrix_core = function
  | [ matrix ] ->
      let a = farr matrix in
      let m i j = a.((3 * i) + j) in
      let trace = m 0 0 +. m 1 1 +. m 2 2 in
      let decision = [| m 0 0; m 1 1; m 2 2; trace |] in
      let choice = argmax4 decision in
      let quat =
        if choice <> 3 then (
          let i = choice in
          let j = (i + 1) mod 3 in
          let k = (j + 1) mod 3 in
          let q012 = Array.make 4 0.0 in
          q012.(i) <- 1.0 -. decision.(3) +. (2.0 *. m i i);
          q012.(j) <- m j i +. m i j;
          q012.(k) <- m k i +. m i k;
          q012.(3) <- m k j -. m j k;
          q012)
        else
          [|
            m 2 1 -. m 1 2; m 0 2 -. m 2 0; m 1 0 -. m 0 1; 1.0 +. decision.(3);
          |]
      in
      let n = norm4 quat in
      mk
        (inexact_or_default (dt_of matrix))
        [| 4 |]
        (Array.map (fun x -> x /. n) quat)
  | _ -> assert false

let from_mrp_core = function
  | [ mrp ] ->
      let a = farr mrp in
      let sq = (a.(0) *. a.(0)) +. (a.(1) *. a.(1)) +. (a.(2) *. a.(2)) in
      let sq1 = sq +. 1.0 in
      mk
        (inexact_or_default (dt_of mrp))
        [| 4 |]
        [|
          2.0 *. a.(0) /. sq1;
          2.0 *. a.(1) /. sq1;
          2.0 *. a.(2) /. sq1;
          (2.0 -. sq1) /. sq1;
        |]
  | _ -> assert false

let inv_core = function
  | [ q ] ->
      let a = farr q in
      mk
        (inexact_or_default (dt_of q))
        [| 4 |]
        [| -.a.(0); -.a.(1); -.a.(2); a.(3) |]
  | _ -> assert false

let magnitude_core = function
  | [ q ] ->
      let a = farr q in
      mk
        (inexact_or_default (dt_of q))
        [||]
        [| 2.0 *. atan2 (norm3 a) (Float.abs a.(3)) |]
  | _ -> assert false

let make_canonical_core = function
  | [ q ] ->
      let a = farr q in
      let is_neg i = a.(i) < 0.0 in
      let is_zero i = a.(i) = 0.0 in
      let neg =
        is_neg 3
        || (is_zero 3 && is_neg 0)
        || (is_zero 3 && is_zero 0 && is_neg 1)
        || (is_zero 3 && is_zero 0 && is_zero 1 && is_neg 2)
      in
      let out = if neg then Array.map (fun x -> -.x) a else a in
      mk (inexact_or_default (dt_of q)) [| 4 |] out
  | _ -> assert false

let compose_quat_core = function
  | [ p; q ] ->
      let dt = inexact_or_default (LN.result_type [ p; q ]) in
      mk dt [| 4 |] (compose_quat_arr (farr p) (farr q))
  | _ -> assert false

let elementary_quat_compose_core = function
  | [ angles; axes; intrinsic; degrees ] ->
      let intr = bool_at intrinsic in
      let deg = bool_at degrees in
      let ang0 = farr angles in
      let ang = if deg then Array.map deg2rad_f ang0 else ang0 in
      let ax = iarr axes in
      let result = ref (make_elem_quat_arr ax.(0) ang.(0)) in
      for idx = 1 to Array.length ax - 1 do
        let q = make_elem_quat_arr ax.(idx) ang.(idx) in
        result :=
          if intr then compose_quat_arr !result q
          else compose_quat_arr q !result
      done;
      mk (inexact_or_default (dt_of angles)) [| 4 |] !result
  | _ -> assert false

let compute_euler_from_quat_core = function
  | [ quat; axes; extrinsic; degrees ] ->
      let q = farr quat in
      let ext = bool_at extrinsic in
      let deg = bool_at degrees in
      let ax0 = iarr axes in
      let ax = if ext then ax0 else [| ax0.(2); ax0.(1); ax0.(0) |] in
      let angle_first = if ext then 0 else 2 in
      let angle_third = if ext then 2 else 0 in
      let i = ax.(0) in
      let j = ax.(1) in
      let k0 = ax.(2) in
      let symmetric = i = k0 in
      let k = if symmetric then 3 - i - j else k0 in
      let sign = float_of_int ((i - j) * (j - k) * (k - i) / 2) in
      let eps = 1e-7 in
      let w = q.(3) in
      let a_ = if symmetric then w else w -. q.(j) in
      let b_ = if symmetric then q.(i) else q.(i) +. (q.(k) *. sign) in
      let c_ = if symmetric then q.(j) else q.(j) +. w in
      let d_ = if symmetric then q.(k) *. sign else (q.(k) *. sign) -. q.(i) in
      let angles = Array.make 3 0.0 in
      angles.(1) <- 2.0 *. atan2 (Float.hypot c_ d_) (Float.hypot a_ b_);
      let case = if Float.abs (angles.(1) -. pi) <= eps then 2 else 0 in
      let case = if Float.abs angles.(1) <= eps then 1 else case in
      let half_sum = atan2 b_ a_ in
      let half_diff = atan2 d_ c_ in
      angles.(0) <-
        (if case = 1 then 2.0 *. half_sum
         else 2.0 *. half_diff *. if ext then -1.0 else 1.0);
      if case = 0 then angles.(angle_first) <- half_sum -. half_diff;
      if case = 0 then angles.(angle_third) <- half_sum +. half_diff;
      if not symmetric then angles.(angle_third) <- angles.(angle_third) *. sign;
      if not symmetric then angles.(1) <- angles.(1) -. (pi /. 2.0);
      for t = 0 to 2 do
        angles.(t) <- pymod (angles.(t) +. pi) (2.0 *. pi) -. pi
      done;
      let out = if deg then Array.map rad2deg_f angles else angles in
      mk (inexact_or_default (dt_of quat)) [| 3 |] out
  | _ -> assert false

let apply_core = function
  | [ matrix; vector; inverse ] ->
      let m0 = farr matrix in
      let mt =
        if bool_at inverse then
          [|
            m0.(0);
            m0.(3);
            m0.(6);
            m0.(1);
            m0.(4);
            m0.(7);
            m0.(2);
            m0.(5);
            m0.(8);
          |]
        else m0
      in
      let v = farr vector in
      let out =
        Array.init 3 (fun r ->
            (mt.((3 * r) + 0) *. v.(0))
            +. (mt.((3 * r) + 1) *. v.(1))
            +. (mt.((3 * r) + 2) *. v.(2)))
      in
      mk (inexact_or_default (LN.result_type [ matrix; vector ])) [| 3 |] out
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

  let getitem r ind = { quat = IDX.take ~axis:0 r.quat ind }

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
    IDX.take ~axis:0 v
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
        (U.subtract compute_times (IDX.take s.times ind))
        (IDX.take s.timedelta ind)
    in
    let rv = IDX.take ~axis:0 s.rotvecs ind in
    let scaled = U.multiply rv (LN.expand_dims alpha [| 1 |]) in
    let result =
      Rotation.compose
        (Rotation.getitem s.rotations ind)
        (Rotation.from_rotvec scaled)
    in
    if single_time then Rotation.getrow result 0 else result
end
