open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module U = Numpy.Ufuncs

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let at v i = L.index_in_dim ~keepdims:false v i

let matrix_of n m f =
  let rows = List.init n (fun i -> L.stack (List.init m (fun j -> f i j)) 0) in
  L.stack rows 0

let inexact x =
  match dtype_of x with
  | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> x
  | _ -> NL.astype x (Dtypes.default_float_dtype ())

let require_vector name v =
  if Array.length (shape v) <> 1 then
    invalid_arg (name ^ ": only one dimensional input is supported")

let float_matrix n m f =
  Concrete
    (Ndarray.of_floats
       (Dtypes.default_float_dtype ())
       [| n; m |]
       (Array.init (n * m) (fun k -> f (k / m) (k mod m))))

let int_matrix n m f =
  Concrete
    (Ndarray.of_floats
       (Dtypes.default_int_dtype ())
       [| n; m |]
       (Array.init (n * m) (fun k -> f (k / m) (k mod m))))

let hilbert n = float_matrix n n (fun i j -> 1.0 /. float_of_int (i + j + 1))

let circulant c =
  let c = NL.atleast_1d c in
  require_vector "circulant" c;
  let n = (shape c).(0) in
  matrix_of n n (fun i j -> at c ((((i - j) mod n) + n) mod n))

let hankel ?r c =
  require_vector "hankel" c;
  let n = (shape c).(0) in
  let r = match r with Some v -> v | None -> L.full_like c 0.0 in
  let m = (shape r).(0) in
  matrix_of n m (fun i j ->
      if i + j < n then at c (i + j) else at r (i + j - n + 1))

let fiedler a =
  let a = NL.atleast_1d a in
  require_vector "fiedler" a;
  let n = (shape a).(0) in
  matrix_of n n (fun i j -> L.abs (L.sub (at a i) (at a j)))

let companion a =
  let a = inexact (NL.atleast_1d a) in
  require_vector "companion" a;
  let n = (shape a).(0) in
  if n < 2 then
    invalid_arg "The length of `a` along the last axis must be at least 2";
  let zero = L.full [||] 0.0 (dtype_of a) in
  let one = L.full [||] 1.0 (dtype_of a) in
  matrix_of (n - 1) (n - 1) (fun i j ->
      if i = 0 then L.neg (L.div (at a (j + 1)) (at a 0))
      else if i = j + 1 then one
      else zero)

let fiedler_companion a =
  let a = inexact (NL.atleast_1d a) in
  require_vector "fiedler_companion" a;
  let n = (shape a).(0) - 1 in
  if n <= 0 then L.full [| 0; 0 |] 0.0 (dtype_of a)
  else
    let coef k = L.neg (L.div (at a k) (at a 0)) in
    let zero = L.full [||] 0.0 (dtype_of a) in
    let one = L.full [||] 1.0 (dtype_of a) in
    if n = 1 then L.reshape (coef 1) [| 1; 1 |]
    else
      matrix_of n n (fun i j ->
          let even = i mod 2 = 0 in
          if i = 0 && j = 0 then coef 1
          else if i = 1 && j = 0 then one
          else if j = i + 1 && even then coef (Stdlib.min (i + 2) n)
          else if j = i + 2 && even then one
          else if j = i - 1 && even && i >= 2 then coef (Stdlib.min (i + 1) n)
          else if j = i - 2 && (not even) && i >= 3 then one
          else zero)

let leslie f s =
  require_vector "leslie" f;
  require_vector "leslie" s;
  let n = (shape f).(0) in
  if n < 2 then
    invalid_arg "The length of f along the last axis must be at least 2";
  if (shape s).(0) <> n - 1 then invalid_arg "Incorrect lengths for f and s.";
  let dt = NL.result_type [ f; s ] in
  let f = NL.astype f dt and s = NL.astype s dt in
  let zero = L.full [||] 0.0 dt in
  matrix_of n n (fun i j ->
      if i = 0 then at f j else if i = j + 1 then at s j else zero)

let dft ?scale ?dtype:requested n =
  let pi = 4.0 *. Float.atan 1.0 in
  let denom =
    match scale with
    | None -> 1.0
    | Some "sqrtn" -> sqrt (float_of_int n)
    | Some "n" -> float_of_int n
    | Some other ->
        invalid_arg
          ("scale must be None, 'sqrtn', or 'n'; " ^ other ^ " is not valid.")
  in
  let dt =
    match requested with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> (
        match Dtypes.default_float_dtype () with
        | Dtype.F64 -> Dtype.Complex128
        | _ -> Dtype.Complex64)
  in
  Concrete
    (Ndarray.of_complex dt [| n; n |]
       (Array.init (n * n) (fun k ->
            let i = k / n and j = k mod n in
            let phase = i * j mod n in
            let angle = -2.0 *. pi *. float_of_int phase /. float_of_int n in
            { Complex.re = cos angle /. denom; im = sin angle /. denom })))

let binomial n k =
  let r = ref 1.0 in
  for i = 1 to k do
    r := !r *. float_of_int (n - k + i) /. float_of_int i
  done;
  Float.round !r

let pascal ?(kind = "symmetric") n =
  let entry =
    match kind with
    | "symmetric" -> fun i j -> binomial (i + j) i
    | "lower" -> fun i j -> if j <= i then binomial i j else 0.0
    | "upper" -> fun i j -> if i <= j then binomial j i else 0.0
    | other ->
        invalid_arg
          ("Expected kind to be on of: ['symmetric', 'lower', 'upper']; got "
         ^ other)
  in
  Concrete
    (Ndarray.of_floats Dtype.F32 [| n; n |]
       (Array.init (n * n) (fun k -> entry (k / n) (k mod n))))

let invpascal ?(kind = "symmetric") n =
  let sign i j = if (i + j) mod 2 = 0 then 1.0 else -1.0 in
  let lower i j = if j <= i then sign i j *. binomial i j else 0.0 in
  let entry =
    match kind with
    | "lower" -> lower
    | "upper" -> fun i j -> lower j i
    | "symmetric" ->
        fun i j ->
          let total = ref 0.0 in
          for k = 0 to n - 1 do
            total := !total +. (lower k i *. lower k j)
          done;
          !total
    | other ->
        invalid_arg
          ("Expected kind to be one of: ['symmetric', 'lower', 'upper']; got "
         ^ other)
  in
  float_matrix n n entry

let helmert ?(full = false) n =
  let entry i j =
    if i = 0 then 1.0 /. sqrt (float_of_int n)
    else
      let norm = sqrt (float_of_int (i * (i + 1))) in
      if j < i then 1.0 /. norm
      else if j = i then -.float_of_int i /. norm
      else 0.0
  in
  if full then float_matrix n n entry
  else float_matrix (n - 1) n (fun i j -> entry (i + 1) j)

let hadamard ?dtype:requested n =
  if n <= 0 || n land (n - 1) <> 0 then
    invalid_arg "n must be a positive power of 2 for this implementation";
  let popcount x =
    let rec go x acc = if x = 0 then acc else go (x lsr 1) (acc + (x land 1)) in
    go x 0
  in
  let dt =
    match requested with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> Dtypes.default_int_dtype ()
  in
  Concrete
    (Ndarray.of_floats dt [| n; n |]
       (Array.init (n * n) (fun k ->
            if popcount (k / n land (k mod n)) mod 2 = 0 then 1.0 else -1.0)))

let convolution_matrix ?(mode = "full") a n =
  if n <= 0 then invalid_arg "n must be a positive integer";
  require_vector "convolution_matrix" a;
  let m = (shape a).(0) in
  let dt = dtype_of a in
  let zero = L.full [||] 0.0 dt in
  let entry row col =
    let k = row - col in
    if k >= 0 && k < m then at a k else zero
  in
  let full_rows = m + n - 1 in
  let first, rows =
    match mode with
    | "full" -> (0, full_rows)
    | "same" ->
        let target = Stdlib.max m n in
        ((full_rows - target) / 2, target)
    | "valid" ->
        let target = Stdlib.max m n - Stdlib.min m n + 1 in
        (Stdlib.min m n - 1, target)
    | other ->
        invalid_arg ("mode must be one of 'full', 'valid', 'same'; got " ^ other)
  in
  matrix_of rows n (fun i j -> entry (first + i) j)

let invhilbert n =
  let b = binomial in
  float_matrix n n (fun i j ->
      let sign = if (i + j) mod 2 = 0 then 1.0 else -1.0 in
      sign
      *. float_of_int (i + j + 1)
      *. b (n + i) (n - j - 1)
      *. b (n + j) (n - i - 1)
      *. (b (i + j) i ** 2.0))

let solve_sylvester a b q =
  let module LN = Numpy.Linalg in
  let n = (shape a).(0) and m = (shape b).(0) in
  let eye k = NL.identity ~dtype:(NL.result_type [ a; b; q ]) k in
  let lhs = U.add (NL.kron a (eye m)) (NL.kron (eye n) (NL.transpose b)) in
  let solved = LN.solve lhs (NL.reshape q [| n * m |]) in
  NL.reshape solved [| n; m |]
