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

let b1 = Core.bind1
let jit1 = Numpy.Jnp_util.jit1
let where = Numpy.Jnp_util.where
let scalar dt x = Core.splat dt [||] x
let iota dt n = b1 (Iota { dtype = dt; shape = [| n |]; dimension = 0 }) []

let column v =
  let n = (shape v).(0) in
  b1 (Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] }) [ v ]

let row v =
  let n = (shape v).(0) in
  b1 (Broadcast_in_dim { shape = [| 1; n |]; dims = [| 1 |] }) [ v ]

let default_float () = Dtypes.default_float_dtype ()

let binom n k =
  jit1 "_binom"
    (function
      | [ n; k ] ->
          let dt = dtype_of n in
          let one = scalar dt 1.0 in
          let a = b1 Lgamma [ b1 Add [ n; one ] ] in
          let b = b1 Lgamma [ b1 Add [ b1 Sub [ n; k ]; one ] ] in
          let c = b1 Lgamma [ b1 Add [ k; one ] ] in
          b1 Exp [ b1 Sub [ b1 Sub [ a; b ]; c ] ]
      | _ -> invalid_arg "_binom: arity")
    [ n; k ]

let comb big_n k =
  let dt = dtype_of big_n in
  let zero = scalar dt 0.0 and one = scalar dt 1.0 in
  let first = b1 Le [ k; big_n ] in
  let second = b1 Ge [ big_n; zero ] in
  let cond = b1 And [ first; second ] in
  let cond = b1 And [ cond; b1 Ge [ k; zero ] ] in
  let safe_n = where cond big_n zero in
  let safe_k = where cond k zero in
  let gammaln v = b1 Lgamma [ v ] in
  let a = gammaln (b1 Add [ safe_n; one ]) in
  let b = gammaln (b1 Add [ safe_k; one ]) in
  let c = gammaln (b1 Sub [ b1 Add [ safe_n; one ]; safe_k ]) in
  where cond (b1 Exp [ b1 Sub [ b1 Sub [ a; b ]; c ] ]) zero

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

let first_of a =
  b1 (Squeeze [| 0 |])
    [
      b1
        (Slice
           { start_indices = [| 0 |]; limit_indices = [| 1 |]; strides = None })
        [ a ];
    ]

let companion a =
  let a = inexact (NL.atleast_1d a) in
  require_vector "companion" a;
  let n = (shape a).(0) in
  if n < 2 then
    invalid_arg "The length of `a` along the last axis must be at least 2";
  let rest =
    b1
      (Slice
         { start_indices = [| 1 |]; limit_indices = [| n |]; strides = None })
      [ a ]
  in
  let negated = b1 Neg [ rest ] in
  let first_row = b1 Div [ negated; first_of a ] in
  let m = n - 1 in
  let out = NL.eye ~m ~k:(-1) ~dtype:(dtype_of first_row) m in
  let dims : scatter_dims =
    {
      update_window_dims = [| 0 |];
      inserted_window_dims = [| 0 |];
      scatter_dims_to_operand_dims = [| 0 |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
    }
  in
  let indices = Concrete (Ndarray.of_floats Dtype.I32 [| 1 |] [| 0.0 |]) in
  b1
    (Scatter { dimension_numbers = dims; unique_indices = true })
    [ out; indices; first_row ]

let fiedler_companion a =
  let a = inexact (NL.atleast_1d a) in
  require_vector "fiedler_companion" a;
  let n = (shape a).(0) - 1 in
  if n <= 0 then L.full [| 0; 0 |] 0.0 (dtype_of a)
  else
    let a = b1 Div [ a; first_of a ] in
    if n = 1 then
      L.reshape
        (b1 Neg
           [
             b1
               (Slice
                  {
                    start_indices = [| 1 |];
                    limit_indices = [| 2 |];
                    strides = None;
                  })
               [ a ];
           ])
        [| 1; 1 |]
    else
      let dt = dtype_of a in
      let coef k = b1 Neg [ at a k ] in
      let zero = scalar dt 0.0 and one = scalar dt 1.0 in
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
  (match scale with
  | None | Some "sqrtn" | Some "n" -> ()
  | Some other ->
      invalid_arg
        ("scale must be None, 'sqrtn', or 'n'; " ^ other ^ " is not valid."));
  let dt =
    match requested with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> (
        match default_float () with
        | Dtype.F64 -> Dtype.Complex128
        | _ -> Dtype.Complex64)
  in
  let pi = 3.141592653589793 in
  jit1 "dft"
    ~key:(Dtype.short_name dt ^ Option.value scale ~default:"none")
    (fun _ ->
      let a = iota dt n in
      let i = column a in
      let factor =
        Concrete
          (Ndarray.of_complex dt [||]
             [| { Complex.re = -0.0; im = -2.0 *. pi } |])
      in
      let scaled = b1 Mul [ factor; i ] in
      let j = row a in
      let product = b1 Mul [ scaled; j ] in
      let count = scalar dt (float_of_int n) in
      let quotient = b1 Div [ product; count ] in
      let omegas = b1 Exp [ quotient ] in
      match scale with
      | Some "sqrtn" ->
          let rdt =
            match dt with Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32
          in
          let root = b1 Sqrt [ scalar rdt (float_of_int n) ] in
          b1 Div [ omegas; b1 (Convert_element_type dt) [ root ] ]
      | Some "n" -> b1 Div [ omegas; count ]
      | _ -> omegas)
    []

let pascal ?(kind = "symmetric") n =
  (match kind with
  | "symmetric" | "lower" | "upper" -> ()
  | other ->
      invalid_arg
        ("Expected kind to be on of: ['symmetric', 'lower', 'upper']; got "
       ^ other));
  jit1 "pascal" ~key:kind
    (fun _ ->
      let a = iota Dtype.F32 n in
      let i = column a in
      let j = row a in
      let lower = binom i j in
      match kind with
      | "lower" -> lower
      | "upper" -> b1 (Transpose [| 1; 0 |]) [ lower ]
      | _ ->
          Numpy.Tensor_contractions.dot lower
            (b1 (Transpose [| 1; 0 |]) [ lower ]))
    []

let invpascal ?(kind = "symmetric") n =
  (match kind with
  | "symmetric" | "lower" | "upper" -> ()
  | other ->
      invalid_arg
        ("Expected kind to be one of: ['symmetric', 'lower', 'upper']; got "
       ^ other));
  jit1 "invpascal" ~key:kind
    (fun _ ->
      let dt = default_float () in
      let a = iota dt n in
      let i = column a in
      let j = row a in
      let difference = b1 Sub [ i; j ] in
      let signs = b1 Pow [ scalar dt (-1.0); difference ] in
      let inverse = b1 Mul [ signs; binom i j ] in
      match kind with
      | "lower" -> inverse
      | "upper" -> b1 (Transpose [| 1; 0 |]) [ inverse ]
      | _ ->
          Numpy.Tensor_contractions.dot
            (b1 (Transpose [| 1; 0 |]) [ inverse ])
            inverse)
    []

let helmert ?(full = false) n =
  if n < 1 then
    invalid_arg (Printf.sprintf "n must be a positive integer; got %d." n);
  jit1 "helmert" ~key:(string_of_bool full)
    (fun _ ->
      let idt = Dtypes.default_int_dtype () in
      let dt = default_float () in
      let i = column (iota idt (n - 1)) in
      let j = row (iota idt n) in
      let k = b1 Add [ i; scalar idt 1.0 ] in
      let k_f = b1 (Convert_element_type dt) [ k ] in
      let next = b1 Add [ k_f; scalar dt 1.0 ] in
      let denom = b1 Sqrt [ b1 Mul [ k_f; next ] ] in
      let below = b1 Le [ j; i ] in
      let inverse = b1 Div [ scalar dt 1.0; denom ] in
      let on = b1 Eq [ j; k ] in
      let diagonal = b1 Div [ b1 Neg [ k_f ]; denom ] in
      let zeros = Core.splat dt (shape denom) 0.0 in
      let inner = where on diagonal zeros in
      let h = where below inverse inner in
      if full then begin
        let root = b1 Sqrt [ scalar dt (float_of_int n) ] in
        let first = b1 Div [ scalar dt 1.0; root ] in
        let first_row =
          b1 (Broadcast_in_dim { shape = [| 1; n |]; dims = [||] }) [ first ]
        in
        b1 (Concatenate 0) [ first_row; h ]
      end
      else h)
    []

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
  jit1 "invhilbert"
    (fun _ ->
      let dt = default_float () in
      let a = iota dt n in
      let i = column a in
      let j = row a in
      let s = b1 Add [ i; j ] in
      let count = scalar dt (float_of_int n) in
      let signs = b1 Pow [ scalar dt (-1.0); s ] in
      let bumped = b1 Add [ s; scalar dt 1.0 ] in
      let first = b1 Mul [ signs; bumped ] in
      let left =
        comb
          (b1 Add [ count; i ])
          (b1 Sub [ b1 Sub [ count; j ]; scalar dt 1.0 ])
      in
      let second = b1 Mul [ first; left ] in
      let right =
        comb
          (b1 Add [ count; j ])
          (b1 Sub [ b1 Sub [ count; i ]; scalar dt 1.0 ])
      in
      let third = b1 Mul [ second; right ] in
      let last = b1 (Integer_pow 2) [ comb s i ] in
      b1 Mul [ third; last ])
    []

let sylvester_triangular_scan r s f =
  let m = (shape f).(0) and n = (shape f).(1) in
  let total = m * n in
  let int_scalar v = Numpy.Jnp_util.py_int v in
  let index_dtype = Dtypes.default_int_dtype () in
  let flat_indices =
    Numpy.Lax_numpy.arange ~dtype:index_dtype
      ~start:(float_of_int (total - 1))
      ~step:(-1.0) (-1.0)
  in
  let element v i j =
    let wrap ind axis =
      let size = (shape v).(axis) in
      L.select
        (L.lt ind (Core.splat (dtype_of ind) [||] 0.0))
        (L.add ind (Core.splat (dtype_of ind) [||] (float_of_int size)))
        ind
    in
    let i = wrap i 0 in
    let j = wrap j 1 in
    NL.reshape (L.dynamic_slice v [ i; j ] [| 1; 1 |]) [||]
  in
  let line v ~row idx =
    let zero = Core.splat (dtype_of idx) [||] 0.0 in
    let wrap ind axis =
      let size = (shape v).(axis) in
      L.select
        (L.lt ind (Core.splat (dtype_of ind) [||] 0.0))
        (L.add ind (Core.splat (dtype_of ind) [||] (float_of_int size)))
        ind
    in
    if row then
      let i = wrap idx 0 and j = wrap zero 1 in
      NL.reshape
        (L.dynamic_slice v [ i; j ] [| 1; (shape v).(1) |])
        [| (shape v).(1) |]
    else
      let i = wrap zero 0 and j = wrap idx 1 in
      NL.reshape
        (L.dynamic_slice v [ i; j ] [| (shape v).(0); 1 |])
        [| (shape v).(0) |]
  in
  let masked_sum mask v =
    Numpy.Reductions.sum
      (Numpy.Jnp_util.where_weak ~y_weak:true mask v
         (Numpy.Jnp_util.py_float 0.0))
  in
  let step = function
    | [ y_flat; idx ] ->
        let i = U.floor_divide idx (int_scalar n) in
        let j = U.remainder idx (int_scalar n) in
        let y = NL.reshape y_flat [| m; n |] in
        let rhs = element f i j in
        let k_row =
          Numpy.Lax_numpy.arange ~dtype:index_dtype (float_of_int m)
        in
        let row_mask = U.greater k_row i in
        let r_row = line r ~row:true i in
        let y_col = line y ~row:false j in
        let row_term = masked_sum row_mask (U.multiply r_row y_col) in
        let k_col =
          Numpy.Lax_numpy.arange ~dtype:index_dtype (float_of_int n)
        in
        let col_mask = U.greater k_col j in
        let y_row = line y ~row:true i in
        let s_col = line s ~row:false j in
        let col_term = masked_sum col_mask (U.multiply y_row s_col) in
        let rhs = U.subtract rhs (U.add row_term col_term) in
        let value = U.divide rhs (U.add (element r i i) (element s j j)) in
        let flat = U.add (U.multiply i (int_scalar n)) j in
        let wrapped =
          L.select
            (L.lt flat (Core.splat (dtype_of flat) [||] 0.0))
            (L.add flat (Core.splat (dtype_of flat) [||] (float_of_int total)))
            flat
        in
        let index =
          L.broadcast_in_dim
            (L.convert_element_type wrapped Dtype.I32)
            [| 1 |] [||]
        in
        [
          L.scatter y_flat index value
            {
              update_window_dims = [||];
              inserted_window_dims = [| 0 |];
              scatter_dims_to_operand_dims = [| 0 |];
              s_operand_batching_dims = [||];
              s_scatter_indices_batching_dims = [||];
            };
        ]
    | _ -> invalid_arg "solve_sylvester: arity"
  in
  match
    L.scan step [ Core.splat (dtype_of f) [| total |] 0.0 ] [ flat_indices ]
  with
  | [ final ] -> NL.reshape final [| m; n |]
  | _ -> invalid_arg "solve_sylvester: arity"

let is_complex_dt = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let sylvester_schur a =
  let complex_dt =
    if dtype_of a = Dtype.F64 then Dtype.Complex128 else Dtype.Complex64
  in
  Numpy.Jnp_util.jit "_schur"
    (function
      | [ a ] ->
          Lax.Linalg.schur
            (if is_complex_dt (dtype_of a) then a else NL.astype a complex_dt)
      | _ -> invalid_arg "schur: arity")
    [ a ]

let solve_sylvester_2d a b c tol =
  let hermitian v = NL.transpose (U.conj v) in
  let mm = Numpy.Tensor_contractions.matmul in
  match (sylvester_schur a, sylvester_schur (NL.transpose (U.conj b))) with
  | [ r; u ], [ s; v ] ->
      let f = mm (mm (hermitian u) (NL.astype c (dtype_of r))) v in
      let y = sylvester_triangular_scan r (hermitian s) f in
      let x = mm (mm u y) (hermitian v) in
      let x = if is_complex_dt (dtype_of c) then x else U.real x in
      let ea = Numpy.Linalg.eigvals a and eb = Numpy.Linalg.eigvals b in
      let sum = U.add (NL.expand_dims ea [| 1 |]) (NL.expand_dims eb [| 0 |]) in
      let close = Numpy.Reductions.any (U.less (U.abs sum) tol) in
      List.hd
        (L.cond close
           (fun _ ->
             [
               U.multiply
                 (Numpy.Array_creation.zeros_like x)
                 (Numpy.Jnp_util.py_float Float.nan);
             ])
           (fun _ -> [ x ])
           [])
  | _ -> invalid_arg "solve_sylvester: arity"

let solve_sylvester a b q =
  let dt = Dtypes.to_inexact_dtype (NL.result_type [ a; b; q ]) in
  let a = NL.astype a dt and b = NL.astype b dt and q = NL.astype q dt in
  let tol = Numpy.Jnp_util.py_float 1e-8 in
  Numpy.Jnp_util.jit1 "solve_sylvester"
    (function
      | [ a; b; q ] ->
          Numpy.Jnp_util.jit1 "_solve_sylvester_2d"
            (function
              | [ a; b; q; tol ] -> solve_sylvester_2d a b q tol
              | _ -> invalid_arg "solve_sylvester: arity")
            [ a; b; q; tol ]
      | _ -> invalid_arg "solve_sylvester: arity")
    [ a; b; q ]
