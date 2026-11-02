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
  let weak_zero = Core.py_scalar dt 0.0 in
  let where0 x = Numpy.Jnp_util.where_weak ~y_weak:true cond x weak_zero in
  let safe_n = where0 big_n in
  let safe_k = where0 k in
  let gammaln v = b1 Lgamma [ v ] in
  let a = gammaln (b1 Add [ safe_n; one ]) in
  let b = gammaln (b1 Add [ safe_k; one ]) in
  let ab = b1 Sub [ a; b ] in
  let c = gammaln (b1 Sub [ b1 Add [ safe_n; one ]; safe_k ]) in
  where0 (b1 Exp [ b1 Sub [ ab; c ] ])

let hilbert n =
  jit1 "hilbert"
    (fun _ ->
      let dt = default_float () in
      let a = b1 (Iota { dtype = dt; shape = [| n; 1 |]; dimension = 0 }) [] in
      let t = b1 (Transpose [| 1; 0 |]) [ a ] in
      let s = b1 Add [ b1 Add [ a; t ]; Core.py_scalar dt 1.0 ] in
      b1 Div [ Core.py_scalar dt 1.0; s ])
    []

let weak_int v = Core.py_scalar (Dtypes.default_int_dtype ()) (float_of_int v)
let int_iota n = iota (Dtypes.default_int_dtype ()) n

let circulant c =
  let c = NL.atleast_1d c in
  require_vector "circulant" c;
  let n = (shape c).(0) in
  let idx =
    U.remainder (b1 Sub [ column (int_iota n); row (int_iota n) ]) (weak_int n)
  in
  NL.take_trailing c [ idx ]

let hankel_patches c r =
  let nc = (shape c).(0) and nr = (shape r).(0) in
  let tail =
    b1
      (Slice
         { start_indices = [| 1 |]; limit_indices = [| nr |]; strides = None })
      [ r ]
  in
  let v = NL.concatenate [ c; tail ] in
  let lifted = L.reshape v [| 1; nc + nr - 1; 1 |] in
  let patches =
    L.conv_general_dilated_patches ~window_strides:[| 1 |] ~padding:L.Valid
      ~dimension_numbers:
        {
          lhs_spec = [| 0; 2; 1 |];
          rhs_spec = [| 1; 0; 2 |];
          out_spec = [| 0; 2; 1 |];
        }
      ~precision:("HIGHEST", "DEFAULT") lifted ~filter_shape:[| nr |]
  in
  b1 (Squeeze [| 0 |]) [ patches ]

let hankel ?r c =
  require_vector "hankel" c;
  let r =
    match r with Some v -> v | None -> Numpy.Array_creation.zeros_like c
  in
  hankel_patches c r

let toeplitz ?r c =
  let r = match r with Some r -> r | None -> U.conj c in
  let c = NL.atleast_1d c and r = NL.atleast_1d r in
  let m = (shape c).(0) and n = (shape r).(0) in
  if m = 0 || n = 0 then
    L.full [| m; n |] 0.0 (Dtypes.promote_types (dtype_of c) (dtype_of r))
  else
    let reversed = b1 (Rev [| 0 |]) [ c ] in
    NL.flip ~axis:[| 0 |] (hankel_patches reversed r)

let fiedler a =
  let a = NL.atleast_1d a in
  require_vector "fiedler" a;
  U.abs (U.subtract (column a) (row a))

let first_of a =
  b1 (Squeeze [| 0 |])
    [
      b1
        (Slice
           { start_indices = [| 0 |]; limit_indices = [| 1 |]; strides = None })
        [ a ];
    ]

let set_first_row out row_values =
  let dims : scatter_dims =
    {
      update_window_dims = [| 0 |];
      inserted_window_dims = [| 0 |];
      scatter_dims_to_operand_dims = [| 0 |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
      s_sorted_unique = true;
      s_clip = false;
    }
  in
  let indices =
    b1
      (Broadcast_in_dim { shape = [| 1 |]; dims = [||] })
      [ Core.py_scalar Dtype.I32 0.0 ]
  in
  b1
    (Scatter { dimension_numbers = dims; unique_indices = true })
    [ out; indices; row_values ]

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
  set_first_row out first_row

let fiedler_companion a =
  let a = inexact (NL.atleast_1d a) in
  if (shape a).(Array.length (shape a) - 1) = 0 then
    invalid_arg
      (Printf.sprintf
         "fiedler_companion requires the last axis of 'a' to have nonzero \
          length, but got an array of shape %s."
         (Util.tuple_string (shape a)));
  require_vector "fiedler_companion" a;
  let n = (shape a).(0) - 1 in
  if n <= 0 then L.full [| 0; 0 |] 0.0 (dtype_of a)
  else
    let a = b1 Div [ a; first_of a ] in
    let slice_from lo hi =
      b1
        (Slice
           {
             start_indices = [| lo |];
             limit_indices = [| hi |];
             strides = None;
           })
        [ a ]
    in
    if n = 1 then b1 Neg [ L.reshape (slice_from 1 2) [| 1; 1 |] ]
    else
      let dt = dtype_of a in
      let i = column (int_iota n) in
      let j = row (int_iota n) in
      let add_int v k = U.add v (weak_int k) in
      let sub_int v k = U.subtract v (weak_int k) in
      let eq_int v k = U.equal v (weak_int k) in
      let super_val =
        U.negative (NL.take_trailing a [ U.minimum (add_int i 2) (weak_int n) ])
      in
      let sub_val =
        U.negative (NL.take_trailing a [ U.minimum (add_int i 1) (weak_int n) ])
      in
      let i_even = eq_int (U.remainder i (weak_int 2)) 0 in
      let i_odd = eq_int (U.remainder i (weak_int 2)) 1 in
      let one = scalar dt 1.0 and zero = scalar dt 0.0 in
      let c1 = U.logical_and (eq_int i 0) (eq_int j 0) in
      let c2 = U.logical_and (eq_int i 1) (eq_int j 0) in
      let c3 = U.logical_and (U.equal j (add_int i 1)) i_even in
      let c4 = U.logical_and (U.equal j (add_int i 2)) i_even in
      let c5 =
        U.logical_and
          (U.logical_and (U.equal j (sub_int i 1)) i_even)
          (U.greater_equal i (weak_int 2))
      in
      let c6 =
        U.logical_and
          (U.logical_and (U.equal j (sub_int i 2)) i_odd)
          (U.greater_equal i (weak_int 3))
      in
      let a1 = U.negative (b1 (Squeeze [| 0 |]) [ slice_from 1 2 ]) in
      NL.select ~default:zero [ c1; c2; c3; c4; c5; c6 ]
        [ a1; one; super_val; one; sub_val; one ]

let leslie f s =
  let f = NL.atleast_1d f and s = NL.atleast_1d s in
  require_vector "leslie" f;
  require_vector "leslie" s;
  let n = (shape f).(0) in
  if n < 2 then
    invalid_arg "The length of f along the last axis must be at least 2";
  if (shape s).(0) <> n - 1 then invalid_arg "Incorrect lengths for f and s.";
  let dt = NL.result_type [ f; s ] in
  let f = NL.astype f dt and s = NL.astype s dt in
  set_first_row (NL.diag ~k:(-1) s) f

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
             [| { Complex.re = 0.0; im = -2.0 *. pi } |])
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
        let first = b1 (Convert_element_type dt) [ first ] in
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
  let dt =
    match requested with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> Dtypes.default_int_dtype ()
  in
  jit1 "hadamard"
    ~key:(Printf.sprintf "%d %s" n (Dtype.short_name dt))
    (fun _ ->
      let rec lg k = if k <= 1 then 0 else 1 + lg (k / 2) in
      let start = Numpy.Array_creation.ones ~dtype:dt [| 1; 1 |] in
      let rec go h times =
        if times = 0 then h
        else
          let open Numpy.Extras in
          go
            (block
               (Group
                  [
                    Group [ Item h; Item h ];
                    Group [ Item h; Item (U.negative h) ];
                  ]))
            (times - 1)
      in
      go start (lg n))
    []

let convolution_matrix ?(mode = "full") a n =
  if n <= 0 then invalid_arg "n must be a positive integer";
  let m = (shape a).(Array.length (shape a) - 1) in
  if m < 1 then
    invalid_arg
      (Printf.sprintf "len(a) must be at least 1; got shape %s."
         (Util.tuple_string (shape a)));
  jit1 "convolution_matrix"
    ~key:(Printf.sprintf "%d %s" n mode)
    (function
      | [ a ] ->
          let rank = Array.length (shape a) in
          let widths =
            Array.init rank (fun d ->
                if d = rank - 1 then (0, n - 1) else (0, 0))
          in
          let az = NL.pad a widths 0.0 in
          let raz = NL.pad (NL.flip ~axis:[| -1 |] a) widths 0.0 in
          let total = m + n - 1 in
          let tb, te =
            match mode with
            | "same" ->
                let trim = Stdlib.min n m - 1 in
                (trim / 2, trim - (trim / 2))
            | "valid" -> (Stdlib.min n m - 1, Stdlib.min n m - 1)
            | "full" -> (0, 0)
            | other ->
                invalid_arg
                  ("mode must be one of 'full', 'valid', 'same'; got " ^ other)
          in
          let col0 = L.slice_in_dim ~axis:(-1) az tb (total - te) in
          let row0 =
            L.slice_in_dim ~axis:(-1) raz (total - n - tb) (total - tb)
          in
          toeplitz ~r:row0 col0
      | _ -> invalid_arg "convolution_matrix: arity")
    [ a ]

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
  let wrap_index ind size =
    L.select
      (L.lt ind (Core.splat (dtype_of ind) [||] 0.0))
      (L.add ind (Core.splat (dtype_of ind) [||] (float_of_int size)))
      ind
  in
  let element = Numpy.Indexing.dynamic_at2 in
  let line v ~row idx =
    let idx =
      if (Core.get_aval idx).weak_type || dtype_of idx <> Dtype.I32 then
        L.convert_element_type idx Dtype.I32
      else idx
    in
    let zero = Core.splat Dtype.I32 [||] 0.0 in
    let starts, sizes, axis =
      if row then begin
        let i = wrap_index idx (shape v).(0) in
        let j = wrap_index zero (shape v).(1) in
        ([ i; j ], [| 1; (shape v).(1) |], 0)
      end
      else begin
        let i = wrap_index zero (shape v).(0) in
        let j = wrap_index idx (shape v).(1) in
        ([ i; j ], [| (shape v).(0); 1 |], 1)
      end
    in
    Core.bind1 (Types.Squeeze [| axis |])
      [ L.dynamic_slice_normalized v starts sizes ]
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
        let wrapped = wrap_index flat total in
        let index =
          L.broadcast_in_dim
            (if dtype_of wrapped = Dtype.I32 then wrapped
             else L.convert_element_type wrapped Dtype.I32)
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
              s_sorted_unique = true;
              s_clip = false;
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
      let ea = NL.expand_dims (Numpy.Linalg.eigvals a) [| 1 |] in
      let eb = NL.expand_dims (Numpy.Linalg.eigvals b) [| 0 |] in
      let sum = U.add ea eb in
      let close = Numpy.Reductions.any (U.less (U.abs sum) tol) in
      List.hd
        (L.cond close
           (fun _ ->
             [
               U.multiply
                 (Numpy.Array_creation.zeros_like x)
                 (Numpy.Jnp_util.py_float Float_format.nan);
             ])
           (fun _ -> [ x ])
           [])
  | _ -> invalid_arg "solve_sylvester: arity"

let solve_sylvester ?(tol = 1e-8) a b q =
  let dt = Dtypes.to_inexact_dtype (NL.result_type [ a; b; q ]) in
  let a = NL.astype a dt and b = NL.astype b dt and q = NL.astype q dt in
  let tol = Numpy.Jnp_util.py_float tol in
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
