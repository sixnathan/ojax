open Types
module L = Lax_api
module NF = Numpy.Fft

let b1 = Core.bind1
let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let rank v = Array.length (shape v)
let canon n a = if a < 0 then a + n else a

let real_of = function
  | Dtype.Complex128 | Dtype.F64 -> Dtype.F64
  | _ -> Dtype.F32

let complex_of = function
  | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
  | _ -> Dtype.Complex64

module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy

let w4 n k =
  let cdt = complex_of (dtype_of k) in
  let k = if dtype_of k = cdt then k else NL.astype k cdt in
  let n_arr =
    Concrete
      (Ndarray.of_complex cdt [||]
         [| { Complex.re = float_of_int n; im = 0.0 } |])
  in
  let coef =
    Concrete
      (Ndarray.of_complex cdt [||]
         [| { Complex.re = 0.0; im = -0.5 *. Float.pi } |])
  in
  let product = UF.multiply coef k in
  UF.exp (UF.true_divide product n_arr)

let other_axes ndim axis =
  Array.of_list (List.filter (fun a -> a <> axis) (List.init ndim Fun.id))

let interleave x axis =
  let size = (shape x).(axis) in
  let v0 = L.slice_in_dim ~stride:2 ~axis x 0 size in
  let sliced = L.slice_in_dim ~stride:2 ~axis x 1 size in
  let v1 = L.rev sliced [| axis |] in
  L.concatenate [ v0; v1 ] axis

let ortho_norm out axis =
  let dt = dtype_of out in
  let n = (shape out).(axis) in
  let first = L.full [| 1 |] 4.0 dt in
  let rest = L.full [| n - 1 |] 2.0 dt in
  let factor = L.concatenate [ first; rest ] 0 in
  let factor = L.expand_dims factor (other_axes (rank out) axis) in
  let scaled =
    UF.multiply factor
      (Concrete (Ndarray.of_floats dt [||] [| float_of_int n |]))
  in
  UF.true_divide out (L.sqrt scaled)

let pad_axis x axis n =
  match n with
  | None -> x
  | Some target ->
      let cfg =
        Array.mapi
          (fun a s -> if a = axis then (0, target - s, 0) else (0, 0, 0))
          (shape x)
      in
      b1 (Pad cfg) [ x; L.full [||] 0.0 (dtype_of x) ]

let check_norm name norm =
  match norm with
  | None | Some "backward" | Some "ortho" -> ()
  | Some other ->
      invalid_arg
        (Printf.sprintf "jax.scipy.fft.%s: norm=%s is not implemented" name
           other)

let check_type t = if t <> 2 then invalid_arg "Only DCT type 2 is implemented."

let is_complex v =
  match dtype_of v with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let inexact x =
  let dt = Dtypes.to_inexact_dtype (dtype_of x) in
  if dt = dtype_of x then x else NL.astype x dt

let two_real v =
  UF.multiply (Concrete (Ndarray.of_floats (dtype_of v) [||] [| 2.0 |])) v

let rec dct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type type_;
  check_norm "dct" norm;
  if is_complex x then begin
    let re = dct ~type_ ?n ~axis ?norm (UF.real x) in
    let im = dct ~type_ ?n ~axis ?norm (UF.imag x) in
    L.complex re im
  end
  else
    let axis = canon (rank x) axis in
    let x = pad_axis x axis n in
    let size = (shape x).(axis) in
    let v = interleave x axis in
    let spectrum = NF.fft ~axis v in
    let real_dt = dtype_of (UF.real spectrum) in
    let k =
      L.expand_dims
        (NL.arange ~dtype:real_dt (float_of_int size))
        (other_axes (rank x) axis)
    in
    let out = UF.multiply spectrum (w4 size k) in
    let out = two_real (UF.real out) in
    if norm = Some "ortho" then ortho_norm out axis else out

let deinterleave x axis =
  let size = (shape x).(axis) in
  let half = (size + 1) / 2 in
  let part lo hi =
    let start = Array.make (rank x) 0 and limit = Array.copy (shape x) in
    start.(axis) <- lo;
    limit.(axis) <- hi;
    L.slice x start limit
  in
  let v0 = part 0 half in
  let v1 = L.rev (part half size) [| axis |] in
  let out = Numpy.Array_creation.zeros ~dtype:(dtype_of x) (shape x) in
  let set_every target first updates =
    let count = (shape updates).(axis) in
    let positions =
      NL.arange ~start:(float_of_int first) ~step:2.0 ~dtype:Dtype.I32
        (float_of_int size)
    in
    L.scatter ~unique_indices:true target
      (L.broadcast_in_dim positions [| count; 1 |] [| 0 |])
      updates
      (Np_ints.mark_sorted_unique
         {
           update_window_dims = other_axes (rank x) axis;
           inserted_window_dims = [| axis |];
           scatter_dims_to_operand_dims = [| axis |];
           s_operand_batching_dims = [||];
           s_scatter_indices_batching_dims = [||];
         })
  in
  set_every (set_every out 0 v0) 1 v1

let rec idct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type type_;
  check_norm "idct" norm;
  if is_complex x then begin
    let re = idct ~type_ ?n ~axis ?norm (UF.real x) in
    let im = idct ~type_ ?n ~axis ?norm (UF.imag x) in
    L.complex re im
  end
  else
    let axis = canon (rank x) axis in
    let x = pad_axis x axis n in
    let size = (shape x).(axis) in
    let x = inexact x in
    let x = if norm = Some "ortho" then x else ortho_norm x axis in
    let x = ortho_norm x axis in
    let k =
      L.expand_dims
        (NL.arange ~dtype:(dtype_of x) (float_of_int size))
        (other_axes (rank x) axis)
    in
    let w = w4 size k in
    let x = NL.astype x (dtype_of w) in
    let x = UF.true_divide x (w4 size k) in
    let cdt = dtype_of x in
    let lit v =
      Concrete (Ndarray.of_complex cdt [||] [| { Complex.re = v; im = 0.0 } |])
    in
    let x = UF.multiply (UF.multiply x (lit 2.0)) (lit (float_of_int size)) in
    deinterleave (UF.real (NF.ifft ~axis x)) axis

let dct2 x ~axes:(axis1, axis2) ~norm =
  let n1 = (shape x).(axis1) and n2 = (shape x).(axis2) in
  let v = interleave (interleave x axis1) axis2 in
  let spectrum = NF.fftn ~axes:[ axis1; axis2 ] v in
  let vdt = dtype_of spectrum in
  let ks n axis =
    L.expand_dims
      (NL.arange ~dtype:vdt (float_of_int n))
      (other_axes (rank x) axis)
  in
  let k1 = ks n1 axis1 in
  let k2 = ks n2 axis2 in
  let w1 = w4 n1 k1 in
  let t1 = UF.multiply (w4 n2 k2) spectrum in
  let w2 = w4 n2 (UF.negative k2) in
  let flipped = NL.flip ~axis:[| axis2 |] spectrum in
  let rolled = NL.roll ~axis:[| axis2 |] flipped [| 1 |] in
  let t2 = UF.multiply w2 rolled in
  let out = UF.multiply w1 (UF.add t1 t2) in
  let out = two_real (UF.real out) in
  if norm = Some "ortho" then ortho_norm (ortho_norm out axis1) axis2 else out

let resolve_axes ?s ?axes x =
  let n = rank x in
  match (axes, s) with
  | Some a, _ -> List.map (canon n) a
  | None, Some sizes ->
      let k = List.length sizes in
      List.init k (fun i -> n - k + i)
  | None, None -> List.init n Fun.id

let pad_all x ~axes ~sizes =
  let cfg =
    Array.mapi
      (fun a sz ->
        let rec find ax ss =
          match (ax, ss) with
          | h :: _, v :: _ when h = a -> v - sz
          | _ :: tl, _ :: vt -> find tl vt
          | _ -> 0
        in
        (0, find axes sizes, 0))
      (shape x)
  in
  b1 (Pad cfg) [ x; L.full [||] 0.0 (dtype_of x) ]

let rec dctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type type_;
  check_norm "dctn" norm;
  if is_complex x then begin
    let re = dctn ~type_ ?s ?axes ?norm (UF.real x) in
    let im = dctn ~type_ ?s ?axes ?norm (UF.imag x) in
    L.complex re im
  end
  else
    let ax = resolve_axes ?s ?axes x in
    match ax with
    | [ a ] ->
        dct
          ?n:(match s with Some (v :: _) -> Some v | _ -> None)
          ~axis:a ?norm x
    | _ -> (
        let x =
          match s with Some sizes -> pad_all x ~axes:ax ~sizes | None -> x
        in
        match ax with
        | [ a1; a2 ] -> dct2 x ~axes:(a1, a2) ~norm
        | _ ->
            let rec blocks x = function
              | a1 :: a2 :: tl -> blocks (dctn ~axes:[ a1; a2 ] ?norm x) tl
              | [ a ] -> dctn ~axes:[ a ] ?norm x
              | [] -> x
            in
            blocks x ax)

let rec idctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type type_;
  check_norm "idctn" norm;
  if is_complex x then begin
    let re = idctn ~type_ ?s ?axes ?norm (UF.real x) in
    let im = idctn ~type_ ?s ?axes ?norm (UF.imag x) in
    L.complex re im
  end
  else
    let ax = resolve_axes ?s ?axes x in
    match ax with
    | [ a ] ->
        idct
          ?n:(match s with Some (v :: _) -> Some v | _ -> None)
          ~axis:a ?norm x
    | _ ->
        let x =
          match s with Some sizes -> pad_all x ~axes:ax ~sizes | None -> x
        in
        List.fold_left (fun v a -> idct ~axis:a ?norm v) x ax
