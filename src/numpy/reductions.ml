module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let weak v = (get_aval v).T.weak_type
let ndim v = Array.length (shape v)
let size v = Array.fold_left ( * ) 1 (shape v)
let bind1 = C.bind1

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let zeros_like v = const_full (dtype v) (shape v) 0.0

let canonicalize_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let reshape v sh = bind1 (T.Reshape sh) [ v ]
let ravel v = reshape v [| size v |]

let broadcast_to v sh =
  let s = shape v in
  if s = sh then v
  else begin
    let nd_in = Array.length s and nd_out = Array.length sh in
    let dims = Array.init nd_in (fun i -> i + (nd_out - nd_in)) in
    bind1 (T.Broadcast_in_dim { shape = sh; dims }) [ v ]
  end

let is_inexact = function D.F32 | D.F64 -> true | _ -> false

let to_inexact_dtype = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.Complex64 | D.Complex128) as c -> c

let to_inexact v = convert v (to_inexact_dtype (dtype v))
let to_bool v = if dtype v = D.Bool then v else convert v D.Bool

let promote_integer_dtype dt =
  let di = Dtypes.default_int_dtype () in
  match dt with
  | D.Bool -> di
  | D.I32 -> if di = D.I64 then D.I64 else D.I32
  | D.I64 -> D.I64
  | D.Uint32 -> D.Uint32
  | (D.F32 | D.F64 | D.Complex64 | D.Complex128) as f -> f

let all_axes v = Array.init (ndim v) (fun i -> i)

let resolve_axes v = function
  | None -> all_axes v
  | Some ax -> Array.map (fun i -> canonicalize_axis i (ndim v)) ax

let mem_axis axes i = Array.exists (fun a -> a = i) axes

let keepdims_shape orig axes =
  Array.mapi (fun i d -> if mem_axis axes i then 1 else d) orig

let reduce_axes prim ?(keepdims = false) axes v =
  let orig = shape v in
  let r = bind1 (prim axes) [ v ] in
  if keepdims then reshape r (keepdims_shape orig axes) else r

let axis_count v axes =
  let s = shape v in
  Array.fold_left (fun acc i -> acc * s.(i)) 1 axes

let isnan v = bind1 T.Ne [ v; v ]
let where c x y = bind1 T.Select_n [ c; y; x ]

let check_where name = function
  | None -> None
  | Some w ->
      if dtype w <> D.Bool then
        invalid_arg
          (Printf.sprintf "jnp.%s: where must be None or a boolean array." name);
      Some w

let sum ?axis ?(keepdims = false) a =
  let axes = resolve_axes a axis in
  let n =
    if dtype a = D.Bool then convert a (Dtypes.default_int_dtype ()) else a
  in
  let rt = promote_integer_dtype (dtype n) in
  reduce_axes (fun ax -> T.Reduce_sum ax) ~keepdims axes (convert n rt)

let prod ?axis ?(keepdims = false) a =
  let axes = resolve_axes a axis in
  let n =
    if dtype a = D.Bool then convert a (Dtypes.default_int_dtype ()) else a
  in
  let rt = promote_integer_dtype (dtype n) in
  reduce_axes (fun ax -> T.Reduce_prod ax) ~keepdims axes (convert n rt)

let max ?axis ?(keepdims = false) a =
  reduce_axes (fun ax -> T.Reduce_max ax) ~keepdims (resolve_axes a axis) a

let min ?axis ?(keepdims = false) a =
  reduce_axes (fun ax -> T.Reduce_min ax) ~keepdims (resolve_axes a axis) a

let amax = max
let amin = min

let all ?axis ?(keepdims = false) a =
  reduce_axes
    (fun ax -> T.Reduce_and ax)
    ~keepdims (resolve_axes a axis) (to_bool a)

let any ?axis ?(keepdims = false) a =
  reduce_axes
    (fun ax -> T.Reduce_or ax)
    ~keepdims (resolve_axes a axis) (to_bool a)

let mean_over cdt axes ?(keepdims = false) a =
  let s = reduce_axes (fun ax -> T.Reduce_sum ax) ~keepdims axes a in
  let n = float_of_int (axis_count a axes) in
  bind1 T.Div [ s; const_full cdt (shape s) n ]

let mean ?axis ?(keepdims = false) a =
  let axes = resolve_axes a axis in
  let cdt = to_inexact_dtype (dtype a) in
  mean_over cdt axes ~keepdims (convert a cdt)

let var ?axis ?(keepdims = false) ?(ddof = 0) a =
  let axes = resolve_axes a axis in
  let cdt = to_inexact_dtype (dtype a) in
  let af = convert a cdt in
  let m = broadcast_to (mean_over cdt axes ~keepdims:true af) (shape af) in
  let centered = bind1 T.Square [ bind1 T.Sub [ af; m ] ] in
  let s = reduce_axes (fun ax -> T.Reduce_sum ax) ~keepdims axes centered in
  let n = axis_count a axes - ddof in
  if n > 0 then bind1 T.Div [ s; const_full cdt (shape s) (float_of_int n) ]
  else const_full cdt (shape s) Float.nan

let std ?axis ?(keepdims = false) ?(ddof = 0) a =
  bind1 T.Sqrt [ var ?axis ~keepdims ~ddof a ]

let ptp ?axis ?(keepdims = false) a =
  bind1 T.Sub [ amax ?axis ~keepdims a; amin ?axis ~keepdims a ]

let count_nonzero ?axis ?(keepdims = false) a =
  sum ?axis ~keepdims (bind1 T.Ne [ a; zeros_like a ])

let average ?axis ?(keepdims = false) ?weights a =
  match weights with
  | None -> mean ?axis ~keepdims (to_inexact a)
  | Some w ->
      let rt, _ = Dtypes.result_type [ (dtype a, weak a); (dtype w, weak w) ] in
      let cdt = to_inexact_dtype rt in
      let sh = shape a in
      let a2 = convert a cdt and w2 = broadcast_to (convert w cdt) sh in
      let num =
        reduce_axes
          (fun ax -> T.Reduce_sum ax)
          ~keepdims (resolve_axes a axis)
          (bind1 T.Mul [ a2; w2 ])
      in
      let den =
        reduce_axes
          (fun ax -> T.Reduce_sum ax)
          ~keepdims (resolve_axes a axis) w2
      in
      bind1 T.Div [ num; den ]

let nan_reduce ~reduce ~init ~nan_if_all_nan ?axis ?(keepdims = false) a =
  if not (is_inexact (dtype a)) then reduce ?axis ?keepdims:(Some keepdims) a
  else begin
    let filled = where (isnan a) (const_full (dtype a) (shape a) init) a in
    let out = reduce ?axis ?keepdims:(Some keepdims) filled in
    if nan_if_all_nan then
      let allnan = all ?axis ~keepdims (isnan a) in
      where allnan (const_full (dtype out) (shape out) Float.nan) out
    else out
  end

let nansum ?axis ?(keepdims = false) a =
  nan_reduce ~reduce:sum ~init:0.0 ~nan_if_all_nan:false ?axis ~keepdims a

let nanprod ?axis ?(keepdims = false) a =
  nan_reduce ~reduce:prod ~init:1.0 ~nan_if_all_nan:false ?axis ~keepdims a

let nanmax ?axis ?(keepdims = false) a =
  nan_reduce ~reduce:amax ~init:neg_infinity ~nan_if_all_nan:true ?axis
    ~keepdims a

let nanmin ?axis ?(keepdims = false) a =
  nan_reduce ~reduce:amin ~init:infinity ~nan_if_all_nan:true ?axis ~keepdims a

let nanmean ?axis ?(keepdims = false) a =
  if not (is_inexact (dtype a)) then mean ?axis ~keepdims a
  else begin
    let axes = resolve_axes a axis in
    let cdt = to_inexact_dtype (dtype a) in
    let notnan = bind1 T.Not [ isnan a ] in
    let normalizer =
      reduce_axes
        (fun ax -> T.Reduce_sum ax)
        ~keepdims axes (convert notnan cdt)
    in
    bind1 T.Div [ nansum ?axis ~keepdims a; normalizer ]
  end

let nanvar ?axis ?(keepdims = false) ?(ddof = 0) a =
  let axes = resolve_axes a axis in
  let cdt = to_inexact_dtype (dtype a) in
  let af = convert a cdt in
  let m = broadcast_to (nanmean ~axis:axes ~keepdims:true af) (shape af) in
  let centered =
    where (isnan af) (const_full cdt (shape af) 0.0) (bind1 T.Sub [ af; m ])
  in
  let centered = bind1 T.Square [ centered ] in
  let notnan = bind1 T.Not [ isnan af ] in
  let count =
    reduce_axes (fun ax -> T.Reduce_sum ax) ~keepdims axes (convert notnan cdt)
  in
  let nz =
    bind1 T.Sub [ count; const_full cdt (shape count) (float_of_int ddof) ]
  in
  let mask = bind1 T.Le [ nz; const_full cdt (shape nz) 0.0 ] in
  let result =
    reduce_axes (fun ax -> T.Reduce_sum ax) ~keepdims axes centered
  in
  let result = where mask (const_full cdt (shape result) Float.nan) result in
  let divisor = where mask (const_full cdt (shape nz) 1.0) nz in
  bind1 T.Div [ result; divisor ]

let nanstd ?axis ?(keepdims = false) ?(ddof = 0) a =
  bind1 T.Sqrt [ nanvar ?axis ~keepdims ~ddof a ]

let cumsum ?axis a =
  let a, ax =
    match axis with
    | None -> (ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let rt = if dtype a = D.Bool then Dtypes.default_int_dtype () else dtype a in
  bind1 (T.Cumsum { axis = ax; reverse = false }) [ convert a rt ]

let cum ?(fill = None) mk_prim ?axis a =
  let a, ax =
    match axis with
    | None -> (ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let a =
    match fill with
    | Some fv when is_inexact (dtype a) ->
        where (isnan a) (const_full (dtype a) (shape a) fv) a
    | _ -> a
  in
  let rt = if dtype a = D.Bool then Dtypes.default_int_dtype () else dtype a in
  bind1 (mk_prim ax) [ convert a rt ]

let cumprod ?axis a =
  cum (fun ax -> T.Cumprod { axis = ax; reverse = false }) ?axis a

let nancumsum ?axis a =
  cum ~fill:(Some 0.0)
    (fun ax -> T.Cumsum { axis = ax; reverse = false })
    ?axis a

let nancumprod ?axis a =
  cum ~fill:(Some 1.0)
    (fun ax -> T.Cumprod { axis = ax; reverse = false })
    ?axis a

let cum_api_axis kind x axis =
  if ndim x = 0 then
    invalid_arg
      (Printf.sprintf
         "The input must be non-scalar to take a cumulative %s, however a \
          scalar value or scalar array was given."
         kind);
  match axis with
  | None ->
      if ndim x > 1 then
        invalid_arg
          (Printf.sprintf
             "The input array has rank %d, however axis was not set to an \
              explicit value. The axis argument is only optional for \
              one-dimensional arrays."
             (ndim x));
      0
  | Some ax -> canonicalize_axis ax (ndim x)

let cumulative_sum ?axis ?(include_initial = false) x =
  let ax = cum_api_axis "sum" x axis in
  let rt = promote_integer_dtype (dtype x) in
  let out = bind1 (T.Cumsum { axis = ax; reverse = false }) [ convert x rt ] in
  if include_initial then begin
    let zsh = Array.copy (shape x) in
    zsh.(ax) <- 1;
    bind1 (T.Concatenate ax) [ const_full rt zsh 0.0; out ]
  end
  else out

let cumulative_prod ?axis ?(include_initial = false) x =
  let ax = cum_api_axis "product" x axis in
  let rt = if dtype x = D.Bool then Dtypes.default_int_dtype () else dtype x in
  let out = bind1 (T.Cumprod { axis = ax; reverse = false }) [ convert x rt ] in
  if include_initial then begin
    let osh = Array.copy (shape x) in
    osh.(ax) <- 1;
    bind1 (T.Concatenate ax) [ const_full rt osh 1.0; out ]
  end
  else out

let transpose_perm v perm = bind1 (T.Transpose perm) [ v ]

let move_to_last v ax =
  let nd = ndim v in
  if ax = nd - 1 then v
  else begin
    let perm = Array.make nd 0 in
    let k = ref 0 in
    for i = 0 to nd - 1 do
      if i <> ax then begin
        perm.(!k) <- i;
        incr k
      end
    done;
    perm.(nd - 1) <- ax;
    transpose_perm v perm
  end

let reduce_last_sum v = bind1 (T.Reduce_sum [| ndim v - 1 |]) [ v ]
let reduce_last_or v = bind1 (T.Reduce_or [| ndim v - 1 |]) [ v ]

let quantile_core a q ~axis ~method_ ~keepdims ~squash_nans =
  let a = to_inexact a in
  let cdt = dtype a in
  let nd0 = ndim a in
  let a, ax, keepdim_shape =
    match axis with
    | None ->
        let ks = if keepdims then Some (Array.make nd0 1) else None in
        (ravel a, 0, ks)
    | Some ax0 ->
        let ax = canonicalize_axis ax0 nd0 in
        let ks =
          if keepdims then begin
            let s = Array.copy (shape a) in
            s.(ax) <- 1;
            Some s
          end
          else None
        in
        (a, ax, ks)
  in
  let ndc = ndim a in
  let am = move_to_last a ax in
  let amsh = shape am in
  let n = amsh.(ndc - 1) in
  let bsh = Array.sub amsh 0 (ndc - 1) in
  let q = convert q cdt in
  let qsh = shape q in
  let qnd = Array.length qsh in
  let target = Array.append qsh bsh in
  let counts =
    if squash_nans then reduce_last_sum (convert (bind1 T.Not [ isnan am ]) cdt)
    else const_full cdt bsh (float_of_int n)
  in
  let am =
    if squash_nans then where (isnan am) (const_full cdt (shape am) infinity) am
    else am
  in
  let am_sorted =
    bind1
      (T.Sort { dimension = ndc - 1; is_stable = true; num_keys = 1 })
      [ am ]
  in
  let q_r = reshape q (Array.append qsh (Array.make (ndc - 1) 1)) in
  let counts_r = reshape counts (Array.append (Array.make qnd 1) bsh) in
  let q_full = broadcast_to q_r target in
  let counts_full = broadcast_to counts_r target in
  let cm1 = bind1 T.Sub [ counts_full; const_full cdt target 1.0 ] in
  let q_scaled = bind1 T.Mul [ q_full; cm1 ] in
  let low = bind1 T.Floor [ q_scaled ] in
  let high = bind1 T.Ceil [ q_scaled ] in
  let high_weight = bind1 T.Sub [ q_scaled; low ] in
  let low_weight = bind1 T.Sub [ const_full cdt target 1.0; high_weight ] in
  let clamp v =
    bind1 T.Max [ const_full cdt target 0.0; bind1 T.Min [ v; cm1 ] ]
  in
  let low_c = clamp low in
  let high_c = clamp high in
  let gshape = Array.append target [| n |] in
  let ng = Array.length gshape in
  let iota =
    bind1 (T.Iota { dtype = cdt; shape = gshape; dimension = ng - 1 }) []
  in
  let am_b = broadcast_to am_sorted gshape in
  let take idx =
    let idx_e = reshape idx (Array.append target [| 1 |]) in
    let idx_b = broadcast_to idx_e gshape in
    let onehot = bind1 T.Eq [ iota; idx_b ] in
    let masked = where onehot am_b (const_full cdt gshape 0.0) in
    reduce_last_sum masked
  in
  let low_value = take low_c in
  let high_value = take high_c in
  let result =
    match method_ with
    | "linear" ->
        bind1 T.Add
          [
            bind1 T.Mul [ low_value; low_weight ];
            bind1 T.Mul [ high_value; high_weight ];
          ]
    | "lower" -> low_value
    | "higher" -> high_value
    | "nearest" ->
        let pred = bind1 T.Le [ high_weight; const_full cdt target 0.5 ] in
        where pred low_value high_value
    | "midpoint" ->
        bind1 T.Mul
          [ bind1 T.Add [ low_value; high_value ]; const_full cdt target 0.5 ]
    | m -> invalid_arg ("jnp.quantile: unsupported method " ^ m)
  in
  let result =
    if squash_nans then result
    else begin
      let anynan = reduce_last_or (isnan am) in
      let anynan_r = reshape anynan (Array.append (Array.make qnd 1) bsh) in
      let anynan_full = broadcast_to anynan_r target in
      where anynan_full (const_full cdt target Float.nan) result
    end
  in
  match keepdim_shape with
  | None -> result
  | Some ks -> reshape result (Array.append qsh ks)

let quantile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  quantile_core a q ~axis ~method_ ~keepdims ~squash_nans:false

let nanquantile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  quantile_core a q ~axis ~method_ ~keepdims ~squash_nans:true

let percentile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  let cdt = to_inexact_dtype (dtype a) in
  let q100 = bind1 T.Div [ convert q cdt; const_full cdt (shape q) 100.0 ] in
  quantile ?axis ~keepdims ~method_ a q100

let nanpercentile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  let cdt = to_inexact_dtype (dtype a) in
  let q100 = bind1 T.Div [ convert q cdt; const_full cdt (shape q) 100.0 ] in
  nanquantile ?axis ~keepdims ~method_ a q100

let median ?axis ?(keepdims = false) a =
  let q = const_full (to_inexact_dtype (dtype a)) [||] 0.5 in
  quantile ?axis ~keepdims ~method_:"midpoint" a q

let nanmedian ?axis ?(keepdims = false) a =
  let q = const_full (to_inexact_dtype (dtype a)) [||] 0.5 in
  nanquantile ?axis ~keepdims ~method_:"midpoint" a q
