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
  | D.I32 | D.Bool -> D.F32
  | D.I64 -> D.F64

let to_inexact v = convert v (to_inexact_dtype (dtype v))
let to_bool v = if dtype v = D.Bool then v else convert v D.Bool

let promote_integer_dtype dt =
  let di = Dtypes.default_int_dtype () in
  match dt with
  | D.Bool -> di
  | D.I32 -> if di = D.I64 then D.I64 else D.I32
  | D.I64 -> D.I64
  | (D.F32 | D.F64) as f -> f

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
