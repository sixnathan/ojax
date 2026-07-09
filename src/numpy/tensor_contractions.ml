module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let pref preferred vs =
  match preferred with Some d -> d | None -> NL.result_type vs

let astype v dt = if dtype v = dt then v else NL.astype v dt

let dot_general lc rc lb rb a b =
  C.bind1
    (T.Dot_general
       { lhs_contract = lc; rhs_contract = rc; lhs_batch = lb; rhs_batch = rb })
    [ a; b ]

let bcast_binop prim a b =
  let dt = NL.result_type [ a; b ] in
  let sh = NL.broadcast_shapes_n [ shape a; shape b ] in
  let a = NL.broadcast_to (astype a dt) sh in
  let b = NL.broadcast_to (astype b dt) sh in
  C.bind1 prim [ a; b ]

let dot ?preferred a b =
  let pdt = pref preferred [ a; b ] in
  let a = astype a pdt and b = astype b pdt in
  let na = ndim a and nb = ndim b in
  let lc, rc =
    if na = 0 || nb = 0 then ([||], [||])
    else if nb = 1 then ([| na - 1 |], [| 0 |])
    else ([| na - 1 |], [| nb - 2 |])
  in
  dot_general lc rc [||] [||] a b

let matmul ?preferred a b =
  let pdt = pref preferred [ a; b ] in
  let a = astype a pdt and b = astype b pdt in
  let na = ndim a and nb = ndim b in
  if na < 1 || nb < 1 then
    invalid_arg "matmul: inputs must have ndim at least 1";
  if na = 1 && nb = 1 then dot_general [| 0 |] [| 0 |] [||] [||] a b
  else if na = 1 then dot_general [| 0 |] [| nb - 2 |] [||] [||] a b
  else if nb = 1 then dot_general [| na - 1 |] [| 0 |] [||] [||] a b
  else begin
    let sa = shape a and sb = shape b in
    let a_batch = Array.sub sa 0 (na - 2) in
    let b_batch = Array.sub sb 0 (nb - 2) in
    let bc = NL.broadcast_shapes_n [ a_batch; b_batch ] in
    let nbat = Array.length bc in
    let a' =
      NL.broadcast_to a (Array.append bc [| sa.(na - 2); sa.(na - 1) |])
    in
    let b' =
      NL.broadcast_to b (Array.append bc [| sb.(nb - 2); sb.(nb - 1) |])
    in
    let idb = Array.init nbat Fun.id in
    dot_general [| nbat + 1 |] [| nbat |] idb idb a' b'
  end

let matvec x1 x2 =
  let n2 = ndim x2 in
  let x2e = NL.expand_dims x2 [| n2 |] in
  let r = matmul x1 x2e in
  NL.squeeze ~axis:[| ndim r - 1 |] r

let vecmat x1 x2 =
  let n1 = ndim x1 in
  let x1e = NL.expand_dims x1 [| n1 - 1 |] in
  let r = matmul x1e x2 in
  NL.squeeze ~axis:[| ndim r - 2 |] r

let vdot ?preferred a b = dot ?preferred (NL.ravel a) (NL.ravel b)

let vecdot ?(axis = -1) ?preferred x1 x2 =
  let pdt = pref preferred [ x1; x2 ] in
  let x1 = NL.moveaxis [| axis |] [| ndim x1 - 1 |] x1 in
  let x2 = NL.moveaxis [| axis |] [| ndim x2 - 1 |] x2 in
  let prod = bcast_binop T.Mul (astype x1 pdt) (astype x2 pdt) in
  C.bind1 (T.Reduce_sum [| ndim prod - 1 |]) [ prod ]

type td_axes = Ax_int of int | Ax_pair of int array * int array

let canon i n = if i < 0 then i + n else i

let tensordot ?preferred ?(axes = Ax_int 2) a b =
  let pdt = pref preferred [ a; b ] in
  let a = astype a pdt and b = astype b pdt in
  let na = ndim a and nb = ndim b in
  let lc, rc =
    match axes with
    | Ax_int k ->
        if k > min na nb then
          invalid_arg "tensordot: number of axes exceeds input ranks";
        (Array.init k (fun i -> na - k + i), Array.init k Fun.id)
    | Ax_pair (ax1, ax2) ->
        if Array.length ax1 <> Array.length ax2 then
          invalid_arg "tensordot: axes lists must have equal length";
        ( Array.map (fun i -> canon i na) ax1,
          Array.map (fun i -> canon i nb) ax2 )
  in
  dot_general lc rc [||] [||] a b

let inner ?preferred a b =
  if ndim a = 0 || ndim b = 0 then begin
    let pdt = pref preferred [ a; b ] in
    bcast_binop T.Mul (astype a pdt) (astype b pdt)
  end
  else tensordot ?preferred ~axes:(Ax_pair ([| -1 |], [| -1 |])) a b

let outer ?preferred a b =
  let pdt = pref preferred [ a; b ] in
  let a = NL.ravel (astype a pdt) and b = NL.ravel (astype b pdt) in
  let a2 = NL.expand_dims a [| 1 |] in
  let b2 = NL.expand_dims b [| 0 |] in
  bcast_binop T.Mul a2 b2
