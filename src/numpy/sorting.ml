module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let bind = C.bind
let bind1 = C.bind1
let prod sh = Array.fold_left ( * ) 1 sh

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (prod sh) x))

let canonicalize_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let broadcasted_iota dt sh dim =
  bind1 (T.Iota { dtype = dt; shape = sh; dimension = dim }) []

let rev dim v = bind1 (T.Rev [| dim |]) [ v ]

let sort ?(axis = Some (-1)) ?(stable = true) ?(descending = false) a =
  let arr, dim =
    match axis with
    | None -> (NL.ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let r =
    bind1 (T.Sort { dimension = dim; is_stable = stable; num_keys = 1 }) [ arr ]
  in
  if descending then rev dim r else r

let argsort ?(axis = Some (-1)) ?(stable = true) ?(descending = false) ?dtype a
    =
  let arr, dim =
    match axis with
    | None -> (NL.ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let idx_dtype =
    match dtype with Some d -> d | None -> Dtypes.default_int_dtype ()
  in
  let iota = broadcasted_iota idx_dtype (shape arr) dim in
  let arr, iota =
    if descending && stable then (rev dim arr, rev dim iota) else (arr, iota)
  in
  let outs =
    bind
      (T.Sort { dimension = dim; is_stable = stable; num_keys = 1 })
      [ arr; iota ]
  in
  let indices = List.nth outs 1 in
  if descending then rev dim indices else indices

let lexsort ?(axis = -1) keys =
  match keys with
  | [] -> invalid_arg "need sequence of keys with len > 0 in lexsort"
  | k0 :: _ ->
      if List.exists (fun k -> shape k <> shape k0) keys then
        invalid_arg "all keys need to be the same shape";
      let nd = ndim k0 in
      let idt = Dtypes.default_int_dtype () in
      if nd = 0 then const_full idt [||] 0.0
      else begin
        let ax = canonicalize_axis axis nd in
        let iota = broadcasted_iota idt (shape k0) ax in
        let operands = List.rev keys @ [ iota ] in
        let outs =
          bind
            (T.Sort
               { dimension = ax; is_stable = true; num_keys = List.length keys })
            operands
        in
        List.nth outs (List.length outs - 1)
      end

let neg v = bind1 T.Neg [ v ]
let topk_vals k axis v = List.hd (bind (T.Top_k { k; axis }) [ v ])

let partition ?(axis = -1) a ~kth =
  let nd = ndim a in
  let axis = canonicalize_axis axis nd in
  let kth = canonicalize_axis kth (shape a).(axis) in
  let last = nd - 1 in
  let arr = NL.swapaxes axis last a in
  let n = (shape arr).(last) in
  let bottom = neg (topk_vals (kth + 1) last (neg arr)) in
  let top = topk_vals (n - kth - 1) last arr in
  let out = NL.concatenate ~axis:last [ bottom; top ] in
  NL.swapaxes last axis out
