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
let const_full dt sh x = Core.splat dt sh x

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

let sort ?axis ?stable ?descending p0 =
  Jnp_util.jit1 "sort"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axis, stable, descending) [])))
    (function
      | [ p0 ] -> sort ?axis ?stable ?descending p0
      | _ -> invalid_arg "sort: arity")
    [ p0 ]

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

let argsort ?axis ?stable ?descending ?dtype p0 =
  Jnp_util.jit1 "argsort"
    ~key:
      (Digest.to_hex
         (Digest.string
            (Marshal.to_string (axis, stable, descending, dtype) [])))
    (function
      | [ p0 ] -> argsort ?axis ?stable ?descending ?dtype p0
      | _ -> invalid_arg "argsort: arity")
    [ p0 ]

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

let lexsort_impl = lexsort

let lexsort ?axis keys =
  match keys with
  | [] -> invalid_arg "need sequence of keys with len > 0 in lexsort"
  | _ ->
      let key =
        "list,"
        ^ string_of_int (List.length keys)
        ^ ","
        ^ Option.fold ~none:"" ~some:string_of_int axis
      in
      Jnp_util.jit1 "lexsort" ~key (fun ks -> lexsort_impl ?axis ks) keys

let lexsort_array ?axis keys =
  let key = "array," ^ Option.fold ~none:"" ~some:string_of_int axis in
  Jnp_util.jit1 "lexsort" ~key
    (function
      | [ arr ] ->
          let n = (shape arr).(0) in
          let row i =
            let sh = shape arr in
            let start = Array.make (Array.length sh) 0 in
            start.(0) <- i;
            let limit = Array.copy sh in
            limit.(0) <- i + 1;
            let strides = Some (Array.make (Array.length sh) 1) in
            bind1 (T.Squeeze [| 0 |])
              [
                bind1
                  (T.Slice
                     { start_indices = start; limit_indices = limit; strides })
                  [ arr ];
              ]
          in
          lexsort_impl ?axis (List.init n row)
      | _ -> invalid_arg "lexsort: arity")
    [ keys ]

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

let partition ?axis p0 ~kth =
  Jnp_util.jit1 "partition"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (axis, kth) [])))
    (function
      | [ p0 ] -> partition ?axis p0 ~kth | _ -> invalid_arg "partition: arity")
    [ p0 ]

let sort ?axis ?stable ?descending a0 =
  Eager_jit.one "numpy.sorting.sort" (axis, stable, descending) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sort ?axis ?stable ?descending a0
      | _ -> assert false)

let argsort ?axis ?stable ?descending ?dtype a0 =
  Eager_jit.one "numpy.sorting.argsort" (axis, stable, descending, dtype)
    [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> argsort ?axis ?stable ?descending ?dtype a0
      | _ -> assert false)

let partition ?axis a0 ~kth =
  Eager_jit.one "numpy.sorting.partition" (axis, kth) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> partition ?axis a0 ~kth
      | _ -> assert false)
