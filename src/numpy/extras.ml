open Types
module NL = Lax_numpy
module U = Ufuncs
module R = Reductions
module AC = Array_creation
module I = Indexing
module L = Lax_api

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let size_of v = Array.fold_left ( * ) 1 (shape v)
let rank v = Array.length (shape v)
let canon n a = if a < 0 then a + n else a
let default_int () = Dtypes.default_int_dtype ()
let default_float () = Dtypes.default_float_dtype ()

let host name v =
  match v with
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ ->
      raise
        (Errors.Non_concrete_boolean_index_error
           (name
          ^ " requires concrete input; pass a size argument to use it under jit"
           ))

let flat_floats nd =
  Array.of_list (List.rev (Ndarray.fold (fun acc x -> x :: acc) [] nd))

let ints_value dt shape_ xs =
  Concrete (Ndarray.of_floats dt shape_ (Array.map float_of_int xs))

let int_scalar i = ints_value (default_int ()) [||] [| i |]
let scalar_like v x = L.full [||] x (dtype_of v)
let flat v = NL.reshape v [| size_of v |]
let shape_int a = Array.fold_left ( * ) 1 a

let is_float_dtype = function
  | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let as_inexact v =
  if is_float_dtype (dtype_of v) then v else NL.astype v (default_float ())

let nonzero_mask a =
  match dtype_of a with
  | Dtype.Bool -> a
  | _ -> U.not_equal a (scalar_like a 0.0)

let strides_of sh =
  let n = Array.length sh in
  let s = Array.make n 1 in
  for d = n - 2 downto 0 do
    s.(d) <- s.(d + 1) * sh.(d + 1)
  done;
  s

let nonzero ?size ?(fill_value = 0) a =
  let sh = shape a in
  let n = Array.length sh in
  let strides = strides_of sh in
  match size with
  | None ->
      let data = flat_floats (host "nonzero" a) in
      let hits =
        List.filter
          (fun i -> data.(i) <> 0.0)
          (List.init (Array.length data) Fun.id)
      in
      let count = List.length hits in
      List.init n (fun d ->
          ints_value (default_int ()) [| count |]
            (Array.of_list
               (List.map (fun i -> i / strides.(d) mod sh.(d)) hits)))
  | Some k ->
      let mask = flat (nonzero_mask a) in
      let total = size_of a in
      let order =
        Sorting.argsort ~axis:(Some 0) ~stable:true (U.logical_not mask)
      in
      let padded =
        if k <= total then L.slice_in_dim order 0 k
        else
          NL.concatenate
            [ order; AC.zeros ~dtype:(dtype_of order) [| k - total |] ]
      in
      let hits = I.take ~axis:0 mask order in
      let valid =
        if k <= total then L.slice_in_dim hits 0 k
        else NL.concatenate [ hits; AC.zeros ~dtype:Dtype.Bool [| k - total |] ]
      in
      List.init n (fun d ->
          let dt = dtype_of padded in
          let stride = L.full [| k |] (float_of_int strides.(d)) dt in
          let dim = L.full [| k |] (float_of_int sh.(d)) dt in
          let coord = U.remainder (U.floor_divide padded stride) dim in
          NL.where_ valid coord (L.full [| k |] (float_of_int fill_value) dt))

let flatnonzero ?size ?fill_value a =
  List.hd (nonzero ?size ?fill_value (flat a))

let argwhere ?size ?fill_value a =
  let a = if rank a = 0 then NL.reshape a [| 1 |] else a in
  match nonzero ?size ?fill_value a with
  | [] -> AC.zeros ~dtype:(default_int ()) [| 0; 0 |]
  | parts -> NL.stack ~axis:1 parts

type unique_result = {
  values : value;
  indices : value;
  inverse : value;
  counts : value;
}

let compare_with_nan a b =
  match (Float.is_nan a, Float.is_nan b) with
  | true, true -> 0
  | true, false -> 1
  | false, true -> -1
  | false, false -> compare a b

let unique_full ?axis ?size ?(fill_value = 0.0) a =
  let nd = host "unique" a in
  let dt = Ndarray.dtype nd in
  let sh = Ndarray.shape nd in
  let data = flat_floats nd in
  let axis = Option.map (canon (Array.length sh)) axis in
  let rows, row_len, moved_shape, row_values =
    match axis with
    | None ->
        let n = Array.length data in
        (n, 1, [| n |], fun i -> [| data.(i) |])
    | Some ax ->
        let moved = NL.moveaxis [| ax |] [| 0 |] a in
        let mdata = flat_floats (host "unique" moved) in
        let msh = shape moved in
        let r = msh.(0) in
        let len = if r = 0 then 0 else Array.length mdata / r in
        (r, len, msh, fun i -> Array.sub mdata (i * len) len)
  in
  let compare_rows i j =
    let x = row_values i and y = row_values j in
    let rec go k =
      if k = row_len then 0
      else
        let c = compare_with_nan x.(k) y.(k) in
        if c <> 0 then c else go (k + 1)
    in
    go 0
  in
  let order = List.stable_sort compare_rows (List.init rows Fun.id) in
  let groups =
    List.fold_left
      (fun acc i ->
        match acc with
        | (first, members) :: rest when compare_rows first i = 0 ->
            (first, i :: members) :: rest
        | _ -> (i, [ i ]) :: acc)
      [] order
    |> List.rev
  in
  let n_unique = List.length groups in
  let inverse = Array.make rows 0 in
  List.iteri
    (fun g (_, members) -> List.iter (fun i -> inverse.(i) <- g) members)
    groups;
  let firsts = Array.of_list (List.map fst groups) in
  let counts_arr =
    Array.of_list (List.map (fun (_, m) -> List.length m) groups)
  in
  let k = match size with Some s -> s | None -> n_unique in
  let idt = default_int () in
  let out_rows = k in
  let value_at r col =
    if r < n_unique then (row_values firsts.(r)).(col)
    else match size with Some _ -> fill_value | None -> 0.0
  in
  let row_shape = Array.sub moved_shape 1 (Array.length moved_shape - 1) in
  let values_flat =
    Array.init (out_rows * row_len) (fun i ->
        value_at (i / Stdlib.max row_len 1) (i mod Stdlib.max row_len 1))
  in
  let values =
    match axis with
    | None -> Concrete (Ndarray.of_floats dt [| out_rows |] values_flat)
    | Some ax ->
        let moved =
          Concrete
            (Ndarray.of_floats dt
               (Array.append [| out_rows |] row_shape)
               values_flat)
        in
        NL.moveaxis [| 0 |] [| ax |] moved
  in
  let pad_or_cut src fill =
    Array.init out_rows (fun i ->
        if i < Array.length src then src.(i) else fill)
  in
  {
    values;
    indices = ints_value idt [| out_rows |] (pad_or_cut firsts 0);
    inverse =
      (match axis with
      | None -> ints_value idt sh inverse
      | Some _ -> ints_value idt [| rows |] inverse);
    counts = ints_value idt [| out_rows |] (pad_or_cut counts_arr 0);
  }

let unique_values ?size ?fill_value a = (unique_full ?size ?fill_value a).values

let unique_counts ?size ?fill_value a =
  let r = unique_full ?size ?fill_value a in
  (r.values, r.counts)

let unique_inverse ?size ?fill_value a =
  let r = unique_full ?size ?fill_value a in
  (r.values, r.inverse)

let unique_all ?size ?fill_value a = unique_full ?size ?fill_value a

let unique ?(return_index = false) ?(return_inverse = false)
    ?(return_counts = false) ?axis ?size ?fill_value a =
  let r = unique_full ?axis ?size ?fill_value a in
  r.values
  :: List.concat
       [
         (if return_index then [ r.indices ] else []);
         (if return_inverse then [ r.inverse ] else []);
         (if return_counts then [ r.counts ] else []);
       ]

let same_dtype a b =
  let dt = NL.result_type [ a; b ] in
  (NL.astype a dt, NL.astype b dt)

let sorted_unique_floats name v =
  let data = flat_floats (host name v) in
  List.sort_uniq compare_with_nan (Array.to_list data)

let of_float_list dt xs =
  let arr = Array.of_list xs in
  Concrete (Ndarray.of_floats dt [| Array.length arr |] arr)

let union1d a b =
  let a, b = same_dtype a b in
  let xs =
    List.sort_uniq compare_with_nan
      (sorted_unique_floats "union1d" a @ sorted_unique_floats "union1d" b)
  in
  of_float_list (dtype_of a) xs

let intersect1d a b =
  let a, b = same_dtype a b in
  let ys = sorted_unique_floats "intersect1d" b in
  of_float_list (dtype_of a)
    (List.filter
       (fun x -> List.exists (fun y -> compare_with_nan x y = 0) ys)
       (sorted_unique_floats "intersect1d" a))

let setdiff1d a b =
  let a, b = same_dtype a b in
  let ys = sorted_unique_floats "setdiff1d" b in
  of_float_list (dtype_of a)
    (List.filter
       (fun x -> not (List.exists (fun y -> compare_with_nan x y = 0) ys))
       (sorted_unique_floats "setdiff1d" a))

let setxor1d a b =
  let a, b = same_dtype a b in
  let xs = sorted_unique_floats "setxor1d" a in
  let ys = sorted_unique_floats "setxor1d" b in
  let only p q =
    List.filter
      (fun x -> not (List.exists (fun y -> compare_with_nan x y = 0) q))
      p
  in
  of_float_list (dtype_of a)
    (List.sort compare_with_nan (only xs ys @ only ys xs))

let sum_at n data ids =
  let m = size_of ids in
  let indices = NL.reshape ids [| m; 1 |] in
  let dims =
    {
      update_window_dims = [||];
      inserted_window_dims = [| 0 |];
      scatter_dims_to_operand_dims = [| 0 |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
    }
  in
  L.scatter_add (AC.zeros ~dtype:(dtype_of data) [| n |]) indices data dims

let bincount ?weights ?length x =
  let n =
    match length with
    | Some l -> l
    | None ->
        let data = flat_floats (host "bincount" x) in
        1 + int_of_float (Array.fold_left Float.max (-1.0) data)
  in
  let x = flat x in
  let weights =
    match weights with
    | Some w -> flat w
    | None -> AC.ones ~dtype:(default_int ()) [| size_of x |]
  in
  sum_at n weights x

let histogram_bin_edges ?(bins = 10) ?range a =
  let a = as_inexact (flat a) in
  let dt = dtype_of a in
  let lo, hi =
    match range with
    | Some (l, h) -> (L.full [||] l dt, L.full [||] h dt)
    | None -> (R.min a, R.max a)
  in
  let same = U.equal lo hi in
  let half = L.full [||] 0.5 dt in
  let lo' = NL.where_ same (U.subtract lo half) lo in
  let hi' = NL.where_ same (U.add hi half) hi in
  let fractions =
    Concrete
      (Ndarray.of_floats dt [| bins |]
         (Array.init bins (fun i -> float_of_int i /. float_of_int bins)))
  in
  let complement = U.subtract (L.full [| bins |] 1.0 dt) fractions in
  let broadcast v = L.broadcast_in_dim v [| bins |] [||] in
  let body =
    U.add
      (U.multiply (broadcast lo') complement)
      (U.multiply (broadcast hi') fractions)
  in
  NL.concatenate [ body; NL.reshape hi' [| 1 |] ]

let histogram_core a edges weights density =
  let a = as_inexact (flat a) in
  let n_edges = size_of edges in
  let idx = NL.searchsorted ~side:"right" edges a in
  let last = L.index_in_dim ~keepdims:false edges (n_edges - 1) in
  let at_end = U.equal a (L.broadcast_in_dim last (shape a) [||]) in
  let idx =
    NL.where_ at_end
      (L.full (shape idx) (float_of_int (n_edges - 1)) (dtype_of idx))
      idx
  in
  let weights =
    match weights with
    | Some w -> flat w
    | None -> AC.ones ~dtype:(dtype_of a) (shape a)
  in
  let summed = sum_at n_edges weights idx in
  let counts = L.slice_in_dim summed 1 n_edges in
  if density then
    let widths = NL.diff edges in
    let total = R.sum counts in
    U.divide (U.divide counts widths)
      (L.broadcast_in_dim total (shape counts) [||])
  else counts

let histogram ?(bins = 10) ?range ?weights ?(density = false) a =
  let edges = histogram_bin_edges ~bins ?range a in
  (histogram_core a edges weights density, edges)

let histogram_edges ?weights ?(density = false) a edges =
  (histogram_core a edges weights density, edges)

let ravel_multi_index ?(mode = "raise") ?(order = "C") coords dims =
  let n = Array.length dims in
  let strides =
    match order with
    | "C" -> strides_of dims
    | "F" ->
        let s = Array.make n 1 in
        for d = 1 to n - 1 do
          s.(d) <- s.(d - 1) * dims.(d - 1)
        done;
        s
    | other -> invalid_arg ("Unrecognized order=" ^ other)
  in
  let clip c d =
    match mode with
    | "clip" -> NL.clip ~min:0.0 ~max:(float_of_int (d - 1)) c
    | "wrap" -> U.remainder c (L.full (shape c) (float_of_int d) (dtype_of c))
    | "raise" -> c
    | other -> invalid_arg ("invalid mode=" ^ other)
  in
  let parts =
    List.mapi
      (fun i c ->
        let c = clip c dims.(i) in
        U.multiply c (L.full (shape c) (float_of_int strides.(i)) (dtype_of c)))
      coords
  in
  match parts with
  | [] -> invalid_arg "ravel_multi_index needs coordinates"
  | first :: rest -> List.fold_left U.add first rest

let histogramdd ?(bins = [||]) ?range ?weights ?(density = false) sample =
  let n_dims = (shape sample).(1) in
  let bins_for i = if Array.length bins = 0 then 10 else bins.(i) in
  let column i = I.take ~axis:1 sample (int_scalar i) in
  let edges =
    List.init n_dims (fun i ->
        let range_i = Option.map (fun r -> r.(i)) range in
        histogram_bin_edges ~bins:(bins_for i) ?range:range_i (column i))
  in
  let idxs =
    List.mapi
      (fun i e ->
        let c = as_inexact (column i) in
        let idx = NL.searchsorted ~side:"right" e c in
        let last = L.index_in_dim ~keepdims:false e (size_of e - 1) in
        let at_end = U.equal c (L.broadcast_in_dim last (shape c) [||]) in
        NL.where_ at_end
          (U.subtract idx (L.full (shape idx) 1.0 (dtype_of idx)))
          idx)
      edges
  in
  let nbins = Array.of_list (List.map (fun e -> size_of e + 1) edges) in
  let xy = ravel_multi_index ~mode:"clip" idxs nbins in
  let hist = bincount ?weights ~length:(shape_int nbins) xy in
  let reshaped = NL.reshape hist nbins in
  let start = Array.make n_dims 1 in
  let limit = Array.map (fun d -> d - 1) nbins in
  let core = L.slice reshaped start limit in
  let out =
    if density then begin
      let core = NL.astype core (dtype_of (as_inexact sample)) in
      let total = R.sum core in
      let scaled = U.divide core (L.broadcast_in_dim total (shape core) [||]) in
      List.fold_left
        (fun acc (i, e) ->
          let widths = NL.diff e in
          let sh = Array.make n_dims 1 in
          sh.(i) <- size_of widths;
          let expanded = NL.reshape widths sh in
          let broad =
            L.broadcast_in_dim expanded (shape acc) (Array.init n_dims Fun.id)
          in
          U.divide acc broad)
        scaled
        (List.mapi (fun i e -> (i, e)) edges)
    end
    else core
  in
  (out, edges)

let histogram2d ?(bins = [| 10; 10 |]) ?range ?weights ?(density = false) x y =
  let x, y = same_dtype x y in
  let sample = L.stack [ flat x; flat y ] 1 in
  match histogramdd ~bins ?range ?weights ~density sample with
  | hist, [ xe; ye ] -> (hist, xe, ye)
  | _ -> failwith "histogram2d: arity"

let interp ?left ?right ?period x xp fp =
  let x = as_inexact x and xp = as_inexact xp and fp = as_inexact fp in
  let x, xp =
    match period with
    | None -> (x, xp)
    | Some p ->
        let p = Float.abs p in
        let dt = dtype_of x in
        let m v = U.remainder v (L.full (shape v) p dt) in
        (m x, m xp)
  in
  let xp, fp =
    match period with
    | None -> (xp, fp)
    | Some p ->
        let p = Float.abs p in
        let order = Sorting.argsort ~axis:(Some 0) xp in
        let sx = I.take ~axis:0 xp order and sf = I.take ~axis:0 fp order in
        let n = size_of sx in
        let dt = dtype_of sx in
        let shift v d = U.add v (L.full (shape v) d dt) in
        ( NL.concatenate
            [
              shift (L.slice_in_dim sx (n - 1) n) (-.p);
              sx;
              shift (L.slice_in_dim sx 0 1) p;
            ],
          NL.concatenate
            [ L.slice_in_dim sf (n - 1) n; sf; L.slice_in_dim sf 0 1 ] )
  in
  let n = size_of xp in
  let i =
    NL.clip ~min:1.0
      ~max:(float_of_int (n - 1))
      (NL.searchsorted ~side:"right" xp x)
  in
  let one = L.full (shape i) 1.0 (dtype_of i) in
  let prev = U.subtract i one in
  let fp_i = I.take ~axis:0 fp i and fp_p = I.take ~axis:0 fp prev in
  let xp_i = I.take ~axis:0 xp i and xp_p = I.take ~axis:0 xp prev in
  let df = U.subtract fp_i fp_p in
  let dx = U.subtract xp_i xp_p in
  let delta = U.subtract x xp_p in
  let eps =
    match dtype_of xp with
    | Dtype.F64 -> 4.930380657631324e-32
    | _ -> 1.4210854715202004e-14
  in
  let dx0 =
    U.less_equal (U.absolute dx) (L.full (shape dx) eps (dtype_of dx))
  in
  let safe = NL.where_ dx0 (L.full (shape dx) 1.0 (dtype_of dx)) dx in
  let f =
    NL.where_ dx0 fp_p (U.add fp_p (U.multiply (U.divide delta safe) df))
  in
  match period with
  | Some _ -> f
  | None ->
      let first_x = L.index_in_dim ~keepdims:false xp 0 in
      let last_x = L.index_in_dim ~keepdims:false xp (n - 1) in
      let broad v = L.broadcast_in_dim v (shape x) [||] in
      let first_f = L.index_in_dim ~keepdims:false fp 0 in
      let last_f = L.index_in_dim ~keepdims:false fp (size_of fp - 1) in
      let fill_left =
        match left with
        | Some v -> L.full (shape x) v (dtype_of x)
        | None -> broad first_f
      in
      let fill_right =
        match right with
        | Some v -> L.full (shape x) v (dtype_of x)
        | None -> broad last_f
      in
      let f = NL.where_ (U.less x (broad first_x)) fill_left f in
      NL.where_ (U.greater x (broad last_x)) fill_right f

let gradient_along_axis a h axis =
  let n = (shape a).(axis) in
  if n < 2 then
    invalid_arg "Shape of array too small to calculate a numerical gradient";
  let cut lo hi = L.slice_in_dim ~axis a lo hi in
  let div_by d v = U.divide v (L.full (shape v) d (dtype_of v)) in
  let upper = div_by h (U.subtract (cut 1 2) (cut 0 1)) in
  let lower = div_by h (U.subtract (cut (n - 1) n) (cut (n - 2) (n - 1))) in
  if n = 2 then NL.concatenate ~axis [ upper; lower ]
  else
    let inner = div_by (2.0 *. h) (U.subtract (cut 2 n) (cut 0 (n - 2))) in
    NL.concatenate ~axis [ upper; inner; lower ]

let gradient ?(spacing = 1.0) ?axis a =
  let a = as_inexact a in
  let n = rank a in
  let axes =
    match axis with
    | Some ax -> List.map (canon n) ax
    | None -> List.init n Fun.id
  in
  List.map (fun ax -> gradient_along_axis a spacing ax) axes

let choose ?(mode = "raise") a choices =
  let idx =
    match mode with
    | "clip" | "raise" ->
        NL.clip ~min:0.0 ~max:(float_of_int (List.length choices - 1)) a
    | "wrap" ->
        U.remainder a
          (L.full (shape a) (float_of_int (List.length choices)) (dtype_of a))
    | other -> invalid_arg ("invalid mode=" ^ other)
  in
  let stacked = NL.stack ~axis:0 choices in
  let expanded = NL.expand_dims idx [| 0 |] in
  L.squeeze (I.take_along_axis ~axis:0 stacked expanded) [| 0 |]

let compress ?axis condition a =
  let cond = flat_floats (host "compress" condition) in
  let kept =
    List.filter
      (fun i -> cond.(i) <> 0.0)
      (List.init (Array.length cond) Fun.id)
  in
  let indices =
    ints_value (default_int ()) [| List.length kept |] (Array.of_list kept)
  in
  match axis with
  | Some ax -> I.take ~axis:ax a indices
  | None -> I.take ~axis:0 (flat a) indices

let extract condition a = compress (flat condition) (flat a)

let delete ?axis a obj =
  let a, ax =
    match axis with Some x -> (a, canon (rank a) x) | None -> (flat a, 0)
  in
  let n = (shape a).(ax) in
  let drop = List.map (canon n) obj in
  let kept =
    List.filter (fun i -> not (List.mem i drop)) (List.init n Fun.id)
  in
  I.take ~axis:ax a
    (ints_value (default_int ()) [| List.length kept |] (Array.of_list kept))

let insert ?axis a position values =
  let a, ax =
    match axis with Some x -> (a, canon (rank a) x) | None -> (flat a, 0)
  in
  let n = (shape a).(ax) in
  let position = if position < 0 then position + n else position in
  let target = Array.copy (shape a) in
  let count =
    if rank values = rank a then (shape values).(ax)
    else if rank values = 0 then 1
    else if rank values = 1 && rank a = 1 then size_of values
    else 1
  in
  target.(ax) <- count;
  let values = NL.astype values (dtype_of a) in
  let piece =
    if rank values = 0 then L.broadcast_in_dim values target [||]
    else if rank values = rank a then values
    else NL.broadcast_to values target
  in
  NL.concatenate ~axis:ax
    [
      L.slice_in_dim ~axis:ax a 0 position;
      piece;
      L.slice_in_dim ~axis:ax a position n;
    ]

let place a mask values =
  let mask_data = flat_floats (host "place" mask) in
  let hits =
    List.filter
      (fun i -> mask_data.(i) <> 0.0)
      (List.init (Array.length mask_data) Fun.id)
  in
  let vs = flat values in
  let n_v = size_of vs in
  let flat_a = flat a in
  let result =
    List.fold_left
      (fun acc (k, i) ->
        let v = L.index_in_dim ~keepdims:true vs (k mod n_v) in
        L.dynamic_update_slice acc (NL.astype v (dtype_of a)) [ int_scalar i ])
      flat_a
      (List.mapi (fun k i -> (k, i)) hits)
  in
  NL.reshape result (shape a)

let piecewise x condlist funclist =
  let n_cond = List.length condlist in
  let n_fun = List.length funclist in
  if n_fun <> n_cond && n_fun <> n_cond + 1 then
    invalid_arg "with N condition(s), either N or N+1 functions are expected";
  let default_fun =
    if n_fun = n_cond + 1 then List.nth funclist n_cond
    else fun v -> L.full_like v 0.0
  in
  let base = default_fun x in
  List.fold_left2
    (fun acc cond f -> NL.where_ cond (NL.astype (f x) (dtype_of acc)) acc)
    base condlist
    (List.filteri (fun i _ -> i < n_cond) funclist)

let fill_diagonal ?(wrap = false) a value =
  if wrap then invalid_arg "wrap=True is not implemented, must use wrap=False";
  let sh = shape a in
  if Array.length sh < 2 then invalid_arg "array must be at least 2-d";
  let is_2d = Array.length sh = 2 in
  if (not is_2d) && Array.exists (fun d -> d <> sh.(0)) sh then
    invalid_arg "All dimensions of input must be of equal length";
  let total = size_of a in
  let step =
    if is_2d then sh.(1) + 1 else Array.fold_left ( + ) 0 (strides_of sh)
  in
  let end_ = if is_2d then Stdlib.min total (sh.(1) * sh.(1)) else total in
  let vs = flat (NL.astype value (dtype_of a)) in
  let n_v = size_of vs in
  let flat_a = flat a in
  let rec loop acc i k =
    if i >= end_ then acc
    else
      let v = L.index_in_dim ~keepdims:true vs (k mod n_v) in
      loop (L.dynamic_update_slice acc v [ int_scalar i ]) (i + step) (k + 1)
  in
  NL.reshape (loop flat_a 0 0) sh

let index_pairs keep n m =
  let pairs = ref [] in
  for i = n - 1 downto 0 do
    for j = m - 1 downto 0 do
      if keep i j then pairs := (i, j) :: !pairs
    done
  done;
  let arr = Array.of_list !pairs in
  let dt = default_int () in
  let count = Array.length arr in
  ( ints_value dt [| count |] (Array.map fst arr),
    ints_value dt [| count |] (Array.map snd arr) )

let tril_indices ?(k = 0) ?m n =
  index_pairs (fun i j -> j - i <= k) n (Option.value m ~default:n)

let triu_indices ?(k = 0) ?m n =
  index_pairs (fun i j -> j - i >= k) n (Option.value m ~default:n)

let tril_indices_from ?k a =
  let sh = shape a in
  if Array.length sh <> 2 then invalid_arg "input array must be 2-d";
  tril_indices ?k ~m:sh.(1) sh.(0)

let triu_indices_from ?k a =
  let sh = shape a in
  if Array.length sh <> 2 then invalid_arg "input array must be 2-d";
  triu_indices ?k ~m:sh.(1) sh.(0)

let mask_indices ?(k = 0) n mask_func =
  let ones = AC.ones ~dtype:(default_int ()) [| n; n |] in
  match nonzero (mask_func ones k) with
  | [ a; b ] -> (a, b)
  | _ -> failwith "mask_indices: arity"

let trim_zeros ?(trim = "fb") a =
  let data = flat_floats (host "trim_zeros" a) in
  let n = Array.length data in
  let first = ref 0 and last = ref n in
  if String.contains trim 'f' then
    while !first < n && data.(!first) = 0.0 do
      incr first
    done;
  if String.contains trim 'b' then
    while !last > !first && data.(!last - 1) = 0.0 do
      decr last
    done;
  L.slice_in_dim a !first !last

let sort_complex a =
  let nd = host "sort_complex" a in
  let dt =
    match Ndarray.dtype nd with
    | Dtype.Complex128 | Dtype.F64 -> Dtype.Complex128
    | _ -> Dtype.Complex64
  in
  let sh = Ndarray.shape nd in
  if Array.length sh <> 1 then
    invalid_arg "sort_complex supports one dimensional input";
  let values =
    Array.init sh.(0) (fun i ->
        match Ndarray.dtype nd with
        | Dtype.Complex64 | Dtype.Complex128 -> Ndarray.get_c nd [| i |]
        | _ -> { Complex.re = Ndarray.get_f nd [| i |]; im = 0.0 })
  in
  Array.stable_sort
    (fun (x : Complex.t) (y : Complex.t) ->
      let c = compare x.re y.re in
      if c <> 0 then c else compare x.im y.im)
    values;
  Concrete (Ndarray.of_complex dt sh values)

let bitwise_count a = NL.astype (L.population_count (U.absolute a)) (dtype_of a)

let frexp x =
  let x = as_inexact x in
  let nd = host "frexp" x in
  let data = flat_floats nd in
  let mantissa = Array.map (fun v -> fst (Float.frexp v)) data in
  let exponent =
    Array.map
      (fun v -> if Float.is_finite v then snd (Float.frexp v) else 0)
      data
  in
  ( Concrete (Ndarray.of_floats (dtype_of x) (shape x) mantissa),
    ints_value Dtype.I32 (shape x) exponent )

let ldexp x exponent =
  let x = as_inexact x in
  let two = L.full (shape x) 2.0 (dtype_of x) in
  U.multiply x (U.power two (NL.astype exponent (dtype_of x)))

let argpartition ?(axis = -1) a kth =
  ignore kth;
  Sorting.argsort ~axis:(Some axis) a

let apply_along_axis f axis arr =
  let n = rank arr in
  let axis = canon n axis in
  let moved = NL.moveaxis [| axis |] [| n - 1 |] arr in
  let rec wrap depth g =
    if depth = 0 then g
    else
      wrap (depth - 1) (fun x ->
          List.hd
            (Interpreters.Batching.vmap
               (fun xs -> [ g (List.hd xs) ])
               [ Some 0 ] [ x ]))
  in
  let result = (wrap (n - 1) f) moved in
  let rn = rank result in
  let produced = rn - (n - 1) in
  if produced = 0 then result
  else
    NL.moveaxis
      (Array.init produced (fun i -> n - 1 + i))
      (Array.init produced (fun i -> axis + i))
      result

let apply_over_axes f a axes =
  List.fold_left
    (fun acc ax ->
      let ax = canon (rank acc) ax in
      let out = f acc ax in
      if rank out = rank acc then out else NL.expand_dims out [| ax |])
    a axes

type nested = Item of value | Group of nested list

let block tree =
  let rec depth = function
    | Item _ -> 0
    | Group [] -> 1
    | Group (h :: _) -> 1 + depth h
  in
  let d = depth tree in
  let rec go level = function
    | Item v -> v
    | Group items ->
        let parts = List.map (go (level + 1)) items in
        let target =
          List.fold_left (fun m v -> Stdlib.max m (rank v)) (d - level) parts
        in
        let lifted = List.map (fun v -> L.broadcast_to_rank v target) parts in
        NL.concatenate ~axis:(level - d) lifted
  in
  go 0 tree

let promote_types a b = Dtypes.promote_types a b
let can_cast a b = Dtypes.promote_types a b = b
let shape_of = shape
let size = size_of
let ndim = rank

let polydiv u v =
  let u = as_inexact u and v = as_inexact v in
  let lu = size_of u and lv = size_of v in
  if lu < lv then (AC.zeros ~dtype:(dtype_of u) [| 1 |], u)
  else
    let dt = dtype_of u in
    let udata = flat_floats (host "polydiv" u) in
    let vdata = flat_floats (host "polydiv" v) in
    let q = Array.make (lu - lv + 1) 0.0 in
    let r = Array.copy udata in
    for k = 0 to lu - lv do
      q.(k) <- r.(k) /. vdata.(0);
      for j = 0 to lv - 1 do
        r.(k + j) <- r.(k + j) -. (q.(k) *. vdata.(j))
      done
    done;
    let tolerance =
      match dt with
      | Dtype.F64 -> 1.4901161193847656e-08
      | _ -> 0.00034526698300124393
    in
    let cleaned =
      Array.map (fun x -> if Float.abs x < tolerance then 0.0 else x) r
    in
    ( Concrete (Ndarray.of_floats dt [| Array.length q |] q),
      Concrete (Ndarray.of_floats dt [| lu |] cleaned) )

let polyfit ?rcond x y deg =
  let x = as_inexact x and y = as_inexact y in
  let order = deg + 1 in
  let lhs = NL.vander ~n:order x in
  let eps =
    match dtype_of x with
    | Dtype.F64 -> 2.220446049250313e-16
    | _ -> 1.1920929e-07
  in
  let rcond =
    match rcond with Some r -> r | None -> float_of_int (size_of x) *. eps
  in
  let scale = U.sqrt (R.sum ~axis:[| 0 |] (U.multiply lhs lhs)) in
  let broad = L.broadcast_in_dim scale (shape lhs) [| 1 |] in
  let scaled = U.divide lhs broad in
  let c, _, _, _ = Linalg.lstsq ~rcond scaled y in
  let scale_c =
    if rank c = 2 then L.broadcast_in_dim scale (shape c) [| 0 |] else scale
  in
  U.divide c scale_c

let roots ?(strip_zeros = true) p =
  let data = flat_floats (host "roots" p) in
  let n = Array.length data in
  let lead = ref 0 in
  if strip_zeros then
    while !lead < n && data.(!lead) = 0.0 do
      incr lead
    done;
  let trail = ref n in
  if strip_zeros then
    while !trail > !lead && data.(!trail - 1) = 0.0 do
      decr trail
    done;
  let core = Array.sub data !lead (!trail - !lead) in
  let degree = Array.length core - 1 in
  let zero_roots = n - !trail in
  let real_dt = dtype_of (as_inexact p) in
  let complex_dt =
    match real_dt with Dtype.F64 -> Dtype.Complex128 | _ -> Dtype.Complex64
  in
  let zeros =
    Concrete
      (Ndarray.of_complex complex_dt [| zero_roots |]
         (Array.make zero_roots Complex.zero))
  in
  if degree < 1 then zeros
  else
    let companion =
      Array.init (degree * degree) (fun i ->
          let r = i / degree and c = i mod degree in
          if r = 0 then -.core.(c + 1) /. core.(0)
          else if r = c + 1 then 1.0
          else 0.0)
    in
    let values =
      Linalg.eigvals
        (Concrete (Ndarray.of_floats real_dt [| degree; degree |] companion))
    in
    if zero_roots = 0 then values else NL.concatenate [ values; zeros ]

let fromfunction f shape_ ~dtype =
  let grid =
    List.init (Array.length shape_) (fun axis ->
        L.broadcasted_iota dtype shape_ axis)
  in
  f grid

let pi = 3.141592653589793
let e = 2.718281828459045
let euler_gamma = 0.5772156649015329
let inf = Float.infinity
let nan = Float.nan

let rec render_elements nd idx dims =
  match dims with
  | [] ->
      let v = Ndarray.get_f nd (Array.of_list (List.rev idx)) in
      Printf.sprintf "%g" v
  | d :: rest ->
      "["
      ^ String.concat " "
          (List.init d (fun i -> render_elements nd (i :: idx) rest))
      ^ "]"

let array_str a =
  let nd = host "array_str" a in
  render_elements nd [] (Array.to_list (Ndarray.shape nd))

let array_repr a =
  "Array(" ^ array_str a ^ ", dtype=" ^ Dtype.short_name (dtype_of a) ^ ")"
