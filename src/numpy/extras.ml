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

let traced_shape_repr dt sh =
  Printf.sprintf "traced array with shape %s[%s]" (Dtype.numpy_name dt)
    (String.concat "," (Array.to_list (Array.map string_of_int sh)))

let unique_context =
  "The error arose for the first argument of jnp.unique(). To make \
   jnp.unique() compatible with JIT and other transforms, you can specify a \
   concrete value for the size argument, which will determine the output size."

let concretization name v =
  let aval = Core.get_aval v in
  let context dt sh text =
    Errors.concretization_type_error ~context:text
      ~error_repr:(traced_shape_repr dt sh) ~origin_msg:(Jaxpr.origin_msg v) ()
  in
  let int_dt = Dtypes.default_int_dtype () in
  let own text = context aval.dtype aval.shape text in
  match name with
  | "nonzero" ->
      context int_dt [||]
        "The size argument of jnp.nonzero must be statically specified to use \
         jnp.nonzero within JAX transformations."
  | "roots" ->
      context int_dt [||]
        "The error occurred in the jnp.roots() function. To use this within a \
         JIT-compiled context, pass strip_zeros=False, but be aware that \
         leading zeros will result in some returned roots being set to NaN."
  | "unique" | "intersect1d" | "setxor1d" -> own unique_context
  | "union1d" -> own "The error arose in union1d()"
  | "setdiff1d" -> own "The error arose in setdiff1d()"
  | "trim_zeros" -> own "Error arose in the `filt` argument of trim_zeros()"
  | "delete" -> own "'obj' array argument of jnp.delete()"
  | "bincount" ->
      own
        "The error occurred because of argument 'x' of jnp.bincount. To avoid \
         this error, pass a static `length` argument."
  | "compress" ->
      Errors.tracer_bool_conversion_error
        ~error_repr:(traced_shape_repr aval.dtype aval.shape)
        ~origin_msg:(Jaxpr.origin_msg v) ()
  | _ ->
      own
        (name
       ^ " requires concrete input; pass a size argument to use it under jit")

let rec host name v =
  match v with
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer { payload = JVP { primal; _ }; _ } -> host name primal
  | Tracer _ -> raise (concretization name v)

let flat_floats nd =
  Array.of_list (List.rev (Ndarray.fold (fun acc x -> x :: acc) [] nd))

let ints_value dt shape_ xs =
  Concrete (Ndarray.of_floats dt shape_ (Array.map float_of_int xs))

let int_scalar i = ints_value (default_int ()) [||] [| i |]
let scalar_like v x = L.full [||] x (dtype_of v)
let flat v = if rank v = 1 then v else NL.reshape v [| size_of v |]
let shape_int a = Array.fold_left ( * ) 1 a

let is_float_dtype = function
  | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let as_inexact v =
  let dt = dtype_of v in
  let target = Dtypes.to_inexact_dtype dt in
  if target = dt then v else NL.astype v target

let nonzero_mask a =
  match dtype_of a with
  | Dtype.Bool -> a
  | _ -> U.not_equal a (scalar_like a 0.0)

let is_complex_value v =
  match dtype_of v with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let real_of_complex v =
  match dtype_of v with
  | Dtype.Complex128 -> Dtype.F64
  | Dtype.Complex64 -> Dtype.F32
  | d -> d

let strides_of sh =
  let n = Array.length sh in
  let s = Array.make n 1 in
  for d = n - 2 downto 0 do
    s.(d) <- s.(d + 1) * sh.(d + 1)
  done;
  s

let scatter_dims_1d =
  {
    update_window_dims = [||];
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
  }

let normalised_indices ~extent indices =
  let dt = dtype_of indices in
  let negative = U.less indices (L.full [||] 0.0 dt) in
  let shifted = U.add indices (L.full [||] (float_of_int extent) dt) in
  let normalised = Core.bind1 Select_n [ negative; indices; shifted ] in
  L.broadcast_in_dim normalised [| size_of indices; 1 |] [| 0 |]

let bincount_length length weights x =
  let wdt = dtype_of weights in
  let operand = L.full [| length |] 0.0 wdt in
  let clipped = NL.clip ~min:0.0 x in
  let wrapped = normalised_indices ~extent:length clipped in
  let updates =
    if rank weights = 0 then L.broadcast_in_dim weights [| size_of x |] [||]
    else weights
  in
  L.scatter_add operand wrapped updates scatter_dims_1d

let nonzero ?size ?fill_value a =
  let sh = shape a in
  let n = Array.length sh in
  let mask = nonzero_mask a in
  let count () = R.sum mask in
  let k =
    match size with
    | Some k -> k
    | None ->
        let total = count () in
        (match total with
        | Tracer _ -> raise (concretization "nonzero" a)
        | _ -> ());
        int_of_float (Ndarray.get_f (host "nonzero" total) [||])
  in
  let int_dt = default_int () in
  if k = 0 then
    List.map (fun _ -> AC.zeros ~dtype:int_dt [| 0 |]) (Array.to_list sh)
  else begin
    let running = R.cumsum mask in
    let histogram =
      bincount_length k
        (Concrete (Ndarray.of_floats (default_int ()) [||] [| 1.0 |]))
        running
    in
    let flat_indices = R.cumsum histogram in
    let fdt = dtype_of flat_indices in
    let strides = strides_of sh in
    let out =
      List.init n (fun d ->
          let quotient =
            U.floor_divide flat_indices
              (Concrete
                 (Ndarray.of_floats fdt [||] [| float_of_int strides.(d) |]))
          in
          U.remainder quotient
            (Concrete (Ndarray.of_floats fdt [||] [| float_of_int sh.(d) |])))
    in
    match fill_value with
    | None -> out
    | Some fill ->
        let idx = NL.arange ~dtype:(default_int ()) (float_of_int k) in
        let fill_mask = U.greater_equal idx (count ()) in
        List.map
          (fun entry ->
            NL.where_ fill_mask
              (Concrete
                 (Ndarray.of_floats (dtype_of entry) [||]
                    [| float_of_int fill |]))
              entry)
          out
  end

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

let traced_unique_sized ?axis ~size ?fill_value a =
  let module S = Sorting in
  let axis =
    match axis with None -> None | Some x -> Some (canon (rank a) x)
  in
  let rows, aux_source =
    match axis with
    | None ->
        let flat_a = NL.reshape a [| size_of a |] in
        (size_of a, flat_a)
    | Some ax ->
        let moved = NL.moveaxis [| ax |] [| 0 |] a in
        ((shape moved).(0), moved)
  in
  let row_rest =
    Array.sub (shape aux_source) 1 (Array.length (shape aux_source) - 1)
  in
  let row_len = Array.fold_left ( * ) 1 row_rest in
  let flat_rows = NL.reshape aux_source [| rows; row_len |] in
  let keys =
    List.init row_len (fun j ->
        L.index_in_dim ~axis:1 ~keepdims:false flat_rows j)
  in
  let perm =
    match keys with
    | [ only ] -> S.argsort only
    | _ -> S.lexsort (List.rev keys)
  in
  let aux = I.take ~axis:0 aux_source perm in
  let aux_flat = NL.reshape aux [| rows; row_len |] in
  let tail = L.slice_in_dim ~axis:0 aux_flat 1 rows in
  let head = L.slice_in_dim ~axis:0 aux_flat 0 (rows - 1) in
  let differs =
    let neq = L.ne tail head in
    let dt = dtype_of a in
    if Dtypes.isdtype dt "numeric" && not (Dtypes.isdtype dt "integral") then
      L.bitwise_and neq
        (L.bitwise_not (L.bitwise_and (L.ne tail tail) (L.ne head head)))
    else neq
  in
  let any_diff = R.any ~axis:[| 1 |] differs in
  let mask =
    NL.concatenate ~axis:0
      [ Concrete (Ndarray.of_floats Dtype.Bool [| 1 |] [| 1.0 |]); any_diff ]
  in
  let ind = List.hd (nonzero ~size mask) in
  let result = I.take ~axis:0 aux ind in
  let idt = dtype_of ind in
  let distinct = R.sum (NL.astype mask idt) in
  let result =
    match fill_value with
    | None -> result
    | Some fv ->
        let positions = NL.arange ~dtype:idt (float_of_int size) in
        let valid = L.lt positions (NL.broadcast_to distinct [| size |]) in
        let wide =
          NL.reshape valid
            (Array.append [| size |] (Array.map (fun _ -> 1) row_rest))
        in
        NL.where_ wide result (L.full_like result fv)
  in
  let values =
    match axis with
    | None -> result
    | Some ax -> NL.moveaxis [| 0 |] [| ax |] result
  in
  let indices = I.take ~axis:0 perm ind in
  let inverse =
    let imask =
      L.sub (L.cumsum ~axis:0 (NL.astype mask idt)) (L.full_like distinct 1.0)
    in
    I.take ~axis:0 imask (S.argsort perm)
  in
  let counts =
    let edges = List.hd (nonzero ~size:(size + 1) mask) in
    let upper = L.slice_in_dim ~axis:0 edges 1 (size + 1) in
    let stops =
      NL.where_
        (L.ne upper (L.full_like upper 0.0))
        upper
        (L.full_like upper (float_of_int rows))
    in
    let starts =
      NL.concatenate ~axis:0
        [
          L.slice_in_dim ~axis:0 edges 0 1;
          L.slice_in_dim ~axis:0 stops 0 (size - 1);
        ]
    in
    L.sub stops starts
  in
  ({ values; indices; inverse; counts }, distinct)

let traced_unique ?axis ~size ?fill_value a =
  fst (traced_unique_sized ?axis ~size ?fill_value a)

let rec unique_full ?axis ?size ?fill_value a =
  match size with
  | _ when is_complex_value a && axis = None ->
      unique_complex ?size ?fill_value a
  | Some size when size_of a > 0 -> traced_unique ?axis ~size ?fill_value a
  | _ ->
      let fill_value = Option.value fill_value ~default:0.0 in
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

and unique_complex ?size ?fill_value a =
  let n = size_of a in
  let source = flat a in
  let parts =
    NL.stack ~axis:1
      [
        NL.astype (U.real source) (real_of_complex a);
        NL.astype (U.imag source) (real_of_complex a);
      ]
  in
  let r = unique_full ~axis:0 ?size parts in
  ignore fill_value;
  let rows = (shape r.values).(0) in
  let column j = L.index_in_dim ~axis:1 ~keepdims:false r.values j in
  let values = NL.astype (L.complex (column 0) (column 1)) (dtype_of a) in
  ignore (n, rows);
  { r with values; inverse = NL.reshape r.inverse (shape a) }

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

let unique_values_full ?size ?fill_value a =
  (unique_full ?size ?fill_value a).values

let of_float_list dt xs =
  let arr = Array.of_list xs in
  Concrete (Ndarray.of_floats dt [| Array.length arr |] arr)

let sized_result dt values ~size ~fill_value =
  match size with
  | None -> of_float_list dt values
  | Some n ->
      let count = List.length values in
      let kept = List.filteri (fun i _ -> i < n) values in
      let fill =
        match (fill_value, kept) with
        | Some f, _ -> f
        | None, [] -> 0.0
        | None, _ -> List.fold_left Float.min Float.infinity kept
      in
      of_float_list dt (kept @ List.init (max 0 (n - count)) (fun _ -> fill))

let lit_of v dt = Concrete (Ndarray.of_floats dt [||] [| v |])

let positions_below n bound =
  let idt = default_int () in
  U.less (NL.arange ~dtype:idt (float_of_int n)) bound

let fill_or_extremes ~fill_value vals valid_count =
  let n = size_of vals in
  let dt = dtype_of vals in
  match fill_value with
  | Some f -> NL.where_ (positions_below n valid_count) vals (lit_of f dt)
  | None ->
      let high = NL.where_ (positions_below n valid_count) vals (R.max vals) in
      NL.where_ (positions_below n valid_count) high (R.min high)

let union1d ?size ?fill_value a b =
  let a, b = same_dtype a b in
  match size with
  | Some n ->
      unique_values_full ~size:n ?fill_value (NL.concatenate [ flat a; flat b ])
  | None ->
      let first = sorted_unique_floats "union1d" a in
      let second = sorted_unique_floats "union1d" b in
      let xs = List.sort_uniq compare_with_nan (first @ second) in
      sized_result (dtype_of a) xs ~size ~fill_value

let flat_float_list name v = Array.to_list (flat_floats (host name v))

let concat_sorted_keys ~keys ~payload =
  Core.bind
    (Sort { dimension = 0; is_stable = true; num_keys = List.length keys })
    (keys @ payload)

let dynamic_set arr piece start = L.dynamic_update_slice arr piece [ start ]

let concat_unique arr1 arr2 ~fill_value =
  let u1, n1 =
    traced_unique_sized ~axis:0 ~size:(size_of arr1) ?fill_value arr1
  in
  let u2, n2 =
    traced_unique_sized ~axis:0 ~size:(size_of arr2) ?fill_value arr2
  in
  let dt = dtype_of u1.values in
  let arr = L.full [| size_of arr1 + size_of arr2 |] 0.0 dt in
  let arr = dynamic_set arr u1.values (lit_of 0.0 (default_int ())) in
  let arr = dynamic_set arr u2.values n1 in
  (arr, U.add n1 n2)

let take_fill aux indices =
  I.take ~mode:"fill" ~fill_value:0.0 ~axis:0 aux indices

let intersect1d_sized ~assume_unique ~size ~fill_value a b =
  let idt = default_int () in
  let aux, mask =
    if assume_unique then begin
      let arr = NL.concatenate [ a; b ] in
      let n = size_of arr in
      match
        concat_sorted_keys ~keys:[ arr ]
          ~payload:[ NL.arange ~dtype:idt (float_of_int n) ]
      with
      | aux :: _ -> (aux, AC.ones ~dtype:Dtype.Bool [| n |])
      | [] -> invalid_arg "intersect1d: arity"
    end
    else begin
      let arr, total = concat_unique a b ~fill_value:(Some 0.0) in
      let n = size_of arr in
      let mask = positions_below n total in
      match
        concat_sorted_keys
          ~keys:[ U.logical_not mask; arr ]
          ~payload:[ NL.arange ~dtype:idt (float_of_int n) ]
      with
      | [ _; aux; _ ] -> (aux, mask)
      | _ -> invalid_arg "intersect1d: arity"
    end
  in
  let n = size_of aux in
  let tail v = L.slice v [| 1 |] [| n |]
  and head v = L.slice v [| 0 |] [| n - 1 |] in
  let aux_mask =
    U.logical_and (Core.bind1 Eq [ tail aux; head aux ]) (tail mask)
  in
  let num_results = R.sum aux_mask in
  let vals =
    if n > 0 then take_fill aux (List.hd (nonzero ~size ~fill_value:n aux_mask))
    else L.full [| size |] 0.0 (dtype_of aux)
  in
  fill_or_extremes ~fill_value vals num_results

let intersect1d ?(assume_unique = false) ?size ?fill_value a b =
  let a, b = same_dtype a b in
  let a = flat a and b = flat b in
  match size with
  | Some n -> intersect1d_sized ~assume_unique ~size:n ~fill_value a b
  | None ->
      let xs =
        if assume_unique then
          let first = flat_float_list "intersect1d" a in
          let second = flat_float_list "intersect1d" b in
          let sorted = List.sort compare_with_nan (first @ second) in
          let rec adjacent = function
            | p :: (q :: _ as rest) ->
                if compare_with_nan p q = 0 then p :: adjacent rest
                else adjacent rest
            | _ -> []
          in
          adjacent sorted
        else
          let first = sorted_unique_floats "intersect1d" a in
          let ys = sorted_unique_floats "intersect1d" b in
          List.filter
            (fun x -> List.exists (fun y -> compare_with_nan x y = 0) ys)
            first
      in
      sized_result (dtype_of a) xs ~size ~fill_value

let setdiff1d ?(assume_unique = false) ?size ?fill_value a b =
  let a, b = same_dtype a b in
  let dt = dtype_of a in
  let a = flat a and b = flat b in
  let fill = Option.value fill_value ~default:0.0 in
  match size with
  | Some n when size_of a = 0 -> L.full [| n |] fill dt
  | Some n ->
      let m = size_of a in
      let arr =
        if assume_unique then a
        else unique_values_full ~size:(if n = 0 then 0 else m) a
      in
      let mask = Setops.in1d ~invert:true arr b in
      let mask =
        if assume_unique then mask
        else
          let first = L.slice arr [| 0 |] [| 1 |] in
          let same =
            R.sum (Core.bind1 Eq [ arr; NL.broadcast_to first [| m |] ])
          in
          let n_unique =
            U.subtract (lit_of (float_of_int (m + 1)) (default_int ())) same
          in
          NL.where_
            (positions_below m n_unique)
            mask
            (Concrete (Ndarray.of_floats Dtype.Bool [||] [| 0.0 |]))
      in
      let picked =
        I.take ~mode:"clip" ~axis:0 arr (List.hd (nonzero ~size:n mask))
      in
      NL.where_ (positions_below n (R.sum mask)) picked (lit_of fill dt)
  | None ->
      let source =
        if assume_unique then flat_float_list "setdiff1d" a
        else sorted_unique_floats "setdiff1d" a
      in
      let ys = flat_float_list "setdiff1d" b in
      sized_result dt
        (List.filter
           (fun x -> not (List.exists (fun y -> compare_with_nan x y = 0) ys))
           source)
        ~size ~fill_value:(Some fill)

let setxor1d_sized ~assume_unique ~size ~fill_value a b =
  let bool_lit v = Concrete (Ndarray.of_floats Dtype.Bool [| 1 |] [| v |]) in
  let aux, flag =
    if assume_unique then begin
      let arr = NL.concatenate [ a; b ] in
      let aux = Sorting.sort ~axis:(Some 0) arr in
      let n = size_of aux in
      let tail = L.slice aux [| 1 |] [| n |]
      and head = L.slice aux [| 0 |] [| n - 1 |] in
      ( aux,
        NL.concatenate
          [
            bool_lit (if n > 0 then 1.0 else 0.0);
            Core.bind1 Ne [ tail; head ];
            bool_lit 1.0;
          ] )
    end
    else begin
      let arr, num_unique = concat_unique a b ~fill_value:None in
      let n = size_of arr in
      let idt = default_int () in
      let mask = positions_below (n + 1) (U.add num_unique (lit_of 1.0 idt)) in
      let aux =
        match
          concat_sorted_keys
            ~keys:[ U.logical_not (L.slice mask [| 1 |] [| n + 1 |]); arr ]
            ~payload:[]
        with
        | [ _; aux ] -> aux
        | _ -> invalid_arg "setxor1d: arity"
      in
      let tail = L.slice aux [| 1 |] [| n |]
      and head = L.slice aux [| 0 |] [| n - 1 |] in
      let body =
        NL.concatenate
          [
            bool_lit (if n > 0 then 1.0 else 0.0);
            Core.bind1 Ne [ tail; head ];
            bool_lit 0.0;
          ]
      in
      let at_end =
        Core.bind1 Eq
          [ NL.arange ~dtype:idt (float_of_int (n + 1)); num_unique ]
      in
      (aux, U.logical_and mask (U.logical_or body at_end))
    end
  in
  let m = size_of flag in
  let aux_mask =
    U.logical_and
      (L.slice flag [| 1 |] [| m |])
      (L.slice flag [| 0 |] [| m - 1 |])
  in
  let num_results = R.sum aux_mask in
  let n = size_of aux in
  let vals =
    if n > 0 then take_fill aux (List.hd (nonzero ~size ~fill_value:n aux_mask))
    else L.full [| size |] 0.0 (dtype_of aux)
  in
  fill_or_extremes ~fill_value vals num_results

let setxor1d ?(assume_unique = false) ?size ?fill_value a b =
  let a, b = same_dtype a b in
  let a = flat a and b = flat b in
  match size with
  | Some n -> setxor1d_sized ~assume_unique ~size:n ~fill_value a b
  | None ->
      let values v =
        if assume_unique then flat_float_list "setxor1d" v
        else sorted_unique_floats "setxor1d" v
      in
      let first = values a in
      let second = values b in
      let sorted = List.sort compare_with_nan (first @ second) in
      let rec singles = function
        | p :: q :: rest when compare_with_nan p q = 0 ->
            let rec drop = function
              | r :: tl when compare_with_nan p r = 0 -> drop tl
              | tl -> tl
            in
            singles (drop rest)
        | p :: rest -> p :: singles rest
        | [] -> []
      in
      sized_result (dtype_of a) (singles sorted) ~size ~fill_value

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

let bincount ?weights ?(minlength = 0) ?length x =
  let x =
    if dtype_of x = Dtype.Bool then L.convert_element_type x Dtype.I32 else x
  in
  (match dtype_of x with
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint32 | Dtype.Uint64 ->
      ()
  | d ->
      raise
        (Errors.jax_type_error
           ("x argument to bincount must have an integer type; got "
          ^ Dtype.short_name d)));
  let n =
    match length with
    | Some l -> l
    | None ->
        let data = flat_floats (host "bincount" x) in
        Stdlib.max minlength
          (if Array.length data = 0 then 0
           else 1 + int_of_float (Array.fold_left Float.max 0.0 data))
  in
  match weights with
  | None ->
      if n = 0 then AC.zeros ~dtype:(default_int ()) [| 0 |]
      else
        bincount_length n
          (Concrete (Ndarray.of_floats (default_int ()) [||] [| 1.0 |]))
          (flat x)
  | Some w ->
      if n = 0 then AC.zeros ~dtype:(dtype_of w) [| 0 |]
      else bincount_length n w (flat x)

let histogram_bin_edges ?(bins = 10) ?range a =
  let a = as_inexact (flat a) in
  let dt = dtype_of a in
  let lo, hi =
    match range with
    | Some (l, h) -> (L.full [||] l dt, L.full [||] h dt)
    | None ->
        let low = R.min a in
        let high = R.max a in
        (low, high)
  in
  let lifted v = L.broadcast_in_dim v [| 1 |] [||] in
  let pair = NL.concatenate [ lifted lo; lifted hi ] in
  let spread () =
    Jnp_util.jit1 "_ptp"
      (function [ v ] -> R.ptp v | _ -> invalid_arg "ptp: arity")
      [ pair ]
  in
  let half = L.full [||] 0.5 dt in
  let zero = L.full [||] 0.0 dt in
  let same = Core.bind1 Types.Eq [ spread (); zero ] in
  let first = L.index_in_dim ~keepdims:false pair 0 in
  let lo' = NL.where_ same (Core.bind1 Types.Sub [ first; half ]) first in
  let same = Core.bind1 Types.Eq [ spread (); zero ] in
  let second = L.index_in_dim ~keepdims:false pair 1 in
  let hi' = NL.where_ same (Core.bind1 Types.Add [ second; half ]) second in
  AC.linspace_v ~num:(bins + 1) ~dtype:dt lo' hi'

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

let compress ?axis ?size ?(fill_value = 0.0) condition a =
  let condition = Jaxpr.stage_constant condition in
  let cond =
    if dtype_of condition = Dtype.Bool then condition
    else L.convert_element_type condition Dtype.Bool
  in
  if rank cond <> 1 then invalid_arg "condition must be a 1D array";
  let arr, ax =
    match axis with
    | None -> (flat a, 0)
    | Some x ->
        let x = canon (rank a) x in
        (NL.moveaxis [| x |] [| 0 |] a, x)
  in
  let extent = (shape arr).(0) in
  let length = (shape cond).(0) in
  let usable = min extent length in
  let piece v start stop =
    if start = 0 && stop = (shape v).(0) then v
    else
      let hi = Array.copy (shape v) in
      hi.(0) <- stop;
      let lo = Array.make (rank v) 0 in
      lo.(0) <- start;
      L.slice v lo hi
  in
  let extra = piece cond usable length in
  let cond = piece cond 0 usable in
  let arr = piece arr 0 usable in
  let finish result = NL.moveaxis [| 0 |] [| ax |] result in
  match size with
  | None ->
      let any_extra = R.any extra in
      (match any_extra with
      | Tracer _ -> raise (concretization "compress" any_extra)
      | _ ->
          if Ndarray.get_f (host "compress" any_extra) [||] <> 0.0 then
            invalid_arg "condition contains entries that are out of bounds");
      let bits = flat_floats (host "compress" cond) in
      let kept =
        List.filter
          (fun i -> bits.(i) <> 0.0)
          (List.init (Array.length bits) Fun.id)
      in
      finish
        (I.take ~mode:"clip" ~axis:0 arr
           (ints_value (default_int ())
              [| List.length kept |]
              (Array.of_list kept)))
  | Some n ->
      if n < 0 || n > (shape arr).(0) then
        invalid_arg
          (Printf.sprintf
             "size must be positive and not greater than the size of the array \
              axis; got size=%d for a.shape[axis]=%d"
             n
             (shape arr).(0));
      let mask =
        NL.expand_dims cond (Array.init (rank arr - 1) (fun i -> i + 1))
      in
      let arr =
        NL.where_ mask arr
          (Concrete (Ndarray.of_floats (dtype_of arr) [||] [| fill_value |]))
      in
      let order =
        Sorting.argsort ~axis:(Some 0) ~stable:true ~descending:true cond
      in
      let picked = I.take ~mode:"fill" ~axis:0 arr order in
      finish (piece picked 0 n)

let extract ?size ?fill_value condition a =
  compress ?size ?fill_value (flat condition) (flat a)

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

let delete_array ?axis a obj =
  let a, axis =
    match axis with
    | Some x -> (a, Some (canon (rank a) x))
    | None -> (flat a, Some 0)
  in
  let values = flat_floats (host "delete" obj) in
  delete ?axis a (Array.to_list (Array.map int_of_float values))

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
  let vals = flat values in
  if size_of a <> size_of mask then
    invalid_arg "place: arr and mask must be the same size";
  if size_of vals = 0 then invalid_arg "Cannot place values from an empty array";
  if size_of a = 0 then a
  else begin
    let n = size_of mask in
    let indices = List.hd (nonzero ~size:n ~fill_value:n (flat mask)) in
    let vals =
      if size_of vals = n then vals
      else if size_of vals > n then L.slice vals [| 0 |] [| n |]
      else
        let repeats =
          int_of_float
            (Float.ceil (float_of_int n /. float_of_int (size_of vals)))
        in
        let tiled = NL.tile vals [| repeats |] in
        if size_of tiled > n then L.slice tiled [| 0 |] [| n |] else tiled
    in
    let vals =
      if dtype_of vals = dtype_of a then vals else NL.astype vals (dtype_of a)
    in
    let wrapped = normalised_indices ~extent:n indices in
    let updated = L.scatter (flat a) wrapped vals scatter_dims_1d in
    NL.reshape updated (shape a)
  end

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

let mask_indices ?(k = 0) ?size n mask_func =
  let ones = AC.ones ~dtype:(default_int ()) [| n; n |] in
  match nonzero ?size (mask_func ones k) with
  | [ a; b ] -> (a, b)
  | _ -> failwith "mask_indices: arity"

let trim_zeros ?(trim = "fb") ?axis a =
  let lowered = String.lowercase_ascii trim in
  let front = String.contains lowered 'f'
  and back = String.contains lowered 'b' in
  let axes =
    match axis with
    | None -> List.init (rank a) Fun.id
    | Some ax -> List.map (canon (rank a)) ax
  in
  if axes = [] || not (front || back) then a
  else
    List.fold_left
      (fun acc ax ->
        let others =
          Array.of_list
            (List.filter (fun i -> i <> ax) (List.init (rank a) Fun.id))
        in
        (match a with
        | Tracer _ -> raise (concretization "trim_zeros" a)
        | _ -> ());
        let mask = flat_floats (host "trim_zeros" (R.any ~axis:others a)) in
        let n = Array.length mask in
        let first = ref 0 and last = ref n in
        while !first < n && mask.(!first) = 0.0 do
          incr first
        done;
        if !first = n then L.slice_in_dim ~axis:ax acc 0 0
        else begin
          while !last > 0 && mask.(!last - 1) = 0.0 do
            decr last
          done;
          let start = if front then !first else 0 in
          let stop = if back then !last else n in
          L.slice_in_dim ~axis:ax acc start stop
        end)
      a axes

let sort_complex a =
  let nd = host "sort_complex" a in
  let dt =
    match Ndarray.dtype nd with
    | Dtype.Complex128 | Dtype.F64 | Dtype.I64 | Dtype.Uint64 ->
        Dtype.Complex128
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
      let order p q =
        if Float.is_nan p then if Float.is_nan q then 0 else 1
        else if Float.is_nan q then -1
        else compare p q
      in
      let c = order x.re y.re in
      if c <> 0 then c else order x.im y.im)
    values;
  Concrete (Ndarray.of_complex dt sh values)

let bitwise_count a = NL.astype (L.population_count (U.absolute a)) (dtype_of a)

let reject_complex name x =
  match dtype_of x with
  | Dtype.Complex64 | Dtype.Complex128 ->
      raise
        (Errors.jax_type_error
           (name ^ " does not support complex-valued inputs"))
  | _ -> ()

let frexp_info = function
  | Dtype.F16 -> (10, 5, -14, 6.103515625e-05, Dtype.I16)
  | Dtype.Bf16 -> (7, 8, -126, 1.1754943508222875e-38, Dtype.I16)
  | Dtype.F64 -> (52, 11, -1022, 2.2250738585072014e-308, Dtype.I64)
  | _ -> (23, 8, -126, 1.1754943508222875e-38, Dtype.I32)

let frexp_body x =
  let dt = dtype_of x in
  let nmant, nexp, minexp, tiny, int_dt = frexp_info dt in
  let lit v =
    if int_dt = Dtype.I64 then Concrete (Ndarray.of_int64s int_dt [||] [| v |])
    else Concrete (Ndarray.of_floats int_dt [||] [| Int64.to_float v |])
  in
  let lit_f v = Concrete (Ndarray.of_floats dt [||] [| v |]) in
  let mask = Int64.pred (Int64.shift_left 1L nexp) in
  let bias = Int64.of_int (1 - minexp) in
  let cond = U.less (L.abs x) (lit_f tiny) in
  let scaled = U.multiply x (lit_f (Float.ldexp 1.0 nmant)) in
  let x1 = NL.where_ cond scaled x in
  let x2 = NL.where_ cond (lit (Int64.of_int (-nmant))) (lit 0L) in
  let x1 = Core.bind1 (Bitcast_convert_type int_dt) [ x1 ] in
  let shifted =
    Core.bind1 Shift_right_arithmetic [ x1; lit (Int64.of_int nmant) ]
  in
  let masked = Core.bind1 And [ shifted; lit mask ] in
  let less_bias = Core.bind1 Sub [ masked; lit bias ] in
  let plus_one = Core.bind1 Add [ less_bias; lit 1L ] in
  let x2 = Core.bind1 Add [ x2; plus_one ] in
  let cleared =
    Core.bind1 And [ x1; lit (Int64.lognot (Int64.shift_left mask nmant)) ]
  in
  let set =
    Core.bind1 Or [ cleared; lit (Int64.shift_left (Int64.pred bias) nmant) ]
  in
  let x1 = Core.bind1 (Bitcast_convert_type dt) [ set ] in
  let inf = U.isinf x in
  let nan = U.isnan x in
  let inf_or_nan = Core.bind1 Or [ inf; nan ] in
  let zero = Core.bind1 Eq [ x; lit_f 0.0 ] in
  let cond = Core.bind1 Or [ inf_or_nan; zero ] in
  let x2 = NL.where_ cond (L.full (shape x2) 0.0 int_dt) x2 in
  let m = NL.where_ cond x x1 in
  let e = if int_dt = Dtype.I32 then x2 else NL.astype x2 Dtype.I32 in
  [ m; e ]

let frexp_custom =
  let self = ref (fun _ -> []) in
  let call =
    Custom_derivatives.custom_jvp ~name:"_frexp" ~jvp_name:"_frexp_jvp"
      (function [ x ] -> frexp_body x | _ -> invalid_arg "frexp: arity")
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ x ], [ t ] -> (
            match !self [ x ] with
            | [ m; e ] ->
                let scale = U.exp2 (U.negative (NL.astype e (dtype_of t))) in
                ( [ m; e ],
                  [
                    U.multiply t scale;
                    Ad_util.symbolic_zero_of_aval (Core.get_aval e);
                  ] )
            | _ -> invalid_arg "frexp: arity")
        | _ -> invalid_arg "frexp: arity")
  in
  self := call;
  call

let frexp x =
  reject_complex "frexp" x;
  let x = as_inexact x in
  Jnp_util.jit "frexp" (fun args -> frexp_custom args) [ x ] |> function
  | [ m; e ] -> (m, e)
  | _ -> invalid_arg "frexp: arity"

let ldexp x exponent =
  reject_complex "ldexp" x;
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

let can_cast ?(casting = "safe") a b =
  let kind = function
    | Dtype.Bool -> 0
    | Dtype.Float0 -> invalid_arg "can_cast does not accept float0"
    | d when Dtypes.issubdtype d `Unsigned -> 1
    | d when Dtypes.issubdtype d `Signed -> 2
    | d when Dtypes.issubdtype d `Floating -> 3
    | _ -> 4
  in
  let bits = function Dtype.Bool -> 1 | d -> Dtypes.itemsize_bits d in
  let mantissa = function
    | Dtype.Complex64 -> 24
    | Dtype.Complex128 -> 53
    | d when Dtypes.issubdtype d `Floating -> (Dtypes.finfo d).nmant + 1
    | _ -> 0
  in
  let safe () =
    if a = b then true
    else
      match (kind a, kind b) with
      | 0, _ -> true
      | ka, kb when ka = kb -> bits a <= bits b
      | 1, 2 -> bits a < bits b
      | (1 | 2), (3 | 4) ->
          let needed = if kind a = 2 then bits a - 1 else bits a in
          mantissa b >= needed || (bits a <= 8 && mantissa b >= 8)
      | 3, 4 -> mantissa a <= mantissa b
      | _ -> false
  in
  match casting with
  | "no" | "equiv" -> a = b
  | "safe" -> safe ()
  | "same_kind" -> safe () || kind a <= kind b
  | "unsafe" -> true
  | c -> invalid_arg ("can_cast: unknown casting " ^ c)

let shape_of = shape

let size ?axis a =
  match axis with None -> size_of a | Some ax -> (shape a).(canon (rank a) ax)

let ndim = rank

let polydiv ?(trim_leading_zeros = false) u v =
  let u = as_inexact u and v = as_inexact v in
  let lu = size_of u and lv = size_of v in
  let dt = dtype_of u in
  let b1 = Core.bind1 in
  let scalar x = L.full [||] x dt in
  let index k =
    Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int k |])
  in
  let scale = b1 Types.Div [ scalar 1.0; L.index_in_dim ~keepdims:false v 0 ] in
  let q = ref (AC.zeros ~dtype:dt [| Stdlib.max (lu - lv + 1) 1 |]) in
  let r = ref u in
  for k = 0 to lu - lv do
    let d = b1 Types.Mul [ scale; L.index_in_dim ~keepdims:false !r k ] in
    q :=
      b1 Types.Dynamic_update_slice
        [ !q; L.broadcast_in_dim d [| 1 |] [||]; index k ];
    let delta = b1 Types.Mul [ b1 Types.Neg [ d ]; v ] in
    let window =
      b1
        (Types.Slice
           {
             start_indices = [| k |];
             limit_indices = [| k + lv |];
             strides = None;
           })
        [ !r ]
    in
    r :=
      b1 Types.Dynamic_update_slice
        [ !r; b1 Types.Add [ window; delta ]; index k ]
  done;
  if not trim_leading_zeros then (!q, !r)
  else begin
    let tolerance =
      match dt with
      | Dtype.F64 -> 1.4901161193847656e-08
      | _ -> 0.00034526698300124393
    in
    let data = flat_floats (host "polydiv" !r) in
    let start = ref 0 in
    while !start < Array.length data && Float.abs data.(!start) < tolerance do
      incr start
    done;
    let kept = Array.sub data !start (Array.length data - !start) in
    (!q, Concrete (Ndarray.of_floats dt [| Array.length kept |] kept))
  end

let polyfit ?rcond ?w x y deg =
  let x = as_inexact x and y = as_inexact y in
  let order = deg + 1 in
  let lhs = NL.vander ~n:order x in
  let lhs, y =
    match w with
    | None -> (lhs, y)
    | Some weights ->
        let weights = as_inexact weights in
        if rank weights <> 1 then
          raise (Errors.jax_type_error "expected a 1-d array for weights");
        if (shape weights).(0) <> (shape y).(0) then
          raise
            (Errors.jax_type_error "expected w and y to have the same length");
        let column = NL.reshape weights [| (shape weights).(0); 1 |] in
        ( U.multiply lhs column,
          U.multiply y (if rank y = 2 then column else weights) )
  in
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

let roots_of_coefficients p =
  let dt = dtype_of p in
  let n = (shape p).(0) in
  let d = n - 1 in
  let tail = L.slice_in_dim ~axis:0 p 1 n in
  let lead = L.slice_in_dim ~axis:0 p 0 1 in
  let row = L.neg (U.divide tail lead) in
  let top = NL.reshape row [| 1; d |] in
  let companion =
    if d = 1 then top
    else NL.concatenate ~axis:0 [ top; NL.eye ~m:d ~dtype:dt (d - 1) ]
  in
  Linalg.eigvals companion

let roots ?(strip_zeros = true) p =
  let p = as_inexact p in
  let n = (shape p).(0) in
  let real_dt = dtype_of p in
  let complex_dt =
    match real_dt with
    | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
    | _ -> Dtype.Complex64
  in
  if not strip_zeros then
    if n < 2 then Concrete (Ndarray.of_complex complex_dt [| 0 |] [||])
    else roots_of_coefficients p
  else begin
    let nd = host "roots" p in
    let nonzero i =
      match real_dt with
      | Dtype.Complex64 | Dtype.Complex128 ->
          Ndarray.get_c nd [| i |] <> Complex.zero
      | _ -> Ndarray.get_f nd [| i |] <> 0.0
    in
    let lead = ref 0 in
    while !lead < n && not (nonzero !lead) do
      incr lead
    done;
    let trail = ref n in
    while !trail > !lead && not (nonzero (!trail - 1)) do
      decr trail
    done;
    let zero_roots = n - !trail in
    let zeros =
      Concrete
        (Ndarray.of_complex complex_dt [| zero_roots |]
           (Array.make zero_roots Complex.zero))
    in
    if !trail - !lead < 2 then zeros
    else
      let values =
        roots_of_coefficients (L.slice_in_dim ~axis:0 p !lead !trail)
      in
      if zero_roots = 0 then values else NL.concatenate [ values; zeros ]
  end

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

type printoptions = {
  precision : int;
  threshold : int;
  edgeitems : int;
  suppress : bool;
}

let default_printoptions =
  { precision = 8; threshold = 1000; edgeitems = 3; suppress = false }

let current_printoptions = ref default_printoptions
let get_printoptions () = !current_printoptions

let set_printoptions ?precision ?threshold ?edgeitems ?suppress () =
  let o = !current_printoptions in
  current_printoptions :=
    {
      precision = Option.value precision ~default:o.precision;
      threshold = Option.value threshold ~default:o.threshold;
      edgeitems = Option.value edgeitems ~default:o.edgeitems;
      suppress = Option.value suppress ~default:o.suppress;
    }

let with_printoptions ?precision ?threshold ?edgeitems ?suppress f =
  let saved = !current_printoptions in
  set_printoptions ?precision ?threshold ?edgeitems ?suppress ();
  Fun.protect ~finally:(fun () -> current_printoptions := saved) f

let render_float o v =
  if Float.is_nan v then "nan"
  else if Float.is_integer v && Float.abs v < 1e16 then Printf.sprintf "%.0f." v
  else if v = Float.infinity then "inf"
  else if v = Float.neg_infinity then "-inf"
  else if o.suppress && Float.abs v < 1e16 then (
    let text = Printf.sprintf "%.*f" o.precision v in
    let n = ref (String.length text) in
    while text.[!n - 1] = '0' do
      decr n
    done;
    String.sub text 0 !n)
  else Printf.sprintf "%.*g" o.precision v

let render_element o nd idx =
  match Ndarray.dtype nd with
  | Dtype.Bool -> if Ndarray.get_f nd idx <> 0.0 then "True" else "False"
  | Dtype.Complex64 | Dtype.Complex128 ->
      let z = Ndarray.get_c nd idx in
      let im = render_float o (Float.abs z.Complex.im) in
      Printf.sprintf "%s%s%sj"
        (render_float o z.Complex.re)
        (if z.Complex.im < 0.0 then "-" else "+")
        im
  | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn
  | Dtype.F8_e3m4 | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn
  | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu
  | Dtype.F32 | Dtype.F64 ->
      render_float o (Ndarray.get_f nd idx)
  | _ -> Int64.to_string (Ndarray.get_i64 nd idx)

let rec render_elements o summarize nd idx dims =
  match dims with
  | [] -> render_element o nd (Array.of_list (List.rev idx))
  | d :: rest ->
      let item i = render_elements o summarize nd (i :: idx) rest in
      let items =
        if summarize && d > 2 * o.edgeitems then
          List.init o.edgeitems item @ [ "..." ]
          @ List.init o.edgeitems (fun i -> item (d - o.edgeitems + i))
        else List.init d item
      in
      "[" ^ String.concat " " items ^ "]"

let array_str a =
  let nd = host "array_str" a in
  let o = !current_printoptions in
  let sh = Ndarray.shape nd in
  let summarize = Array.fold_left ( * ) 1 sh > o.threshold in
  render_elements o summarize nd [] (Array.to_list sh)

let print_options () =
  let o = !current_printoptions in
  {
    Array_print.default_options with
    precision = o.precision;
    threshold = o.threshold;
    edgeitems = o.edgeitems;
    suppress = o.suppress;
  }

let array_repr a =
  Array_print.array_repr ~options:(print_options ()) (host "array_repr" a)

let array_str_jax a =
  Array_print.array_str ~options:(print_options ()) (host "array_str" a)

let bit_positions bitorder =
  let bits =
    Core.bind1 (Iota { dtype = Dtype.Uint8; shape = [| 8 |]; dimension = 0 }) []
  in
  bits

let check_bitorder bitorder =
  if bitorder <> "little" && bitorder <> "big" then
    invalid_arg "'order' must be either 'little' or 'big'"

let packbits ?axis ?(bitorder = "big") a =
  check_bitorder bitorder;
  (match dtype_of a with
  | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.Complex64
  | Dtype.Complex128 ->
      invalid_arg "Expected an input array of integer or boolean data type"
  | _ -> ());
  Jnp_util.jit1 "packbits"
    ~key:
      (Printf.sprintf "%s/%s"
         (match axis with None -> "n" | Some x -> string_of_int x)
         bitorder)
    (function
      | [ a ] ->
          let nonzero = U.not_equal a (L.full [||] 0.0 (dtype_of a)) in
          let arr = NL.astype nonzero Dtype.Uint8 in
          let bits = bit_positions bitorder in
          let bits =
            if bitorder = "big" then Core.bind1 (Rev [| 0 |]) [ bits ] else bits
          in
          let arr, ax =
            match axis with
            | None -> (flat arr, 0)
            | Some x -> (arr, canon (rank arr) x)
          in
          let arr = NL.swapaxes ax (-1) arr in
          let last = (shape arr).(rank arr - 1) in
          let remainder = last mod 8 in
          let arr =
            if remainder = 0 then arr
            else
              L.pad arr
                (L.full [||] 0.0 Dtype.Uint8)
                (Array.init (rank arr) (fun d ->
                     if d = rank arr - 1 then (0, 8 - remainder, 0)
                     else (0, 0, 0)))
          in
          let sa = shape arr in
          let r = Array.length sa in
          let arr =
            NL.reshape arr
              (Array.concat [ Array.sub sa 0 (r - 1); [| sa.(r - 1) / 8; 8 |] ])
          in
          let bits = NL.expand_dims bits (Array.init (rank arr - 1) Fun.id) in
          let shifted = U.left_shift arr bits in
          let summed = R.sum ~axis:[| -1 |] shifted in
          let packed = NL.astype summed Dtype.Uint8 in
          NL.swapaxes ax (-1) packed
      | _ -> invalid_arg "packbits: arity")
    [ a ]

let unpackbits ?axis ?count ?(bitorder = "big") a =
  if dtype_of a <> Dtype.Uint8 then
    invalid_arg "Expected an input array of unsigned byte data type";
  check_bitorder bitorder;
  Jnp_util.jit1 "unpackbits"
    ~key:
      (Printf.sprintf "%s/%s/%s"
         (match axis with None -> "n" | Some x -> string_of_int x)
         (match count with None -> "n" | Some c -> string_of_int c)
         bitorder)
    (function
      | [ a ] ->
          let positions = bit_positions bitorder in
          let bits =
            Core.bind1 Shift_left
              [
                Concrete (Ndarray.of_floats Dtype.Uint8 [||] [| 1.0 |]);
                positions;
              ]
          in
          let bits =
            if bitorder = "big" then Core.bind1 (Rev [| 0 |]) [ bits ] else bits
          in
          let arr, ax =
            match axis with
            | None -> (flat a, 0)
            | Some x -> (a, canon (rank a) x)
          in
          let arr = NL.swapaxes ax (-1) arr in
          let widened = NL.expand_dims arr [| rank arr |] in
          let expanded = NL.expand_dims bits (Array.init (rank arr) Fun.id) in
          let masked = U.bitwise_and widened expanded in
          let positive = U.greater masked (L.full [||] 0.0 Dtype.Uint8) in
          let unpacked = NL.astype positive Dtype.Uint8 in
          let su = shape unpacked in
          let r = Array.length su in
          let unpacked =
            NL.reshape unpacked
              (Array.append
                 (Array.sub su 0 (r - 2))
                 [| su.(r - 2) * su.(r - 1) |])
          in
          let unpacked =
            match count with
            | None -> unpacked
            | Some c ->
                let su = shape unpacked in
                let r = Array.length su in
                let last = su.(r - 1) in
                if c > last then
                  L.pad unpacked
                    (L.full [||] 0.0 Dtype.Uint8)
                    (Array.init r (fun d ->
                         if d = r - 1 then (0, c - last, 0) else (0, 0, 0)))
                else
                  let keep = if c >= 0 then c else max (last + c) 0 in
                  let hi = Array.copy su in
                  hi.(r - 1) <- keep;
                  L.slice unpacked (Array.make r 0) hi
          in
          NL.swapaxes ax (-1) unpacked
      | _ -> invalid_arg "unpackbits: arity")
    [ a ]

let interp ?left ?right ?period a0 a1 a2 =
  Eager_jit.one "numpy.extras.interp" (left, right, period) [] [ a0; a1; a2 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1; a2 ] -> interp ?left ?right ?period a0 a1 a2
      | _ -> assert false)

let polyfit ?rcond ?w a0 a1 a2 =
  Eager_jit.one "numpy.extras.polyfit" (rcond, a2) [ w ] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ w ], [ a0; a1 ] -> polyfit ?rcond ?w a0 a1 a2
      | _ -> assert false)

let bitwise_count a0 =
  Eager_jit.one "numpy.extras.bitwise_count" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> bitwise_count a0
      | _ -> assert false)

let ldexp a0 a1 =
  Eager_jit.one "numpy.extras.ldexp" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> ldexp a0 a1
      | _ -> assert false)

let frexp a0 =
  Eager_jit.pair "numpy.extras.frexp" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> frexp a0 | _ -> assert false)

let sort_complex a0 =
  Eager_jit.one "numpy.extras.sort_complex" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sort_complex a0
      | _ -> assert false)

let packbits ?axis ?bitorder a0 =
  Eager_jit.one "numpy.extras.packbits" (axis, bitorder) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> packbits ?axis ?bitorder a0
      | _ -> assert false)

let unpackbits ?axis ?count ?bitorder a0 =
  Eager_jit.one "numpy.extras.unpackbits" (axis, count, bitorder) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> unpackbits ?axis ?count ?bitorder a0
      | _ -> assert false)

let gradient ?spacing ?axis a0 =
  Eager_jit.many "numpy.extras.gradient" (spacing, axis) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> gradient ?spacing ?axis a0
      | _ -> assert false)
