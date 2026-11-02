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

let canonicalize_axis axis n =
  if axis < -n || axis >= n then
    invalid_arg
      (Printf.sprintf "axis %d is out of bounds for array of dimension %d" axis
         n);
  canon n axis

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
    s_sorted_unique = false;
    s_clip = false;
  }

let normalised_indices ~extent indices =
  let dt = dtype_of indices in
  let negative = U.less indices (L.full [||] 0.0 dt) in
  let shifted = U.add indices (L.full [||] (float_of_int extent) dt) in
  let normalised = Core.bind1 Select_n [ negative; indices; shifted ] in
  let narrow =
    if dt = Dtype.I64 then
      Core.bind1 (Convert_element_type Dtype.I32) [ normalised ]
    else normalised
  in
  L.broadcast_in_dim narrow [| size_of indices; 1 |] [| 0 |]

let check_index_extent dt extent =
  let bound =
    match dt with
    | Dtype.I8 -> Some (127, "int8")
    | Dtype.I16 -> Some (32767, "int16")
    | Dtype.I32 -> Some (2147483647, "int32")
    | _ -> None
  in
  match bound with
  | Some (largest, name) when extent > largest ->
      invalid_arg
        (Printf.sprintf "Python integer %d out of bounds for %s" extent name)
  | _ -> ()

let normalize_index idx extent =
  let dt = dtype_of idx in
  check_index_extent dt extent;
  match dt with
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint32 | Dtype.Uint64 -> idx
  | _ ->
      let negative = Core.bind1 Lt [ idx; Core.splat dt [||] 0.0 ] in
      let shifted =
        Core.bind1 Add [ idx; Core.splat dt [||] (float_of_int extent) ]
      in
      Core.bind1 Select_n [ negative; idx; shifted ]

let expand_last v =
  let sh = shape v in
  L.broadcast_in_dim v (Array.append sh [| 1 |])
    (Array.init (Array.length sh) Fun.id)

let index_dtype extents =
  if List.exists (fun d -> d > 2147483647) extents then Dtype.I64 else Dtype.I32

let index_array ~extents idxs =
  let normal = List.map2 normalize_index idxs extents in
  let normal = NL.broadcast_arrays normal in
  let dt = index_dtype extents in
  let converted =
    List.map
      (fun v -> if dtype_of v = dt then v else L.convert_element_type v dt)
      normal
  in
  match List.map expand_last converted with
  | [ only ] -> only
  | parts -> Core.bind1 (Concatenate (rank (List.hd parts) - 1)) parts

let at_update ?(axis = 0) build x idxs y =
  let xa = Core.get_aval x and ya = Core.get_aval y in
  let k = List.length idxs in
  let sh = xa.shape in
  let extents = List.init k (fun j -> sh.(axis + j)) in
  let indices = index_array ~extents idxs in
  let idx_shape =
    let s = shape indices in
    Array.sub s 0 (Array.length s - 1)
  in
  let m = Array.length idx_shape in
  let trailing = Array.sub sh (axis + k) (Array.length sh - axis - k) in
  let slice_shape = Array.concat [ Array.sub sh 0 axis; idx_shape; trailing ] in
  let dt, weak =
    Dtypes.result_type [ (xa.dtype, xa.weak_type); (ya.dtype, ya.weak_type) ]
  in
  let x' = Core.convert_like ~weak x dt in
  let y' = Core.convert_like ~weak y dt in
  let y' = NL.broadcast_to y' slice_shape in
  let dims =
    {
      update_window_dims =
        Array.append (Array.init axis Fun.id)
          (Array.init (Array.length trailing) (fun i -> axis + m + i));
      inserted_window_dims = Array.init k (fun j -> axis + j);
      scatter_dims_to_operand_dims = Array.init k (fun j -> axis + j);
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
      s_sorted_unique = false;
      s_clip = false;
    }
  in
  let out = build x' indices y' dims in
  Core.convert_like ~weak:xa.weak_type out xa.dtype

let at_set ?axis x idxs y =
  at_update ?axis (fun a i u d -> L.scatter a i u d) x idxs y

let at_add ?axis x idxs y =
  at_update ?axis (fun a i u d -> L.scatter_add a i u d) x idxs y

let array_of_values vs =
  let lift v =
    let sh = shape v in
    L.broadcast_in_dim v (Array.append [| 1 |] sh)
      (Array.init (Array.length sh) (fun i -> i + 1))
  in
  match List.map lift vs with
  | [ only ] -> only
  | parts -> Core.bind1 (Concatenate 0) parts

let bincount_length length weights x =
  check_index_extent (dtype_of x) length;
  let operand = L.full [| length |] 0.0 (dtype_of weights) in
  at_add operand [ NL.clip ~min:0.0 x ] weights

let nonzero ?size ?fill_value a =
  let sh = shape a in
  let n = Array.length sh in
  if n = 0 then
    invalid_arg
      "Calling nonzero on 0d arrays is not allowed. Use \
       jnp.atleast_1d(scalar).nonzero() instead.";
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
  if k = 0 || Array.exists (fun d -> d = 0) sh then
    List.map (fun _ -> AC.zeros ~dtype:int_dt [| k |]) (Array.to_list sh)
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
          U.remainder quotient (Jnp_util.py_int sh.(d)))
    in
    match fill_value with
    | None -> out
    | Some fill ->
        let idx = NL.arange ~dtype:(default_int ()) (float_of_int k) in
        let fill_mask = U.greater_equal idx (count ()) in
        List.map
          (fun entry -> NL.where_ fill_mask (Jnp_util.py_int fill) entry)
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

let index_gather a idx =
  let sh = shape a in
  let r = Array.length sh in
  let m = rank idx in
  let indices = index_array ~extents:[ sh.(0) ] [ idx ] in
  let slice_shape = Array.append (shape idx) (Array.sub sh 1 (r - 1)) in
  if Array.exists (fun d -> d = 0) slice_shape then
    L.full slice_shape 0.0 (dtype_of a)
  else
    Core.bind1
      (Gather
         {
           dimension_numbers =
             {
               offset_dims = Array.init (r - 1) (fun i -> m + i);
               collapsed_slice_dims = [| 0 |];
               start_index_map = [| 0 |];
               g_operand_batching_dims = [||];
               g_start_indices_batching_dims = [||];
               g_sorted_unique = false;
             };
           slice_sizes = Array.mapi (fun i d -> if i = 0 then 1 else d) sh;
         })
      [ a; indices ]

let fancy_take x indices =
  let sh = shape x in
  let r = Array.length sh in
  let d = List.length indices in
  let normalised =
    List.mapi
      (fun k idx ->
        let dt = dtype_of idx in
        let negative = U.less idx (L.full [||] 0.0 dt) in
        let shifted = U.add idx (L.full [||] (float_of_int sh.(k)) dt) in
        Core.bind1 Select_n [ negative; idx; shifted ])
      indices
  in
  let narrowed =
    List.map
      (fun idx ->
        if dtype_of idx = Dtype.I64 then
          Core.bind1 (Convert_element_type Dtype.I32) [ idx ]
        else idx)
      normalised
  in
  let lifted =
    List.map
      (fun idx -> L.broadcast_in_dim idx [| size_of idx; 1 |] [| 0 |])
      narrowed
  in
  let start =
    match lifted with [ only ] -> only | _ -> L.concatenate lifted 1
  in
  let slice_shape =
    Array.append (shape (List.hd indices)) (Array.sub sh d (r - d))
  in
  if Array.exists (fun e -> e = 0) slice_shape then
    L.full slice_shape 0.0 (dtype_of x)
  else
    Core.bind1
      (Gather
         {
           dimension_numbers =
             {
               offset_dims = Array.init (r - d) (fun i -> i + 1);
               collapsed_slice_dims = Array.init d Fun.id;
               start_index_map = Array.init d Fun.id;
               g_operand_batching_dims = [||];
               g_start_indices_batching_dims = [||];
               g_sorted_unique = false;
             };
           slice_sizes = Array.mapi (fun i e -> if i < d then 1 else e) sh;
         })
      [ x; start ]

let leading_slice v start stop =
  let sh = shape v in
  Core.bind1
    (Slice
       {
         start_indices = Array.mapi (fun i _ -> if i = 0 then start else 0) sh;
         limit_indices = Array.mapi (fun i d -> if i = 0 then stop else d) sh;
         strides = None;
       })
    [ v ]

let at_leading_slice_set x start y =
  let index =
    L.broadcast_in_dim
      (Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int start |]))
      [| 1 |] [||]
  in
  let r = rank x in
  L.scatter x index y
    {
      update_window_dims = Array.init r Fun.id;
      inserted_window_dims = [||];
      scatter_dims_to_operand_dims = [| 0 |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
      s_sorted_unique = true;
      s_clip = false;
    }

let unique_sorted_mask ar axis equal_nan =
  Jnp_util.jit "_unique_sorted_mask"
    ~key:(Printf.sprintf "%d,%b" axis equal_nan)
    (function
      | [ ar ] ->
          let aux =
            if axis = 0 then ar else NL.moveaxis [| axis |] [| 0 |] ar
          in
          let aux =
            if is_complex_value aux then
              NL.where_ (U.isnan aux) (scalar_like aux Float_format.nan) aux
            else aux
          in
          let sh = shape aux in
          let rows = sh.(0) in
          let row_len =
            Array.fold_left ( * ) 1 (Array.sub sh 1 (Array.length sh - 1))
          in
          let rows = if row_len = 0 then 1 else rows in
          let perm =
            if row_len = 0 then L.full [| 1 |] 0.0 (default_int ())
            else
              Sorting.lexsort_array
                (L.rev
                   (NL.transpose (NL.reshape aux [| rows; row_len |]))
                   [| 0 |])
          in
          let aux = index_gather aux perm in
          if size_of aux = 0 then
            [ aux; L.full [| rows |] 0.0 Dtype.Bool; perm ]
          else
            let ones = L.full [| rows |] 1.0 Dtype.Bool in
            let tail = leading_slice aux 1 rows in
            let head = leading_slice aux 0 (rows - 1) in
            let inexact =
              Dtypes.isdtype (dtype_of aux) "numeric"
              && not (Dtypes.isdtype (dtype_of aux) "integral")
            in
            let differs =
              if inexact && equal_nan then
                let ne = L.ne tail head in
                let both = U.bitwise_and (U.isnan tail) (U.isnan head) in
                U.bitwise_and ne (U.logical_not both)
              else L.ne tail head
            in
            let any_diff =
              R.any
                ~axis:(Array.init (Array.length sh - 1) (fun i -> i + 1))
                differs
            in
            [ aux; at_leading_slice_set ones 1 any_diff; perm ]
      | _ -> invalid_arg "unique: arity")
    [ ar ]

type unique_flags = {
  want_index : bool;
  want_inverse : bool;
  want_counts : bool;
}

let no_unique_flags =
  { want_index = false; want_inverse = false; want_counts = false }

let all_unique_flags =
  { want_index = true; want_inverse = true; want_counts = true }

let traced_unique_sized ?(flags = all_unique_flags) ?(equal_nan = true) ?axis
    ~size ?fill_value a =
  let idt = default_int () in
  let reshape_inverse = axis = None in
  let arr = match axis with None -> NL.ravel a | Some _ -> a in
  let ax = match axis with None -> 0 | Some x -> canon (rank a) x in
  let aux, mask, perm =
    match unique_sorted_mask arr ax equal_nan with
    | [ aux; mask; perm ] -> (aux, mask, perm)
    | _ -> assert false
  in
  let mask_size = size_of mask in
  let aux_empty = size_of aux = 0 in
  let ind = List.hd (nonzero ~size mask) in
  let result = if aux_empty then aux else index_gather aux ind in
  let result =
    match fill_value with
    | None -> result
    | Some fv when (shape result).(0) = 0 ->
        let sh = Array.copy (shape result) in
        sh.(0) <- size;
        L.full sh fv (dtype_of result)
    | Some fv ->
        let positions = NL.arange ~dtype:idt (float_of_int size) in
        let sum = R.sum mask in
        let valid = U.less positions sum in
        let valid =
          let extra = rank result - 1 in
          if extra = 0 then valid
          else NL.expand_dims valid (Array.init extra (fun i -> i + 1))
        in
        NL.where_ valid result
          (Concrete (Ndarray.of_floats (dtype_of result) [||] [| fv |]))
  in
  let values = if ax = 0 then result else NL.moveaxis [| 0 |] [| ax |] result in
  let empty = ints_value idt [| 0 |] [||] in
  let indices =
    if not flags.want_index then empty
    else if aux_empty then perm
    else index_gather perm ind
  in
  let axis_size = (shape arr).(ax) in
  let inverse =
    if flags.want_inverse && aux_empty then begin
      let inv = L.full [| axis_size |] 0.0 idt in
      if reshape_inverse then NL.reshape inv (shape a) else inv
    end
    else if flags.want_inverse then begin
      let imask = U.subtract (R.cumsum mask) (Jnp_util.py_int 1) in
      let zeros = L.full (shape mask) 0.0 idt in
      let inv = at_set zeros [ perm ] imask in
      if reshape_inverse then NL.reshape inv (shape a) else inv
    end
    else empty
  in
  let counts =
    if flags.want_counts && aux_empty then
      if axis_size <> 0 then L.full [| 1 |] (float_of_int axis_size) idt
      else L.full [| 0 |] 0.0 idt
    else if flags.want_counts then begin
      let idx = List.hd (nonzero ~size:(size + 1) mask) in
      let tail = leading_slice idx 1 (size + 1) in
      let tail' = leading_slice idx 1 (size + 1) in
      let stops = NL.where_ tail tail' (Jnp_util.py_int mask_size) in
      NL.diff (at_leading_slice_set idx 1 stops)
    end
    else empty
  in
  ({ values; indices; inverse; counts }, fun () -> R.sum mask)

let traced_unique ?flags ?equal_nan ?axis ~size ?fill_value a =
  fst (traced_unique_sized ?flags ?equal_nan ?axis ~size ?fill_value a)

let rec unique_full ?flags ?equal_nan ?axis ?size ?fill_value a =
  (let extent =
     match axis with
     | None -> size_of a
     | Some ax -> (shape a).(canonicalize_axis ax (rank a))
   in
   match (size, fill_value) with
   | Some s, None when extent = 0 && s <> 0 ->
       invalid_arg
         "jnp.unique: for zero-sized input with nonzero size argument, \
          fill_value must be specified"
   | _ -> ());
  match size with
  | _ when is_complex_value a && axis = None ->
      unique_complex ?size ?fill_value a
  | Some size ->
      let axis_size =
        match axis with
        | None -> size_of a
        | Some x -> (shape a).(canon (rank a) x)
      in
      if axis_size = 0 && size <> 0 && fill_value = None then
        invalid_arg
          "jnp.unique: for zero-sized input with nonzero size argument, \
           fill_value must be specified";
      traced_unique ?flags ?equal_nan ?axis ~size ?fill_value a
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
      let equal_nan = Option.value equal_nan ~default:true in
      let same_group i j =
        compare_rows i j = 0
        && (equal_nan
           || not
                (Array.exists Float.is_nan (row_values i)
                || Array.exists Float.is_nan (row_values j)))
      in
      let groups =
        List.fold_left
          (fun acc i ->
            match acc with
            | (first, members) :: rest when same_group first i ->
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

let unique_values ?size ?fill_value a =
  (unique_full ~flags:no_unique_flags ~equal_nan:false ?size ?fill_value a)
    .values

let unique_counts ?size ?fill_value a =
  let flags = { no_unique_flags with want_counts = true } in
  let r = unique_full ~flags ~equal_nan:false ?size ?fill_value a in
  (r.values, r.counts)

let unique_inverse ?size ?fill_value a =
  let flags = { no_unique_flags with want_inverse = true } in
  let r = unique_full ~flags ~equal_nan:false ?size ?fill_value a in
  (r.values, r.inverse)

let unique_all ?size ?fill_value a =
  unique_full ~equal_nan:false ?size ?fill_value a

let unique ?(return_index = false) ?(return_inverse = false)
    ?(return_counts = false) ?axis ?equal_nan ?size ?fill_value ?sorted:_ a =
  let flags =
    {
      want_index = return_index;
      want_inverse = return_inverse;
      want_counts = return_counts;
    }
  in
  let r = unique_full ~flags ?equal_nan ?axis ?size ?fill_value a in
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
    traced_unique_sized ~flags:no_unique_flags ~axis:0 ~size:(size_of arr1)
      ?fill_value arr1
  in
  let n1 = n1 () in
  let u2, n2 =
    traced_unique_sized ~flags:no_unique_flags ~axis:0 ~size:(size_of arr2)
      ?fill_value arr2
  in
  let n2 = n2 () in
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
                if p = q then p :: adjacent rest else adjacent rest
            | _ -> []
          in
          adjacent sorted
        else
          let first = sorted_unique_floats "intersect1d" a in
          let ys = sorted_unique_floats "intersect1d" b in
          List.filter (fun x -> List.exists (fun y -> x = y) ys) first
      in
      sized_result (dtype_of a) xs ~size ~fill_value

let intersect1d_indices ?(assume_unique = false) a b =
  let a, b = same_dtype a b in
  let a = flat a and b = flat b in
  let dt = dtype_of a in
  let first = flat_floats (host "intersect1d" a) in
  let second = flat_floats (host "intersect1d" b) in
  let first_at arr x =
    let rec go i = if arr.(i) = x then i else go (i + 1) in
    go 0
  in
  let triples =
    if assume_unique then
      let n = Array.length first in
      let tagged =
        Array.to_list (Array.mapi (fun i x -> (x, i)) first)
        @ Array.to_list (Array.mapi (fun i x -> (x, n + i)) second)
      in
      let sorted =
        List.stable_sort (fun (p, _) (q, _) -> compare_with_nan p q) tagged
      in
      let rec adjacent = function
        | (p, i) :: ((q, j) :: _ as rest) ->
            if p = q then (p, i, j - n) :: adjacent rest else adjacent rest
        | _ -> []
      in
      adjacent sorted
    else
      List.filter_map
        (fun x ->
          if Array.mem x second then
            Some (x, first_at first x, first_at second x)
          else None)
        (sorted_unique_floats "intersect1d" a)
  in
  let idt =
    if assume_unique then Nd_index.index_dtype (shape a) else default_int ()
  in
  let count = List.length triples in
  let of_ints f =
    ints_value idt [| count |] (Array.of_list (List.map f triples))
  in
  ( Concrete
      (Ndarray.of_floats dt [| count |]
         (Array.of_list (List.map (fun (x, _, _) -> x) triples))),
    of_ints (fun (_, i, _) -> i),
    of_ints (fun (_, _, j) -> j) )

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
        (List.filter (fun x -> not (List.exists (fun y -> x = y) ys)) source)
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
        | p :: q :: rest when p = q ->
            let rec drop = function
              | r :: tl when p = r -> drop tl
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
      s_sorted_unique = false;
      s_clip = false;
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
  if rank x <> 1 then invalid_arg "only 1-dimensional input supported.";
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

let plain_slice v start stop =
  Core.bind1
    (Slice
       {
         start_indices = [| start |];
         limit_indices = [| stop |];
         strides = None;
       })
    [ v ]

let histogram_bin_edges ?(bins = 10) ?range ?weights:_ a =
  let a = as_inexact a in
  let dt = dtype_of a in
  let pair =
    match range with
    | Some (l, h) ->
        Jaxpr.stage_constant
          (Concrete (Ndarray.of_floats dt [| 2 |] [| l; h |]))
    | None ->
        let low = R.min a in
        let high = R.max a in
        let lifted v = L.broadcast_in_dim v [| 1 |] [||] in
        let low = lifted low in
        let high = lifted high in
        NL.concatenate [ low; high ]
  in
  let same () =
    let spread =
      Jnp_util.jit1 "_ptp"
        (function [ v ] -> R.ptp v | _ -> invalid_arg "ptp: arity")
        [ pair ]
    in
    U.equal spread (Jnp_util.py_int 0)
  in
  let half = Jnp_util.py_float 0.5 in
  let cond = same () in
  let shifted = U.subtract (NL.index_at pair 0) half in
  let lo' = NL.where_ cond shifted (NL.index_at pair 0) in
  let cond = same () in
  let shifted = U.add (NL.index_at pair 1) half in
  let hi' = NL.where_ cond shifted (NL.index_at pair 1) in
  AC.linspace_v ~num:(bins + 1) ~dtype:dt lo' hi'

let histogram_core a edges weights density =
  let n_edges = size_of edges in
  let idx = NL.searchsorted ~side:"right" edges a in
  let at_end = U.equal a (NL.index_at edges (-1)) in
  let idx = NL.where_ at_end (Jnp_util.py_int (n_edges - 1)) idx in
  let summed =
    at_add (L.full [| n_edges |] 0.0 (dtype_of weights)) [ idx ] weights
  in
  let counts = plain_slice summed 1 n_edges in
  if density then
    let widths = NL.diff edges in
    let scaled = U.divide counts widths in
    U.divide scaled (R.sum counts)
  else counts

let histogram_weights a weights =
  match weights with
  | None ->
      let a = as_inexact a in
      (a, AC.ones_like a)
  | Some w -> (
      if shape a <> shape w then
        invalid_arg "weights should have the same shape as a.";
      match NL.promote_values [ as_inexact a; as_inexact w ] with
      | [ a; w ] -> (a, w)
      | _ -> assert false)

let histogram ?(bins = 10) ?range ?weights ?(density = false) a =
  let a, weights = histogram_weights a weights in
  let edges = histogram_bin_edges ~bins ?range a in
  (histogram_core a edges weights density, edges)

let histogram_edges ?weights ?(density = false) a edges =
  let a, weights = histogram_weights a weights in
  (histogram_core a edges weights density, edges)

let ravel_multi_index ?(mode = "raise") ?(order = "C") ?dtype coords dims =
  let n = Array.length dims in
  let dt =
    match dtype with
    | Some d ->
        let d = Dtypes.canonicalize_dtype d in
        if not (Dtypes.issubdtype d `Integer) then
          invalid_arg
            (Printf.sprintf "dtype=dtype('%s') is not an integer type."
               (Dtype.numpy_name d));
        d
    | None ->
        fst
          (Dtypes.result_type
             ((default_int (), true)
             :: List.map
                  (fun c ->
                    let a = Core.get_aval c in
                    (a.dtype, a.weak_type))
                  coords))
  in
  let check c d =
    let values = flat_floats (host "ravel_multi_index" c) in
    if Array.exists (fun v -> v < 0.0 || v >= float_of_int d) values then
      invalid_arg "invalid entry in coordinates array"
  in
  let coords =
    match mode with
    | "raise" ->
        List.iteri (fun i c -> check c dims.(i)) coords;
        coords
    | "clip" ->
        List.mapi
          (fun i c -> NL.clip ~min:0.0 ~max:(float_of_int (dims.(i) - 1)) c)
          coords
    | "wrap" ->
        List.mapi (fun i c -> U.remainder c (Jnp_util.py_int dims.(i))) coords
    | "ignore" -> coords
    | other -> invalid_arg ("invalid mode=" ^ other)
  in
  let strides =
    match order with
    | "C" -> strides_of dims
    | "F" ->
        let s = Array.make n 1 in
        for d = 1 to n - 1 do
          s.(d) <- s.(d - 1) * dims.(d - 1)
        done;
        s
    | other -> invalid_arg ("invalid order=" ^ other)
  in
  let start = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
  let _, result =
    List.fold_left
      (fun (i, acc) c ->
        let term = U.multiply (NL.astype c dt) (Jnp_util.py_int strides.(i)) in
        (i + 1, U.add acc term))
      (0, start) coords
  in
  result

let histogramdd ?(bins = [||]) ?range ?weights ?(density = false) sample =
  let sample, weights =
    match weights with
    | None -> (as_inexact sample, None)
    | Some w -> (
        match NL.promote_values [ as_inexact sample; as_inexact w ] with
        | [ s; w ] -> (s, Some w)
        | _ -> assert false)
  in
  let n_dims = (shape sample).(1) in
  if Array.length bins <> 0 && Array.length bins <> n_dims then
    invalid_arg "should be a bin for each dimension.";
  let bins_for i = if Array.length bins = 0 then 10 else bins.(i) in
  let column i =
    let sh = shape sample in
    L.squeeze
      (Core.bind1
         (Slice
            {
              start_indices = [| 0; i |];
              limit_indices = [| sh.(0); i + 1 |];
              strides = None;
            })
         [ sample ])
      [| 1 |]
  in
  let per_dim =
    List.init n_dims (fun i ->
        let range_i = Option.map (fun r -> r.(i)) range in
        let c = column i in
        let e = histogram_bin_edges ~bins:(bins_for i) ?range:range_i c in
        let c = column i in
        let idx = NL.searchsorted ~side:"right" e c in
        let c = column i in
        let at_end = U.equal c (NL.index_at e (-1)) in
        let idx = NL.where_ at_end (U.subtract idx (Jnp_util.py_int 1)) idx in
        (idx, e))
  in
  let idxs = List.map fst per_dim and edges = List.map snd per_dim in
  let nbins = Array.of_list (List.map (fun e -> size_of e + 1) edges) in
  let dedges = List.map NL.diff edges in
  let xy = ravel_multi_index ~mode:"clip" idxs nbins in
  let hist = bincount ?weights ~length:(shape_int nbins) xy in
  let reshaped = NL.reshape hist nbins in
  let start = Array.make n_dims 1 in
  let limit = Array.map (fun d -> d - 1) nbins in
  let core =
    Core.bind1
      (Slice { start_indices = start; limit_indices = limit; strides = None })
      [ reshaped ]
  in
  let out =
    if density then begin
      let core = NL.astype core (dtype_of sample) in
      let core = U.divide core (R.sum core) in
      List.fold_left (fun acc norm -> U.divide acc norm) core (NL.ix_ dedges)
    end
    else core
  in
  (out, edges)

let histogram2d ?(bins = [| 10; 10 |]) ?range ?weights ?(density = false) x y =
  let sample = NL.transpose (array_of_values (NL.promote_values [ x; y ])) in
  match histogramdd ~bins ?range ?weights ~density sample with
  | hist, [ xe; ye ] -> (hist, xe, ye)
  | _ -> failwith "histogram2d: arity"

let interp_body ?left ?right ?period x xp fp =
  if shape xp <> shape fp || rank xp <> 1 then
    invalid_arg "xp and fp must be one-dimensional arrays of equal size";
  let x, xp =
    match NL.promote_values [ as_inexact x; as_inexact xp ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let fp = as_inexact fp in
  let idx = NL.take_trailing in
  let int_lit = Jnp_util.py_int in
  let x, xp, fp =
    match period with
    | None -> (x, xp, fp)
    | Some p ->
        let p = U.absolute p in
        let x = U.remainder x p in
        let xp = U.remainder xp p in
        let xp, fp = match L.sort_key_val xp fp with k, v -> (k, v) in
        let n = size_of xp in
        let last = plain_slice xp (n - 1) n in
        let last = U.subtract last p in
        let first = U.add (plain_slice xp 0 1) p in
        let xp = NL.concatenate [ last; xp; first ] in
        let fp_last = plain_slice fp (n - 1) n in
        let fp_first = plain_slice fp 0 1 in
        (x, xp, NL.concatenate [ fp_last; fp; fp_first ])
  in
  let n = size_of xp in
  let i =
    NL.clip ~min:1.0
      ~max:(float_of_int (n - 1))
      (NL.searchsorted ~side:"right" xp x)
  in
  let prev () = U.subtract i (int_lit 1) in
  let at_fp j = idx fp [ j ] and at_xp j = idx xp [ j ] in
  let fp_i = at_fp i in
  let fp_p = at_fp (prev ()) in
  let df = U.subtract fp_i fp_p in
  let xp_i = at_xp i in
  let xp_p = at_xp (prev ()) in
  let dx = U.subtract xp_i xp_p in
  let xp_q = at_xp (prev ()) in
  let delta = U.subtract x xp_q in
  let eps =
    match dtype_of xp with
    | Dtype.F64 -> 4.930380657631324e-32
    | _ -> 1.4210854715202004e-14
  in
  let dx0 =
    Core.bind1 Types.Le
      [
        Core.bind1 Types.Abs [ dx ];
        Concrete (Ndarray.of_floats (dtype_of dx) [||] [| eps |]);
      ]
  in
  let fp_r = at_fp (prev ()) in
  let fp_s = at_fp (prev ()) in
  let safe = NL.where_ dx0 (int_lit 1) dx in
  let f =
    NL.where_ dx0 fp_r (U.add fp_s (U.multiply (U.divide delta safe) df))
  in
  let left_arr = match left with None -> NL.index_at fp 0 | Some v -> v in
  let f =
    if period = None then NL.where_ (U.less x (NL.index_at xp 0)) left_arr f
    else f
  in
  let right_arr =
    match right with None -> NL.index_at fp (-1) | Some v -> v
  in
  let f =
    if period = None then
      NL.where_ (U.greater x (NL.index_at xp (-1))) right_arr f
    else f
  in
  f

let interp ?left ?right ?period a0 a1 a2 =
  let lift = Option.map Jnp_util.py_float in
  Eager_jit.one ~jit:"_interp" ~vals_first:true "numpy.extras.interp" ()
    [ lift left; lift right; lift period ]
    [ a0; a1; a2 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ left; right; period ], [ a0; a1; a2 ] ->
          interp_body ?left ?right ?period a0 a1 a2
      | _ -> assert false)

let gradient_along_axis a h axis =
  let n = (shape a).(axis) in
  let cut lo hi = L.slice_in_dim ~axis a lo hi in
  let upper_edge =
    let hi = cut 1 2 in
    U.subtract hi (cut 0 1)
  in
  let lower_edge =
    let hi = cut (n - 1) n in
    U.subtract hi (cut (n - 2) (n - 1))
  in
  let inner =
    let hi = cut 2 n in
    let diff = U.subtract hi (cut 0 (n - 2)) in
    U.divide (U.multiply diff (Jnp_util.py_float 0.5)) h
  in
  let lower = U.divide lower_edge h in
  let upper = U.divide upper_edge h in
  NL.concatenate ~axis [ upper; inner; lower ]

let gradient ?spacing ?axis a =
  let n = rank a in
  let axes =
    match axis with
    | Some ax -> List.map (canon n) ax
    | None -> List.init n Fun.id
  in
  if axes <> [] && List.exists (fun ax -> (shape a).(ax) < 2) axes then
    invalid_arg
      "Shape of array too small to calculate a numerical gradient, at least 2 \
       elements are required.";
  let key = Digest.to_hex (Digest.string (Marshal.to_string axis [])) in
  match spacing with
  | None ->
      Jnp_util.jit "gradient" ~key
        (function
          | [ a ] ->
              let a = as_inexact a in
              let h = Jnp_util.py_float 1.0 in
              List.map (fun ax -> gradient_along_axis a h ax) axes
          | _ -> invalid_arg "gradient: arity")
        [ a ]
  | Some spacing ->
      Jnp_util.jit "gradient" ~key
        (function
          | [ a; spacing ] ->
              let aval v = Core.get_aval v in
              let dt, is_weak =
                Dtypes.result_type
                  [
                    ((aval a).dtype, (aval a).weak_type);
                    ((aval spacing).dtype, (aval spacing).weak_type);
                  ]
              in
              let target = Dtypes.to_inexact_dtype dt in
              let a = Core.convert_like ~weak:is_weak a target in
              let h = Core.convert_like ~weak:is_weak spacing target in
              List.map (fun ax -> gradient_along_axis a h ax) axes
          | _ -> invalid_arg "gradient: arity")
        [ a; Jnp_util.py_float spacing ]

let choose ?(mode = "raise") a choices =
  let n = List.length choices in
  let arr =
    match mode with
    | "raise" ->
        let values = flat_floats (host "choose" a) in
        if Array.exists (fun v -> v < 0.0 || v >= float_of_int n) values then
          invalid_arg "invalid entry in choice array";
        a
    | "wrap" -> U.remainder a (Jnp_util.py_int n)
    | "clip" -> NL.clip ~min:0.0 ~max:(float_of_int (n - 1)) a
    | other -> invalid_arg ("invalid mode=" ^ other)
  in
  match NL.broadcast_arrays (arr :: choices) with
  | [] -> invalid_arg "choose: arity"
  | arr :: choices ->
      let sh = shape arr in
      let r = Array.length sh in
      let stacked = array_of_values choices in
      let dt = default_int () in
      let grids =
        List.init r (fun i ->
            let idx =
              Core.bind1
                (Iota { dtype = dt; shape = [| sh.(i) |]; dimension = 0 })
                []
            in
            let s = Array.init r (fun d -> if d = i then sh.(i) else 1) in
            if r = 1 then idx else L.broadcast_in_dim idx s [| i |])
      in
      let extents = n :: Array.to_list sh in
      let indices = index_array ~extents (arr :: grids) in
      if Array.exists (fun e -> e = 0) sh then L.full sh 0.0 (dtype_of stacked)
      else
        Core.bind1
          (Gather
             {
               dimension_numbers =
                 {
                   offset_dims = [||];
                   collapsed_slice_dims = Array.init (r + 1) Fun.id;
                   start_index_map = Array.init (r + 1) Fun.id;
                   g_operand_batching_dims = [||];
                   g_start_indices_batching_dims = [||];
                   g_sorted_unique = false;
                 };
               slice_sizes = Array.make (r + 1) 1;
             })
          [ stacked; indices ]

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
  List.iter
    (fun i ->
      if i < -n || i >= n then
        raise
          (Errors.Jax_index_error
             (Printf.sprintf "index %d is out of bounds for axis 0 with size %d"
                i n)))
    obj;
  let drop = List.map (canon n) obj in
  let kept =
    List.filter (fun i -> not (List.mem i drop)) (List.init n Fun.id)
  in
  I.take_index ~axis:ax a
    (ints_value (default_int ()) [| List.length kept |] (Array.of_list kept))

let delete_array ?axis ?(assume_unique_indices = false) a obj =
  let a, ax =
    match axis with Some x -> (a, canon (rank a) x) | None -> (flat a, 0)
  in
  if assume_unique_indices && Dtypes.issubdtype (dtype_of obj) `Integer then begin
    let extent = (shape a).(ax) in
    let dt = dtype_of obj in
    let idx = flat obj in
    let negative = U.less idx (Jnp_util.py_int 0) in
    let wrapped = U.add idx (Jnp_util.py_int extent) in
    let normal = NL.where_ negative wrapped idx in
    let clipped = NL.clip ~min:0.0 ~max:(float_of_int extent) normal in
    let sorted = Sorting.sort clipped in
    let count = size_of sorted in
    let ramp n =
      L.broadcast_in_dim
        (Core.bind1 (Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [])
        [| n |] [| 0 |]
    in
    let shifted = U.subtract sorted (ramp count) in
    let base = ramp (extent - count) in
    let above =
      U.greater_equal
        (L.expand_dims base [| 0 |])
        (L.expand_dims shifted [| 1 |])
    in
    let adjust = R.sum ~dtype:dt ~axis:[| 0 |] above in
    I.take_index ~axis:ax a (U.add base adjust)
  end
  else
    let values = flat_floats (host "delete" obj) in
    delete ~axis:ax a (Array.to_list (Array.map int_of_float values))

let tile_to_size vals n =
  let m = size_of vals in
  let vals =
    if m < n then
      NL.tile vals
        [| int_of_float (Float.ceil (float_of_int n /. float_of_int m)) |]
    else vals
  in
  if size_of vals > n then L.slice vals [| 0 |] [| n |] else vals

let insert ?axis a position values =
  let a, ax =
    match axis with Some x -> (a, canon (rank a) x) | None -> (flat a, 0)
  in
  let ndim = rank a in
  let dt = dtype_of a in
  let values =
    let v = if dtype_of values = dt then values else NL.astype values dt in
    if rank v >= ndim then v
    else
      let lead = ndim - rank v in
      L.broadcast_in_dim v
        (Array.append (Array.make lead 1) (shape v))
        (Array.init (rank v) (fun i -> i + lead))
  in
  let index =
    let raveled = NL.reshape (Jnp_util.py_int position) [| 1 |] in
    L.squeeze raveled [| 0 |]
  in
  let values = if ax = 0 then values else NL.moveaxis [| 0 |] [| ax |] values in
  let n_insert = (shape values).(ax) in
  let indices = L.broadcast_in_dim index [| n_insert |] [||] in
  let n_input = (shape a).(ax) in
  let out_shape = Array.copy (shape a) in
  out_shape.(ax) <- n_input + n_insert;
  let out = L.full out_shape 0.0 dt in
  let indices =
    NL.where_
      (U.less indices (Jnp_util.py_int 0))
      (U.add indices (Jnp_util.py_int n_input))
      indices
  in
  let indices = NL.clip ~min:0.0 ~max:(float_of_int n_input) indices in
  let order = Sorting.argsort ~axis:(Some 0) indices in
  let values_ind =
    at_add indices [ order ]
      (NL.arange ~dtype:(dtype_of indices) (float_of_int n_insert))
  in
  let arr_mask =
    at_set
      (L.full [| n_input + n_insert |] 1.0 Dtype.Bool)
      [ values_ind ]
      (Concrete (Ndarray.of_floats Dtype.Bool [||] [| 0.0 |]))
  in
  let arr_ind = List.hd (nonzero ~size:n_input arr_mask) in
  let out = at_set ~axis:ax out [ values_ind ] values in
  at_set ~axis:ax out [ arr_ind ] a

let place a mask values =
  let vals = NL.ravel values in
  if size_of a <> size_of mask then
    invalid_arg "place: arr and mask must be the same size";
  if size_of vals = 0 then invalid_arg "Cannot place values from an empty array";
  if size_of a = 0 then a
  else begin
    let n = size_of mask in
    let indices = List.hd (nonzero ~size:n ~fill_value:n (NL.ravel mask)) in
    let vals = tile_to_size vals n in
    NL.reshape (at_set (NL.ravel a) [ indices ] vals) (shape a)
  end

let vectorize_scalar f args =
  let shapes = List.map shape args in
  let broadcast = NL.broadcast_shapes_n shapes in
  let r = Array.length broadcast in
  let filled =
    List.map
      (fun s -> Array.append (Array.make (r - Array.length s) 1) s)
      shapes
  in
  let squeezed =
    List.map
      (fun a ->
        let s = shape a in
        let axes =
          List.filter (fun i -> s.(i) = 1) (List.init (Array.length s) Fun.id)
        in
        if axes = [] then a else NL.squeeze ~axis:(Array.of_list axes) a)
      args
  in
  let func = ref f and expand = ref [] in
  for pos = r - 1 downto 0 do
    let in_axes =
      List.map (fun s -> if s.(pos) = 1 then None else Some 0) filled
    in
    if List.for_all Option.is_none in_axes then expand := pos :: !expand
    else begin
      let inner = !func in
      func := fun xs -> Interpreters.Batching.vmap inner in_axes xs
    end
  done;
  let out = !func squeezed in
  match !expand with
  | [] -> out
  | dims -> List.map (fun v -> NL.expand_dims v (Array.of_list dims)) out

let piecewise_counter = ref 0

let piecewise x condlist funclist =
  let n_cond = List.length condlist in
  let n_fun = List.length funclist in
  if n_fun <> n_cond && n_fun <> n_cond + 1 then
    invalid_arg "with N condition(s), either N or N+1 functions are expected";
  let leading, callables =
    if n_fun = n_cond + 1 then
      ( None,
        List.nth funclist n_cond
        :: List.filteri (fun i _ -> i < n_cond) funclist )
    else (Some (Jnp_util.py_int 0), funclist)
  in
  let conds = array_of_values condlist in
  incr piecewise_counter;
  let key = "piecewise" ^ string_of_int !piecewise_counter in
  Jnp_util.jit1 "_piecewise" ~key
    (fun args ->
      let x, conds, const =
        match args with
        | [ x; c ] -> (x, c, None)
        | [ x; c; k ] -> (x, c, Some k)
        | _ -> invalid_arg "piecewise: arity"
      in
      let dt = dtype_of x in
      let lead =
        let sh = shape conds in
        let first =
          L.slice conds
            (Array.make (Array.length sh) 0)
            (Array.mapi (fun d e -> if d = 0 then 1 else e) sh)
        in
        L.full (shape first) 0.0 (dtype_of first)
      in
      let indices =
        NL.argmax ~axis:0
          (R.cumsum ~axis:0 (NL.concatenate ~axis:0 [ lead; conds ]))
      in
      let of_fun f ops =
        match ops with
        | [ v ] -> [ NL.astype (f v) dt ]
        | _ -> invalid_arg "piecewise: arity"
      in
      let branches =
        (match const with
          | Some c -> [ (fun _ -> [ NL.astype (NL.copy c) dt ]) ]
          | None -> [])
        @ List.map of_fun callables
      in
      let switch = function
        | [ i; v ] -> L.switch i branches [ v ]
        | _ -> invalid_arg "piecewise: arity"
      in
      List.hd (vectorize_scalar switch [ indices; x ]))
    ([ x; conds ] @ Option.to_list leading)

let fill_diagonal ?(wrap = false) a value =
  if wrap then invalid_arg "wrap=True is not implemented, must use wrap=False";
  let sh = shape a in
  if Array.length sh < 2 then invalid_arg "array must be at least 2-d";
  let is_2d = Array.length sh = 2 in
  if (not is_2d) && Array.exists (fun d -> d <> sh.(0)) sh then
    invalid_arg "All dimensions of input must be of equal length";
  let n = Array.fold_left Stdlib.min sh.(0) sh in
  let idx =
    Core.bind1
      (Iota { dtype = default_int (); shape = [| n |]; dimension = 0 })
      []
  in
  let v = if rank value = 0 then value else tile_to_size (NL.ravel value) n in
  let v = if dtype_of v = dtype_of a then v else NL.astype v (dtype_of a) in
  at_set a (List.init (Array.length sh) (fun _ -> idx)) v

let rec triu_size n m k =
  if k < 0 then (n * m) - triu_size m n (1 - k)
  else if k >= m then 0
  else
    let mk = Stdlib.min n (m - k) in
    (mk * (mk + 1) / 2) + (mk * (m - k - mk))

let nonzero_pair v ~size =
  match nonzero ~size v with
  | [ i; j ] -> (i, j)
  | _ -> failwith "tri indices: arity"

let tril_indices ?(k = 0) ?m n =
  let m = Option.value m ~default:n in
  let ones = L.full [| n; m |] 1.0 (default_float ()) in
  nonzero_pair (NL.tril ~k ones) ~size:(triu_size m n (-k))

let triu_indices ?(k = 0) ?m n =
  let m = Option.value m ~default:n in
  let ones = L.full [| n; m |] 1.0 (default_float ()) in
  nonzero_pair (NL.triu ~k ones) ~size:(triu_size n m k)

let tril_indices_from ?k a =
  let sh = shape a in
  if Array.length sh <> 2 then invalid_arg "Only 2-D inputs are accepted";
  tril_indices ?k ~m:sh.(1) sh.(0)

let triu_indices_from ?k a =
  let sh = shape a in
  if Array.length sh <> 2 then invalid_arg "Only 2-D inputs are accepted";
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
  let out_dt =
    match dtype_of a with
    | Dtype.Complex128 | Dtype.F64 | Dtype.I64 | Dtype.Uint64 ->
        Dtype.Complex128
    | _ -> Dtype.Complex64
  in
  let sorted =
    Core.bind1
      (Sort { dimension = rank a - 1; is_stable = true; num_keys = 1 })
      [ a ]
  in
  if dtype_of sorted = out_dt then sorted
  else L.convert_element_type sorted out_dt

let sort_complex a0 =
  Eager_jit.one ~jit:"sort_complex" "numpy.extras.sort_complex" () [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sort_complex a0
      | _ -> assert false)

let bitwise_count a = NL.astype (L.population_count (U.absolute a)) (dtype_of a)

let bitwise_count a0 =
  Eager_jit.one ~jit:"bitwise_count" "numpy.extras.bitwise_count" () [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> bitwise_count a0
      | _ -> assert false)

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

let ldexp x1 x2 =
  reject_complex "ldexp" x1;
  (match dtype_of x2 with
  | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.Complex64
  | Dtype.Complex128 ->
      invalid_arg "ldexp not supported for input types"
  | _ -> ());
  let x1 = as_inexact x1 in
  let dt = dtype_of x1 in
  let x2 = L.convert_element_type x2 dt in
  let m, e = frexp x1 in
  let e = NL.astype (U.add (NL.astype e dt) x2) dt in
  let positive () = U.greater e (Jnp_util.py_int 0) in
  let cond = positive () in
  let m = Jnp_util.where cond (U.multiply m (Jnp_util.py_int 2)) m in
  let cond = positive () in
  let e = Jnp_util.where cond (U.subtract e (Jnp_util.py_int 1)) e in
  let scaled =
    U.multiply m (U.power (Jnp_util.py_float 2.0) (NL.astype e dt))
  in
  let special = U.bitwise_or (U.isinf x1) (U.equal x1 (Jnp_util.py_int 0)) in
  Jnp_util.where special x1 scaled

let ldexp a0 a1 =
  Eager_jit.one ~jit:"ldexp" "numpy.extras.ldexp" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> ldexp a0 a1
      | _ -> assert false)

let argpartition ?(axis = -1) a kth =
  ignore kth;
  Sorting.argsort ~axis:(Some axis) a

let apply_along_axis f axis arr =
  let n = rank arr in
  let axis = canonicalize_axis axis n in
  let func = ref (fun xs -> [ f (List.hd xs) ]) in
  for i = 1 to n - axis - 1 do
    let inner = !func in
    func :=
      fun xs ->
        Interpreters.Batching.vmap_flat ~out_axis:(-1) inner [ Some i ] xs
  done;
  for _ = 1 to axis do
    let inner = !func in
    func := fun xs -> Interpreters.Batching.vmap inner [ Some 0 ] xs
  done;
  List.hd (!func [ arr ])

let apply_over_axes f a axes =
  List.fold_left
    (fun acc ax ->
      let ax = canon (rank acc) ax in
      let out = f acc ax in
      if rank out = rank acc then out else NL.expand_dims out [| ax |])
    a axes

type nested = Item of value | Group of nested list

let block tree =
  let rec leaves = function
    | Item v -> [ v ]
    | Group items -> List.concat_map leaves items
  in
  let rec shape_key = function
    | Item _ -> "i"
    | Group items -> "[" ^ String.concat "," (List.map shape_key items) ^ "]"
  in
  Jnp_util.jit1 "block" ~key:(shape_key tree)
    (fun args ->
      let remaining = ref args in
      let rec go = function
        | Item _ ->
            let v = List.hd !remaining in
            remaining := List.tl !remaining;
            (v, 1)
        | Group [] ->
            invalid_arg "jax.numpy.block does not allow empty list arguments"
        | Group items ->
            let parts = List.map go items in
            let depth = snd (List.hd parts) in
            if List.exists (fun (_, d) -> d <> depth) parts then
              invalid_arg "Mismatched list depths in jax.numpy.block";
            let target =
              List.fold_left (fun m (v, _) -> Stdlib.max m (rank v)) depth parts
            in
            let lifted =
              List.map (fun (v, _) -> L.broadcast_to_rank v target) parts
            in
            (NL.concatenate ~axis:(-depth) lifted, depth + 1)
      in
      fst (go tree))
    (leaves tree)

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
  let scale = b1 Types.Div [ scalar 1.0; NL.index_at v 0 ] in
  let q = ref (AC.zeros ~dtype:dt [| Stdlib.max (lu - lv + 1) 1 |]) in
  let r = ref u in
  let start k =
    L.broadcast_in_dim
      (Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int k |]))
      [| 1 |] [||]
  in
  let dims ~window =
    {
      update_window_dims = (if window then [| 0 |] else [||]);
      inserted_window_dims = (if window then [||] else [| 0 |]);
      scatter_dims_to_operand_dims = [| 0 |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
      s_sorted_unique = true;
      s_clip = false;
    }
  in
  for k = 0 to lu - lv do
    let d = b1 Types.Mul [ scale; NL.index_at !r k ] in
    let at = start k in
    q := L.scatter !q at d (dims ~window:false);
    let delta = b1 Types.Mul [ b1 Types.Neg [ d ]; v ] in
    let at = start k in
    r := L.scatter_add !r at delta (dims ~window:true)
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

let polyfit_parts ?rcond ?w x y deg =
  let order = deg + 1 in
  if deg < 0 then invalid_arg "expected deg >= 0";
  let type_error msg = raise (Errors.Jax_type_error msg) in
  if rank x <> 1 then type_error "expected 1D vector for x";
  if size_of x = 0 then type_error "expected non-empty vector for x";
  if rank y < 1 || rank y > 2 then type_error "expected 1D or 2D array for y";
  if (shape x).(0) <> (shape y).(0) then
    type_error "expected x and y to have same length";
  let x = as_inexact x and y = as_inexact y in
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
    | _ -> 1.1920928955078125e-07
  in
  let rcond =
    match rcond with Some r -> r | None -> float_of_int (size_of x) *. eps
  in
  let scale = U.sqrt (R.sum ~axis:[| 0 |] (U.multiply lhs lhs)) in
  let scaled = U.divide lhs (L.broadcast_in_dim scale [| 1; order |] [| 1 |]) in
  let c, resids, rank_, s = Linalg.lstsq ~rcond scaled y in
  let c =
    if rank c > 1 then
      NL.transpose (U.divide (NL.transpose c) (L.expand_dims scale [| 0 |]))
    else U.divide c scale
  in
  (c, resids, rank_, s, rcond, scaled, scale)

let polyfit ?rcond ?w x y deg =
  let c, _, _, _, _, _, _ = polyfit_parts ?rcond ?w x y deg in
  c

let polyfit_full_body ?rcond ?w x y deg =
  let c, resids, rank_, s, rcond, _, _ = polyfit_parts ?rcond ?w x y deg in
  let dt = Dtypes.default_float_dtype () in
  [ c; resids; rank_; s; Concrete (Ndarray.of_floats dt [||] [| rcond |]) ]

let polyfit_full ?rcond ?w a0 a1 a2 =
  match
    Eager_jit.many ~jit:"polyfit" ~vals_first:true "numpy.extras.polyfit_full"
      ("full", rcond, a2) [ w ] [ a0; a1 ] (fun opts vals ->
        match (opts, vals) with
        | [ w ], [ a0; a1 ] -> polyfit_full_body ?rcond ?w a0 a1 a2
        | _ -> assert false)
  with
  | [ c; r; k; s; rc ] -> (c, r, k, s, rc)
  | _ -> assert false

let polyfit_cov_body ?rcond ?w ~unscaled x y deg =
  let c, resids, _, _, _, lhs, scale = polyfit_parts ?rcond ?w x y deg in
  let inv = Linalg.inv (Tensor_contractions.dot (NL.transpose lhs) lhs) in
  let vbase = U.divide inv (Tensor_contractions.outer scale scale) in
  let n = (shape x).(0) in
  let order = deg + 1 in
  let fac =
    if unscaled then Jnp_util.py_float 1.0
    else begin
      if n <= order then
        invalid_arg
          "the number of data points must exceed order to scale the covariance \
           matrix";
      U.divide resids (Jnp_util.py_int (n - order))
    end
  in
  if rank y = 1 then
    let fac = NL.atleast_1d fac in
    (c, U.multiply vbase (L.expand_dims fac [| 0 |]))
  else
    let v3 = L.expand_dims vbase [| 2 |] in
    let fac = NL.atleast_1d fac in
    (c, U.multiply v3 (L.expand_dims fac [| 0; 1 |]))

let polyfit_cov ?rcond ?w ?(unscaled = false) a0 a1 a2 =
  Eager_jit.pair ~jit:"polyfit" ~vals_first:true "numpy.extras.polyfit_cov"
    ("cov", rcond, unscaled, a2) [ w ] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [ w ], [ a0; a1 ] -> polyfit_cov_body ?rcond ?w ~unscaled a0 a1 a2
      | _ -> assert false)

let polyfit ?rcond ?w a0 a1 a2 =
  Eager_jit.one ~jit:"polyfit" ~vals_first:true "numpy.extras.polyfit"
    (rcond, a2) [ w ] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [ w ], [ a0; a1 ] -> polyfit ?rcond ?w a0 a1 a2
      | _ -> assert false)

let roots_of_coefficients p =
  Eager_jit.one "numpy.extras._roots_no_zeros" () [] [ p ] (fun _ vals ->
      let p = List.hd vals in
      let dt = dtype_of p in
      let n = (shape p).(0) in
      let d = n - 1 in
      let ones = L.full [| d - 1 |] 1.0 dt in
      let companion = NL.diag ~k:(-1) ones in
      let tail = L.slice_in_dim ~axis:0 p 1 n in
      let lead = L.index_in_dim ~axis:0 ~keepdims:false p 0 in
      let row = U.divide (U.negative tail) lead in
      let companion =
        L.dynamic_update_index_in_dim ~axis:0 companion row
          (Core.splat Dtype.I32 [||] 0.0)
      in
      Linalg.eigvals companion)

let roots ?(strip_zeros = true) p =
  let p = NL.atleast_1d (as_inexact p) in
  if rank p <> 1 then invalid_arg "Input must be a rank-1 array.";
  let n = (shape p).(0) in
  let real_dt = dtype_of p in
  let complex_dt =
    match real_dt with
    | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
    | _ -> Dtype.Complex64
  in
  if n < 2 then Concrete (Ndarray.of_complex complex_dt [| 0 |] [||])
  else if not strip_zeros then roots_of_coefficients p
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
    if n - !lead < 2 then Concrete (Ndarray.of_complex complex_dt [| 0 |] [||])
    else roots_of_coefficients (L.slice_in_dim ~axis:0 p !lead n)
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
let nan = Float_format.nan

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

let print_options_with ?max_line_width ?precision ?suppress_small () =
  let o = print_options () in
  {
    o with
    Array_print.linewidth = Option.value max_line_width ~default:o.linewidth;
    precision = Option.value precision ~default:o.precision;
    suppress = Option.value suppress_small ~default:o.suppress;
  }

let array_repr ?max_line_width ?precision ?suppress_small a =
  Array_print.array_repr
    ~options:(print_options_with ?max_line_width ?precision ?suppress_small ())
    (host "array_repr" a)

let array_str_jax ?max_line_width ?precision ?suppress_small a =
  Array_print.array_str
    ~options:
      {
        (print_options_with ?max_line_width ?precision ?suppress_small ()) with
        widen_elements = true;
      }
    (host "array_str" a)

let numpy_array_repr ?max_line_width ?precision ?suppress_small a =
  Array_print.numpy_array_repr
    ~options:
      {
        (print_options_with ?max_line_width ?precision ?suppress_small ()) with
        widen_elements = true;
      }
    (host "array_repr" a)

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

let broadcast_shapes = function
  | [] -> [||]
  | shapes -> L.broadcast_shapes shapes

let iterable v = Array.length (shape v) > 0
let complex_ = Scalar_types.complex128

let dtype name =
  match name with
  | "bool" | "bool_" -> Dtype.Bool
  | "int8" -> Dtype.I8
  | "int16" -> Dtype.I16
  | "int32" -> Dtype.I32
  | "int64" -> Dtype.I64
  | "uint8" -> Dtype.Uint8
  | "uint16" -> Dtype.Uint16
  | "uint32" -> Dtype.Uint32
  | "uint64" -> Dtype.Uint64
  | "float16" -> Dtype.F16
  | "float32" -> Dtype.F32
  | "float64" -> Dtype.F64
  | "complex64" -> Dtype.Complex64
  | "complex128" -> Dtype.Complex128
  | other ->
      raise (Invalid_argument ("data type '" ^ other ^ "' not understood"))

type ndarray = value
type ufunc = Ufunc.t
type complex_warning = Complex_warning of string
