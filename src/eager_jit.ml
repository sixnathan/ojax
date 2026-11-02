open Types

let rebuild opts vals_count flat =
  let rec take_opts opts flat =
    match (opts, flat) with
    | [], rest -> ([], rest)
    | None :: tl, rest ->
        let built, rest = take_opts tl rest in
        (None :: built, rest)
    | Some _ :: tl, v :: rest ->
        let built, rest = take_opts tl rest in
        (Some v :: built, rest)
    | Some _ :: _, [] -> invalid_arg "Eager_jit: argument underflow"
  in
  let built, rest = take_opts opts flat in
  if List.length rest <> vals_count then invalid_arg "Eager_jit: argument count";
  (built, rest)

let static_key statics opts =
  Digest.to_hex
    (Digest.string
       (Marshal.to_string
          (statics, List.map Option.is_some opts)
          [ Marshal.Closures ]))

let inline_names =
  let table = Hashtbl.create 128 in
  List.iter
    (fun name -> Hashtbl.replace table name ())
    [
      "nn.functions.sigmoid";
      "nn.functions.relu";
      "nn.functions.relu6";
      "nn.functions.log1mexp";
      "numpy.linalg.pinv";
      "numpy.lax_numpy.ravel";
      "numpy.lax_numpy.swapaxes";
      "numpy.lax_numpy.unstack";
      "numpy.tensor_contractions.inner";
      "numpy.tensor_contractions.matmul";
      "numpy.tensor_contractions.outer";
      "numpy.tensor_contractions.vdot";
      "numpy.tensor_contractions.dot";
      "numpy.ufuncs.abs";
      "numpy.ufuncs.absolute";
      "numpy.ufuncs.acos";
      "numpy.ufuncs.acosh";
      "numpy.ufuncs.arccos";
      "numpy.ufuncs.arcsin";
      "numpy.ufuncs.arcsinh";
      "numpy.ufuncs.arctan";
      "numpy.ufuncs.arctan2";
      "numpy.ufuncs.arctanh";
      "numpy.ufuncs.asin";
      "numpy.ufuncs.asinh";
      "numpy.ufuncs.atan";
      "numpy.ufuncs.atan2";
      "numpy.ufuncs.atanh";
      "numpy.ufuncs.bitwise_invert";
      "numpy.ufuncs.bitwise_left_shift";
      "numpy.ufuncs.bitwise_not";
      "numpy.ufuncs.bitwise_right_shift";
      "numpy.ufuncs.cbrt";
      "numpy.ufuncs.ceil";
      "numpy.ufuncs.conjugate";
      "numpy.ufuncs.cos";
      "numpy.ufuncs.cosh";
      "numpy.ufuncs.deg2rad";
      "numpy.ufuncs.equal";
      "numpy.ufuncs.exp";
      "numpy.ufuncs.exp2";
      "numpy.ufuncs.expm1";
      "numpy.ufuncs.fabs";
      "numpy.ufuncs.float_power";
      "numpy.ufuncs.floor";
      "numpy.ufuncs.greater";
      "numpy.ufuncs.greater_equal";
      "numpy.ufuncs.imag";
      "numpy.ufuncs.invert";
      "numpy.ufuncs.isfinite";
      "numpy.ufuncs.isnan";
      "numpy.ufuncs.left_shift";
      "numpy.ufuncs.less";
      "numpy.ufuncs.less_equal";
      "numpy.ufuncs.log";
      "numpy.ufuncs.log10";
      "numpy.ufuncs.log1p";
      "numpy.ufuncs.log2";
      "numpy.ufuncs.logical_not";
      "numpy.ufuncs.nextafter";
      "numpy.ufuncs.not_equal";
      "numpy.ufuncs.positive";
      "numpy.ufuncs.rad2deg";
      "numpy.ufuncs.real";
      "numpy.ufuncs.reciprocal";
      "numpy.ufuncs.right_shift";
      "numpy.ufuncs.sign";
      "numpy.ufuncs.sin";
      "numpy.ufuncs.sinh";
      "numpy.ufuncs.spacing";
      "numpy.ufuncs.sqrt";
      "numpy.ufuncs.square";
      "numpy.ufuncs.tan";
      "numpy.ufuncs.tanh";
      "numpy.ufuncs.true_divide";
      "numpy.ufuncs.divide";
      "numpy.ufuncs.power";
      "numpy.ufuncs.add";
      "numpy.ufuncs.bitwise_and";
      "numpy.ufuncs.bitwise_or";
      "numpy.ufuncs.bitwise_xor";
      "numpy.ufuncs.logaddexp";
      "numpy.ufuncs.logaddexp2";
      "numpy.ufuncs.logical_and";
      "numpy.ufuncs.logical_or";
      "numpy.ufuncs.logical_xor";
      "numpy.ufuncs.maximum";
      "numpy.ufuncs.minimum";
      "numpy.ufuncs.multiply";
      "numpy.ufuncs.negative";
      "numpy.ufuncs.subtract";
      "numpy.reductions.all";
      "numpy.reductions.amax";
      "numpy.reductions.amin";
      "numpy.reductions.any";
      "numpy.reductions.average";
      "numpy.reductions.max";
      "numpy.reductions.mean";
      "numpy.reductions.min";
      "numpy.reductions.prod";
      "numpy.reductions.sum";
      "numpy.lax_numpy.argmax";
      "numpy.lax_numpy.argmin";
      "numpy.lax_numpy.moveaxis";
      "numpy.lax_numpy.squeeze";
      "numpy.einsum._einsum";
    ];
  table

let short_name name =
  match String.rindex_opt name '.' with
  | Some i -> String.sub name (i + 1) (String.length name - i - 1)
  | None -> name

let many ?jit ?(vals_first = false) name statics opts vals f =
  let present = List.filter_map Fun.id opts in
  let split flat =
    if vals_first then
      let n = List.length vals in
      let front = List.filteri (fun i _ -> i < n) flat in
      let back = List.filteri (fun i _ -> i >= n) flat in
      rebuild opts n (back @ front)
    else rebuild opts (List.length vals) flat
  in
  let args = if vals_first then vals @ present else present @ vals in
  List.iter Core.check_not_deleted args;
  match jit with
  | Some jname when Core.top_is_staging args ->
      Jaxpr.named_jit ~name:jname ~key:(static_key statics opts)
        (fun flat ->
          let opts, vals = split flat in
          f opts vals)
        args
  | _ ->
      let outs =
        Backend.jitted_keyed
          ~inline:(Hashtbl.mem inline_names name)
          (name, Obj.repr (statics, List.map Option.is_some opts))
          (fun flat ->
            let opts, vals = split flat in
            f opts vals)
          args
      in
      if Debug_special.needs_check () && not (Core.top_is_staging args) then
        Debug_special.check_jit ~name:(short_name name)
          ~rerun:(fun () ->
            let opts, vals = split args in
            ignore (f opts vals))
          (List.filter_map
             (function
               | Concrete a -> Some a
               | Device b -> Some (Pjrt.Buffer.to_host b)
               | Tracer _ -> None)
             outs);
      outs

let one ?jit ?vals_first name statics opts vals f =
  match many ?jit ?vals_first name statics opts vals (fun o v -> [ f o v ]) with
  | [ out ] -> out
  | _ -> failwith "Eager_jit: expected one result"

let pair ?jit ?vals_first name statics opts vals f =
  match
    many ?jit ?vals_first name statics opts vals (fun o v ->
        let a, b = f o v in
        [ a; b ])
  with
  | [ a; b ] -> (a, b)
  | _ -> failwith "Eager_jit: expected two results"
