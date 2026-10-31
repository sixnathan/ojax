open Types

type layout = Default | Major_to_minor of int array

type attr = ffi_attr =
  | Ffi_bool of bool
  | Ffi_int of int64
  | Ffi_float of float
  | Ffi_string of string
  | Ffi_array of Ndarray.t
  | Ffi_dict of (string * ffi_attr) list
  | Ffi_seq of ffi_attr list
  | Ffi_dtype of Dtype.t

type library = nativeint
type type_registration = { type_id : nativeint; type_info : nativeint option }

let dlopen = Pjrt.Abi.ffi_dlopen
let dlsym = Pjrt.Abi.ffi_dlsym

let xla_platform_name = function
  | "cpu" -> "Host"
  | "gpu" -> "CUDA"
  | other -> other

let pending_targets : (string * string * nativeint) list ref = ref []
let pending_types : (string * string * type_registration) list ref = ref []
let host_platform = "Host"

let register_ffi_target ?(platform = "cpu") ?(api_version = 1) name fn =
  if api_version <> 1 then
    invalid_arg
      (Printf.sprintf
         "register_ffi_target: api_version=%d is not available through the \
          PJRT FFI extension; only the typed FFI (api_version=1) can be \
          registered"
         api_version);
  let xla = xla_platform_name platform in
  if xla = host_platform then
    Pjrt.Abi.ffi_register_handler (Backend.xla_plugin ()) name xla fn
  else pending_targets := (name, xla, fn) :: !pending_targets

let register_ffi_type_id ?platform:_ _name _obj =
  invalid_arg "register_ffi_type_id is not supported after jaxlib version 381."

let register_ffi_type ?(platform = "cpu") name registration =
  let xla = xla_platform_name platform in
  if xla = host_platform then
    Pjrt.Abi.ffi_register_type (Backend.xla_plugin ()) name registration.type_id
      (Option.value registration.type_info ~default:0n)
  else pending_types := (name, xla, registration) :: !pending_types

let allowed_vmap_methods =
  [
    "sequential";
    "sequential_unrolled";
    "expand_dims";
    "broadcast_all";
    "legacy_vectorized";
  ]

let aval_string (a : aval) =
  Printf.sprintf "%s%s[%s]"
    (if a.weak_type then "~" else "")
    (Dtype.numpy_name a.dtype)
    (String.concat "," (Array.to_list (Array.map string_of_int a.shape)))

let layout_string l =
  match Array.to_list l with
  | [ x ] -> Printf.sprintf "(%d,)" x
  | xs -> "(" ^ String.concat ", " (List.map string_of_int xs) ^ ")"

let default_layout (a : aval) =
  let n = Array.length a.shape in
  Array.init n (fun i -> n - 1 - i)

let lowering_layout a = function
  | Default -> default_layout a
  | Major_to_minor m ->
      let n = Array.length m in
      Array.init n (fun i -> m.(n - 1 - i))

let sorted_attrs attributes =
  List.sort (fun (a, _) (b, _) -> String.compare a b) attributes

let make_call ~has_side_effect ~vmap_method ~input_layouts ~output_layouts
    ~input_output_aliases ~custom_call_api_version ~legacy_backend_config target
    result_avals =
  (match vmap_method with
  | Some m when not (List.mem m allowed_vmap_methods) ->
      invalid_arg
        (Printf.sprintf
           "vmap_method must be on of the allowed methods ['sequential', \
            'sequential_unrolled', 'expand_dims', 'broadcast_all', \
            'legacy_vectorized', None], but got: %s"
           m)
  | _ -> ());
  if custom_call_api_version >= 4 && legacy_backend_config <> None then
    invalid_arg
      (Printf.sprintf
         "The use of the legacy_backend_config parameter requires \
          custom_call_api_version < 4; got %d."
         custom_call_api_version);
  fun ?(attributes = []) args ->
    let in_avals = List.map Core.get_aval args in
    let static_inputs =
      match input_layouts with
      | None -> List.map default_layout in_avals
      | Some layouts ->
          if List.length layouts <> List.length in_avals then
            invalid_arg
              (Printf.sprintf
                 "The number of input arguments (%d) must equal the number of \
                  input layouts (%d)."
                 (List.length in_avals) (List.length layouts));
          List.map2 lowering_layout in_avals layouts
    in
    let static_outputs =
      match output_layouts with
      | None -> List.map default_layout result_avals
      | Some layouts ->
          if List.length layouts <> List.length result_avals then
            invalid_arg
              (Printf.sprintf
                 "The number of outputs (%d) must equal the number of output \
                  layouts (%d)."
                 (List.length result_avals) (List.length layouts));
          List.map2 lowering_layout result_avals layouts
    in
    let nargs = List.length args and nres = List.length result_avals in
    let aliases = List.sort compare input_output_aliases in
    List.iter
      (fun (i, o) ->
        let prefix =
          Printf.sprintf "input_output_aliases contains the mapping '%d:%d'" i o
        in
        if i >= nargs then
          invalid_arg
            (Printf.sprintf "%s with input index %d outside the range [0, %d)."
               prefix i nargs);
        if o >= nres then
          invalid_arg
            (Printf.sprintf "%s with output index %d outside the range [0, %d)."
               prefix o nres);
        let in_aval = List.nth in_avals i
        and out_aval = List.nth result_avals o in
        if in_aval.shape <> out_aval.shape || in_aval.dtype <> out_aval.dtype
        then
          invalid_arg
            (Printf.sprintf
               "%s referring to an input with abstract value %s and an output \
                with a different abstract value %s."
               prefix (aval_string in_aval) (aval_string out_aval));
        let il = List.nth static_inputs i and ol = List.nth static_outputs o in
        if il <> ol then
          invalid_arg
            (Printf.sprintf
               "%s referring to an input with layout %s and an output with a \
                different layout %s."
               prefix (layout_string il) (layout_string ol)))
      aliases;
    if custom_call_api_version < 4 && attributes <> [] then
      invalid_arg
        (Printf.sprintf
           "The use of ffi_call attributes requires a custom call API version \
            of at least 4; got api_version=%d."
           custom_call_api_version);
    Core.bind
      (Ffi_call
         {
           ffi_target = target;
           ffi_result_avals = result_avals;
           ffi_vmap_method = vmap_method;
           ffi_has_side_effect = has_side_effect;
           ffi_input_layouts = static_inputs;
           ffi_output_layouts = static_outputs;
           ffi_aliases = aliases;
           ffi_api_version = custom_call_api_version;
           ffi_legacy_backend_config = legacy_backend_config;
           ffi_attributes = sorted_attrs attributes;
         })
      args

let ffi_call ?(has_side_effect = false) ?vmap_method ?input_layouts
    ?output_layouts ?(input_output_aliases = []) ?(custom_call_api_version = 4)
    ?legacy_backend_config target result_avals =
  make_call ~has_side_effect ~vmap_method ~input_layouts ~output_layouts
    ~input_output_aliases ~custom_call_api_version ~legacy_backend_config target
    result_avals

let ffi_call1 ?has_side_effect ?vmap_method ?input_layouts ?output_layout
    ?input_output_aliases ?custom_call_api_version ?legacy_backend_config target
    result_aval =
  let call =
    ffi_call ?has_side_effect ?vmap_method ?input_layouts
      ?output_layouts:(Option.map (fun l -> [ l ]) output_layout)
      ?input_output_aliases ?custom_call_api_version ?legacy_backend_config
      target [ result_aval ]
  in
  fun ?attributes args -> List.hd (call ?attributes args)

let eager_impl prim inputs =
  let avals =
    List.map
      (fun nd ->
        {
          shape = Ndarray.shape nd;
          dtype = Ndarray.dtype nd;
          weak_type = false;
        })
      inputs
  in
  let closed = Jaxpr.make_jaxpr avals (fun args -> Core.bind prim args) in
  let text = Stablehlo.Emit.emit_closed_jaxpr closed in
  let outs =
    Backend.run_text_forced text (List.map (fun nd -> Concrete nd) inputs)
  in
  List.map
    (function
      | Concrete nd -> nd
      | Device b -> Pjrt.Buffer.to_host b
      | Tracer _ -> failwith "ffi_call: unexpected tracer result")
    outs

let () = Core.ffi_impl := eager_impl
