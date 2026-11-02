type 'a flag = { name : string; mutable value : 'a; validate : 'a -> unit }

let make ?(validate = ignore) name value = { name; value; validate }
let value flag = flag.value
let name flag = flag.name

let set flag v =
  flag.validate v;
  flag.value <- v

let with_value flag v thunk =
  flag.validate v;
  let prev = flag.value in
  flag.value <- v;
  Fun.protect ~finally:(fun () -> flag.value <- prev) thunk

let py_list values =
  "[" ^ String.concat ", " (List.map (Printf.sprintf "'%s'") values) ^ "]"

let optional_enum values = function
  | None -> ()
  | Some v ->
      if not (List.mem v values) then
        invalid_arg
          (Printf.sprintf
             "new enum value must be None or in %s, got %s of type <class \
              'str'>."
             (py_list values) v)

let enum_class class_name values v =
  if not (List.mem v values) then
    invalid_arg (Printf.sprintf "'%s' is not a valid %s" v class_name)

let enable_x64 = make "jax_enable_x64" false
let x64_enabled () = enable_x64.value
let disable_jit = make "jax_disable_jit" false
let debug_nans = make "jax_debug_nans" false
let debug_infs = make "jax_debug_infs" false
let log_compiles = make "jax_log_compiles" false
let no_tracing = make "jax_no_tracing" false
let no_execution = make "jax_no_execution" false
let numpy_rank_promotion = make "jax_numpy_rank_promotion" "allow"
let enable_checks = make "jax_enable_checks" false
let debug_key_reuse = make "jax_debug_key_reuse" false
let check_tracer_leaks = make "jax_check_tracer_leaks" false
let explain_cache_misses = make "jax_explain_cache_misses" false
let enable_custom_prng = make "jax_enable_custom_prng" false
let softmax_custom_jvp = make "jax_softmax_custom_jvp" false
let threefry_partitionable = make "jax_threefry_partitionable" true
let allow_f16_reductions = make "jax_allow_f16_reductions" true
let thread_guard = make "jax_thread_guard" false
let auto_pcast = make "jax_auto_pcast" true
let random_seed_offset = make "jax_random_seed_offset" 0
let high_dynamic_range_gumbel = make "jax_high_dynamic_range_gumbel" false

let remove_size_one_mesh_axis_from_type =
  make "jax_remove_size_one_mesh_axis_from_type" false

let jax2tf_associative_scan_reductions =
  make "jax2tf_associative_scan_reductions" false

let numpy_dtype_promotion =
  make
    ~validate:(enum_class "NumpyDtypePromotion" [ "standard"; "strict" ])
    "jax_numpy_dtype_promotion" "standard"

let legacy_prng_key =
  make
    ~validate:(enum_class "LegacyPrngKeyState" [ "allow"; "warn"; "error" ])
    "jax_legacy_prng_key" "allow"

let matmul_levels =
  [
    ("default", "DEFAULT");
    ("high", "HIGH");
    ("highest", "HIGHEST");
    ("bfloat16", "DEFAULT");
    ("tensorfloat32", "HIGH");
    ("float32", "HIGHEST");
  ]

let matmul_presets =
  [
    "ANY_F8_ANY_F8_F32";
    "ANY_F8_ANY_F8_F32_FAST_ACCUM";
    "ANY_F8_ANY_F8_ANY";
    "ANY_F8_ANY_F8_ANY_FAST_ACCUM";
    "F16_F16_F16";
    "F16_F16_F32";
    "BF16_BF16_BF16";
    "BF16_BF16_F32";
    "BF16_BF16_F32_X3";
    "BF16_BF16_F32_X6";
    "BF16_BF16_F32_X9";
    "TF32_TF32_F32";
    "TF32_TF32_F32_X3";
    "F32_F32_F32";
    "F64_F64_F64";
  ]

let default_matmul_precision =
  make
    ~validate:(optional_enum (List.map fst matmul_levels @ matmul_presets))
    "jax_default_matmul_precision" None

let matmul_level () =
  match default_matmul_precision.value with
  | None -> None
  | Some name -> (
      match List.assoc_opt name matmul_levels with
      | Some level -> Some level
      | None -> Some "DEFAULT")

let matmul_preset () =
  match default_matmul_precision.value with
  | Some name when List.mem name matmul_presets -> Some name
  | _ -> None

let cpu_matmul_presets =
  [
    "F16_F16_F16";
    "F32_F32_F32";
    "F64_F64_F64";
    "BF16_BF16_F32";
    "BF16_BF16_F32_X3";
    "BF16_BF16_F32_X6";
  ]

let check_matmul_preset_on_cpu () =
  match matmul_preset () with
  | Some name when not (List.mem name cpu_matmul_presets) ->
      invalid_arg
        (Printf.sprintf
           "The precision '%s' is not supported by dot_general on CPU" name)
  | _ -> ()

let guard_levels =
  [ "allow"; "log"; "disallow"; "log_explicit"; "disallow_explicit" ]

let transfer_flag name = make ~validate:(optional_enum guard_levels) name None

let transfer_guard_host_to_device =
  transfer_flag "jax_transfer_guard_host_to_device"

let transfer_guard_device_to_device =
  transfer_flag "jax_transfer_guard_device_to_device"

let transfer_guard_device_to_host =
  transfer_flag "jax_transfer_guard_device_to_host"

let with_transfer_guard level thunk =
  with_value transfer_guard_host_to_device (Some level) (fun () ->
      with_value transfer_guard_device_to_device (Some level) (fun () ->
          with_value transfer_guard_device_to_host (Some level) thunk))

let array_garbage_collection_guard =
  make
    ~validate:(optional_enum [ "allow"; "log"; "fatal" ])
    "jax_array_garbage_collection_guard" None

let log_sink : (string -> unit) ref = ref prerr_endline
let log message = !log_sink message

let set_log_sink sink =
  let prev = !log_sink in
  log_sink := sink;
  prev

let py_bool b = if b then "True" else "False"
let py_opt = function None -> "None" | Some s -> s
let user_contexts : (string * (unit -> string)) list ref = ref []

let trace_context () =
  List.map (fun (name, read) -> (name, read ())) !user_contexts
  @ [
      (auto_pcast.name, py_bool auto_pcast.value);
      ( remove_size_one_mesh_axis_from_type.name,
        py_bool remove_size_one_mesh_axis_from_type.value );
      (enable_x64.name, py_bool enable_x64.value);
      (debug_key_reuse.name, py_bool debug_key_reuse.value);
      (numpy_dtype_promotion.name, numpy_dtype_promotion.value);
      (numpy_rank_promotion.name, numpy_rank_promotion.value);
      (default_matmul_precision.name, py_opt default_matmul_precision.value);
      (allow_f16_reductions.name, py_bool allow_f16_reductions.value);
      (softmax_custom_jvp.name, py_bool softmax_custom_jvp.value);
      (threefry_partitionable.name, py_bool threefry_partitionable.value);
      (random_seed_offset.name, string_of_int random_seed_offset.value);
      (high_dynamic_range_gumbel.name, py_bool high_dynamic_range_gumbel.value);
    ]

let export_calling_convention_version =
  make "jax_export_calling_convention_version" 10

let platforms =
  make "jax_platforms"
    (Option.value (Sys.getenv_opt "OJAX_PLATFORMS") ~default:"")

let platform_resolver = ref (fun () -> "cpu")
let current_platform () = !platform_resolver ()
let num_cpu_devices = make "jax_num_cpu_devices" (-1)

type 'a user_context = 'a option flag

let make_user_context ?default_value () =
  let id = List.length !user_contexts in
  let flag = make (Printf.sprintf "user_context_%d" id) default_value in
  let read () =
    Digest.to_hex
      (Digest.string (Marshal.to_string flag.value [ Marshal.Closures ]))
  in
  user_contexts := !user_contexts @ [ (flag.name, read) ];
  flag

let with_user_context ctx v thunk = with_value ctx (Some v) thunk
let user_context_value ctx = ctx.value

type config_value =
  | Config_bool of bool
  | Config_int of int
  | Config_string of string
  | Config_optional_string of string option

let bool_options =
  List.map
    (fun f -> (f.name, f))
    [
      enable_x64;
      disable_jit;
      debug_nans;
      debug_infs;
      log_compiles;
      no_tracing;
      no_execution;
      enable_checks;
      debug_key_reuse;
      check_tracer_leaks;
      explain_cache_misses;
      enable_custom_prng;
      softmax_custom_jvp;
      threefry_partitionable;
      allow_f16_reductions;
      thread_guard;
      jax2tf_associative_scan_reductions;
      auto_pcast;
      remove_size_one_mesh_axis_from_type;
      high_dynamic_range_gumbel;
    ]

let string_options =
  List.map
    (fun f -> (f.name, f))
    [ numpy_rank_promotion; numpy_dtype_promotion; legacy_prng_key; platforms ]

let optional_options =
  List.map
    (fun f -> (f.name, f))
    [
      default_matmul_precision;
      transfer_guard_host_to_device;
      transfer_guard_device_to_device;
      transfer_guard_device_to_host;
      array_garbage_collection_guard;
    ]

let int_options =
  List.map
    (fun f -> (f.name, f))
    [ export_calling_convention_version; num_cpu_devices; random_seed_offset ]

let unrecognized name =
  raise (Invalid_argument ("Unrecognized config option: " ^ name))

let read name =
  match List.assoc_opt name bool_options with
  | Some f -> Config_bool f.value
  | None -> (
      match List.assoc_opt name string_options with
      | Some f -> Config_string f.value
      | None -> (
          match List.assoc_opt name optional_options with
          | Some f -> Config_optional_string f.value
          | None -> (
              match List.assoc_opt name int_options with
              | Some f -> Config_int f.value
              | None -> unrecognized name)))

let update name v =
  match v with
  | Config_bool b -> (
      match List.assoc_opt name bool_options with
      | Some f -> set f b
      | None -> unrecognized name)
  | Config_int i -> (
      match List.assoc_opt name int_options with
      | Some f -> set f i
      | None -> unrecognized name)
  | Config_string s -> (
      match
        ( List.assoc_opt name string_options,
          List.assoc_opt name optional_options )
      with
      | Some f, _ -> set f s
      | None, Some f -> set f (Some s)
      | None, None -> unrecognized name)
  | Config_optional_string s -> (
      match List.assoc_opt name optional_options with
      | Some f -> set f s
      | None -> unrecognized name)
