open Types

module type S = sig
  type buffer
  type compiled

  val of_host : Ndarray.t -> buffer
  val to_host : buffer -> Ndarray.t
  val compile : closed_jaxpr -> compiled
  val execute : compiled -> buffer list -> buffer list
end

module Interpreter = struct
  type buffer = Ndarray.t
  type compiled = closed_jaxpr

  let of_host nd = nd
  let to_host nd = nd
  let compile cj = cj

  let concrete = function
    | Concrete nd -> nd
    | Device b -> Pjrt.Buffer.to_host b
    | Tracer _ -> failwith "Backend.Interpreter: unexpected tracer result"

  let execute cj bufs =
    List.map concrete
      (Jaxpr.eval_closed_jaxpr cj (List.map (fun nd -> Concrete nd) bufs))
end

let nvidia_device_nodes = [ "/dev/nvidia0"; "/dev/nvidiactl"; "/dev/dxg" ]
let has_visible_nvidia_gpu () = List.exists Sys.file_exists nvidia_device_nodes
let cudnn_library = "libcudnn.so.9"
let cudnn_loadable () = Pjrt.Abi.library_loadable cudnn_library

let gpu_ready () =
  Pjrt.Discover.gpu_configured ()
  && has_visible_nvidia_gpu () && cudnn_loadable ()

let check_gpu_runtime () =
  ignore (Pjrt.Discover.gpu_preflight ());
  if not (cudnn_loadable ()) then
    raise
      (Pjrt.Discover.Error
         (Printf.sprintf
            "Unable to load cuDNN. Is it installed? %s must be on              \
             LD_LIBRARY_PATH for the cuda platform"
            cudnn_library))

let gpu_client_options () =
  let open Pjrt.Client in
  let env name default =
    match Sys.getenv_opt name with Some "" | None -> default | Some v -> v
  in
  let allocator =
    String.lowercase_ascii (env "XLA_PYTHON_CLIENT_ALLOCATOR" "default")
  in
  if
    not
      (List.mem allocator [ "default"; "platform"; "bfc"; "cuda_async"; "vmm" ])
  then
    invalid_arg
      (Printf.sprintf
         "XLA_PYTHON_CLIENT_ALLOCATOR env var must be \"default\",           \
          \"platform\", \"bfc\", \"cuda_async\", or \"vmm\", got \"%s\""
         allocator);
  let memory_fraction =
    match
      (env "XLA_CLIENT_MEM_FRACTION" "", env "XLA_PYTHON_CLIENT_MEM_FRACTION" "")
    with
    | "", "" -> []
    | v, "" | "", v -> [ ("memory_fraction", Float (float_of_string v)) ]
    | _, _ ->
        invalid_arg
          "XLA_CLIENT_MEM_FRACTION is specified together with            \
           XLA_PYTHON_CLIENT_MEM_FRACTION. Remove the latter one, it \
           is            deprecated."
  in
  let preallocate =
    match env "XLA_PYTHON_CLIENT_PREALLOCATE" "" with
    | "" -> []
    | v ->
        [ ("preallocate", Bool (not (List.mem v [ "false"; "False"; "0" ]))) ]
  in
  let collective_memory =
    match env "XLA_PYTHON_CLIENT_COLLECTIVE_MEM_SIZE_MB" "" with
    | "" -> []
    | v ->
        [
          ( "collective_memory_size",
            Int64 (Int64.mul (Int64.of_string v) 1048576L) );
        ]
  in
  let flag name = int_of_string (env name "0") <> 0 in
  [ ("platform_name", String "cuda"); ("allocator", String allocator) ]
  @ memory_fraction @ preallocate @ collective_memory
  @ [
      ( "abort_collectives_on_failure",
        Bool (flag "XLA_PYTHON_CLIENT_ABORT_COLLECTIVES_ON_FAILURE") );
      ( "use_tfrt_gpu_client",
        Bool (flag "XLA_PYTHON_CLIENT_USE_TFRT_GPU_CLIENT") );
    ]

let expand_platform = function
  | "gpu" | "cuda" -> "cuda"
  | "cpu" -> "cpu"
  | other ->
      invalid_arg
        (Printf.sprintf
           "Unknown backend %s: ojax supports the platforms cpu, cuda and gpu"
           other)

let requested_platforms () =
  match Config.value Config.platforms with
  | "" -> None
  | names -> Some (List.map expand_platform (String.split_on_char ',' names))

let check_requested = function
  | "cuda" ->
      if has_visible_nvidia_gpu () then (
        check_gpu_runtime ();
        true)
      else false
  | _ -> true

let resolve_default () =
  match requested_platforms () with
  | Some names -> (
      match List.filter check_requested names with
      | first :: _ -> first
      | [] ->
          failwith
            (Printf.sprintf "No visible device for the requested platforms %s"
               (Config.value Config.platforms)))
  | None ->
      if Pjrt.Discover.gpu_configured () && has_visible_nvidia_gpu () then (
        check_gpu_runtime ();
        "cuda")
      else "cpu"

let default_platform = lazy (resolve_default ())
let override = ref None

let current_platform () =
  match !override with Some p -> p | None -> Lazy.force default_platform

let with_platform name thunk =
  let platform = expand_platform name in
  if platform = "cuda" then check_gpu_runtime ();
  let previous = !override in
  override := Some platform;
  Fun.protect ~finally:(fun () -> override := previous) thunk

let () = Config.platform_resolver := current_platform

module Xla = struct
  type buffer = Pjrt.Buffer.t
  type compiled = Pjrt.Executable.t

  let plugins = Hashtbl.create 2

  let plugin_for platform =
    match Hashtbl.find_opt plugins platform with
    | Some plugin -> plugin
    | None ->
        let path =
          match platform with
          | "cuda" -> Pjrt.Discover.gpu_preflight ()
          | _ -> Pjrt.Discover.preflight ()
        in
        let plugin = Pjrt.Abi.open_plugin path in
        Hashtbl.replace plugins platform plugin;
        plugin

  let clients = Hashtbl.create 2

  let client_for platform =
    match Hashtbl.find_opt clients platform with
    | Some client -> client
    | None ->
        let plugin = plugin_for platform in
        let client =
          match platform with
          | "cuda" -> Pjrt.Client.create ~options:(gpu_client_options ()) plugin
          | _ -> Pjrt.Client.create plugin
        in
        Hashtbl.replace clients platform client;
        client

  let client () = client_for (current_platform ())
  let of_host nd = Pjrt.Buffer.of_host (client ()) nd
  let to_host buf = Pjrt.Buffer.to_host buf

  let compile_text ?options text =
    let start = Sys.time () in
    let exec = Pjrt.Executable.compile ?options (client ()) text in
    Monitoring.record_event_duration_secs
      ~kwargs:[ ("fun_name", Monitoring.Text "jit") ]
      "/jax/core/compile/backend_compile_duration"
      (Sys.time () -. start);
    exec

  let compile cj = compile_text (Stablehlo.Emit.emit_closed_jaxpr cj)

  let execute exec bufs =
    Array.to_list (Pjrt.Executable.execute exec (Array.of_list bufs))
end

let plugin_available () =
  try
    ignore (Pjrt.Discover.preflight ());
    true
  with Pjrt.Discover.Error _ -> false

let backend_mode =
  lazy
    (match Sys.getenv_opt "OJAX_BACKEND" with
    | Some "interpreter" -> `Interpreter
    | Some "xla" -> `Xla
    | None -> `Auto
    | Some other ->
        invalid_arg
          (Printf.sprintf
             "OJAX_BACKEND: unknown backend %S (expected \"interpreter\" or \
              \"xla\")"
             other))

let cpu_uses_xla = lazy (plugin_available ())

let use_xla () =
  match (Lazy.force backend_mode, current_platform ()) with
  | `Interpreter, "cuda" ->
      invalid_arg "OJAX_BACKEND=interpreter cannot run the cuda platform"
  | `Interpreter, _ -> false
  | `Xla, _ | `Auto, "cuda" -> true
  | `Auto, _ -> Lazy.force cpu_uses_xla

let device_of_buffer buffer =
  Hashtbl.fold
    (fun platform client found ->
      if Pjrt.Buffer.belongs_to buffer client then Some (platform ^ ":0")
      else found)
    Xla.clients None

let value_device = function
  | Device buffer -> device_of_buffer buffer
  | Concrete _ | Tracer _ -> None

let default_backend () =
  match current_platform () with "cuda" -> "gpu" | platform -> platform

let devices ?backend () =
  let platform =
    match backend with
    | Some name -> expand_platform name
    | None -> current_platform ()
  in
  match platform with
  | "cuda" ->
      List.init
        (Pjrt.Client.device_count (Xla.client_for "cuda"))
        (Printf.sprintf "cuda:%d")
  | _ -> [ "cpu:0" ]

let platform_version () =
  if use_xla () then Pjrt.Client.platform_version (Xla.client ())
  else "interpreter"

let device_kind () =
  if use_xla () then Pjrt.Client.device_kind (Xla.client ()) else "cpu"

let is_device = function Device _ -> true | Concrete _ | Tracer _ -> false
let is_hostable = function Concrete _ | Device _ -> true | Tracer _ -> false
let runtime_error reason = raise (Errors.Jax_runtime_error reason)

let host_to_device_text nd =
  Printf.sprintf
    "host-to-device transfer: aval=ShapedArray(%s[%s]), \
     dst_sharding=SingleDeviceSharding(device=CpuDevice(id=0), \
     memory_kind=device)"
    (Dtype.numpy_name (Ndarray.dtype nd))
    (String.concat ","
       (List.map string_of_int (Array.to_list (Ndarray.shape nd))))

let guard_host_to_device ~explicit nd =
  let disallow () =
    raise
      (Errors.Jax_runtime_error
         ("INVALID_ARGUMENT: Disallowed " ^ host_to_device_text nd))
  in
  match Config.value Config.transfer_guard_host_to_device with
  | None | Some "allow" -> ()
  | Some "log" -> if not explicit then Config.log (host_to_device_text nd)
  | Some "disallow" -> if not explicit then disallow ()
  | Some "log_explicit" -> Config.log (host_to_device_text nd)
  | Some _ -> disallow ()

let guard_implicit_args args =
  List.iter
    (function
      | Concrete nd -> guard_host_to_device ~explicit:false nd
      | Device _ | Tracer _ -> ())
    args

let is_live b =
  match Pjrt.Buffer.dimensions b with
  | (_ : int array) -> true
  | exception _ -> false

let on_garbage_collected b =
  if is_live b then
    match Config.value Config.array_garbage_collection_guard with
    | None | Some "allow" -> ()
    | Some level ->
        Config.log
          "`jax.Array` was garbage collected before it was explicitly destroyed";
        if level = "fatal" then Unix.kill (Unix.getpid ()) Sys.sigabrt

let track_device b =
  Gc.finalise on_garbage_collected b;
  Device b

let to_input = function
  | Device b when Pjrt.Buffer.belongs_to b (Xla.client ()) -> (b, false)
  | Device b -> (Xla.of_host (Pjrt.Buffer.to_host b), true)
  | Concrete nd -> (Xla.of_host nd, true)
  | Tracer _ -> failwith "Backend: xla execution requires concrete arguments"

let xla_run exec args resident =
  let inputs = List.map to_input args in
  Fun.protect
    ~finally:(fun () ->
      List.iter (fun (b, owned) -> if owned then Pjrt.Buffer.destroy b) inputs)
    (fun () ->
      let outs =
        try Xla.execute exec (List.map fst inputs)
        with Pjrt.Abi.Error reason -> runtime_error reason
      in
      if resident then List.map track_device outs
      else
        Fun.protect
          ~finally:(fun () -> List.iter Pjrt.Buffer.destroy outs)
          (fun () -> List.map (fun b -> Concrete (Xla.to_host b)) outs))

let compiled_on_xla = ref 0
let host_fallbacks : string list ref = ref []
let stats () = (!compiled_on_xla, List.rev !host_fallbacks)

let mentions_unsupported_type text =
  let n = String.length text in
  let rec scan i =
    i + 3 <= n && (String.sub text i 3 = "f6E" || scan (i + 1))
  in
  scan 0

let emit_timed cj =
  let start = Sys.time () in
  let text = Stablehlo.Emit.emit_closed_jaxpr cj in
  Monitoring.record_event_duration_secs
    ~kwargs:[ ("fun_name", Monitoring.Text "jit") ]
    "/jax/core/compile/jaxpr_to_mlir_module_duration"
    (Sys.time () -. start);
  text

let executor ?options (cj : closed_jaxpr) : value list -> value list =
  let precision = Config.value Config.default_matmul_precision in
  let per_platform = Hashtbl.create 2 in
  let compiled_for platform =
    match Hashtbl.find_opt per_platform platform with
    | Some compiled -> compiled
    | None ->
        let compiled =
          Config.with_value Config.default_matmul_precision precision (fun () ->
              match emit_timed cj with
              | text when mentions_unsupported_type text ->
                  host_fallbacks :=
                    "f6 dtypes have no XLA type" :: !host_fallbacks;
                  None
              | text -> (
                  incr compiled_on_xla;
                  match Xla.compile_text ?options text with
                  | exec -> Some exec
                  | exception Pjrt.Abi.Error reason -> runtime_error reason)
              | exception Invalid_argument reason
                when String.starts_with ~prefix:"The precision" reason ->
                  invalid_arg reason
              | exception (Failure reason | Invalid_argument reason) ->
                  host_fallbacks := reason :: !host_fallbacks;
                  None)
        in
        Hashtbl.replace per_platform platform compiled;
        compiled
  in
  (match options with
  | Some (_ :: _) when not (use_xla ()) ->
      runtime_error
        "UNIMPLEMENTED: compiler options need the PJRT backend; set \
         OJAX_PJRT_PLUGIN"
  | Some _ when use_xla () -> ignore (compiled_for (current_platform ()))
  | _ -> ());
  fun args ->
    if use_xla () then (
      if Config.value Config.no_execution then
        failwith "compiled execution is disabled by no_execution";
      guard_implicit_args args;
      match compiled_for (current_platform ()) with
      | Some compiled when List.for_all is_hostable args ->
          xla_run compiled args (List.exists is_device args)
      | Some _ | None -> Jaxpr.eval_closed_jaxpr cj args)
    else (
      guard_implicit_args args;
      Jaxpr.eval_closed_jaxpr cj args)

let run_text text args =
  if use_xla () then (
    guard_implicit_args args;
    let exec =
      try Xla.compile_text text
      with Pjrt.Abi.Error reason -> runtime_error reason
    in
    xla_run exec args (List.exists is_device args))
  else
    failwith
      "Running an exported module needs the PJRT plugin; set OJAX_PJRT_PLUGIN"

let xla_plugin () = Xla.plugin_for "cpu"
let xla_client () = Xla.client_for "cpu"
let forced_cache : (string, Pjrt.Executable.t) Hashtbl.t = Hashtbl.create 16

let run_text_forced text args =
  with_platform "cpu" (fun () ->
      let exec =
        match Hashtbl.find_opt forced_cache text with
        | Some e -> e
        | None ->
            let e =
              try Xla.compile_text text
              with Pjrt.Abi.Error reason -> runtime_error reason
            in
            Hashtbl.replace forced_cache text e;
            e
      in
      xla_run exec args false)

let of_host_value v =
  (match v with
  | Concrete nd -> guard_host_to_device ~explicit:true nd
  | Device _ | Tracer _ -> ());
  if use_xla () then
    match v with
    | Concrete nd -> track_device (Xla.of_host nd)
    | Device _ | Tracer _ -> v
  else v

let to_host_value = function
  | Device b -> Concrete (Pjrt.Buffer.to_host b)
  | (Concrete _ | Tracer _) as v -> v
