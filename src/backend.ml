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

let cuda_path () =
  match Sys.getenv_opt "CUDA_ROOT" with
  | Some root -> Some root
  | None -> (
      match Sys.getenv_opt Pjrt.Discover.gpu_env_var with
      | None -> None
      | Some plugin ->
          let site =
            Filename.dirname (Filename.dirname (Filename.dirname plugin))
          in
          let candidate =
            Filename.concat (Filename.concat site "nvidia") "cu13"
          in
          if
            Sys.file_exists
              (Filename.concat (Filename.concat candidate "bin") "ptxas")
          then Some candidate
          else None)

let export_cuda_data_dir () =
  match cuda_path () with
  | None -> ()
  | Some path ->
      let flag = "--xla_gpu_cuda_data_dir=" ^ path in
      let merged =
        match Sys.getenv_opt "XLA_FLAGS" with
        | None | Some "" -> flag
        | Some flags -> flags ^ " " ^ flag
      in
      Unix.putenv "XLA_FLAGS" merged

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

  let forced_cpu_devices () =
    let configured = Config.value Config.num_cpu_devices in
    if configured > 0 then Some configured
    else
      let prefix = "--xla_force_host_platform_device_count=" in
      let n = String.length prefix in
      match Sys.getenv_opt "XLA_FLAGS" with
      | None -> None
      | Some flags -> (
          String.split_on_char ' ' flags
          |> List.filter_map (fun flag ->
              if String.length flag > n && String.sub flag 0 n = prefix then
                int_of_string_opt (String.sub flag n (String.length flag - n))
              else None)
          |> List.rev
          |> function
          | latest :: _ -> Some latest
          | [] -> None)

  let plugins = Hashtbl.create 2

  let plugin_for platform =
    match Hashtbl.find_opt plugins platform with
    | Some plugin -> plugin
    | None ->
        let path =
          match platform with
          | "cuda" ->
              export_cuda_data_dir ();
              Pjrt.Discover.gpu_preflight ()
          | _ -> Pjrt.Discover.preflight ()
        in
        let plugin = Pjrt.Abi.open_plugin path in
        Hashtbl.replace plugins platform plugin;
        plugin

  let lapack_registered = ref None

  let lapack_targets_ready () =
    match !lapack_registered with
    | Some ready -> ready
    | None ->
        let ready =
          Lapack_seam.Lapack.single_routines
          &&
          let plugin = plugin_for "cpu" in
          try
            List.iter
              (fun (target, handler) ->
                Pjrt.Abi.ffi_register_handler plugin target "Host" handler)
              Lapack_seam.Lapack.ffi_targets;
            true
          with Pjrt.Abi.Error _ -> false
        in
        lapack_registered := Some ready;
        ready

  let () = Stablehlo.Emit.lapack_ffi_ready := lapack_targets_ready
  let gpu_solver_registered = ref None

  let gpu_solver_targets_ready () =
    match !gpu_solver_registered with
    | Some ready -> ready
    | None ->
        let ready =
          Linalg.Gpu_solver.available ()
          &&
          let plugin = plugin_for "cuda" in
          try
            List.iter
              (fun (target, handler) ->
                Pjrt.Abi.ffi_register_handler plugin target "CUDA" handler)
              (Linalg.Gpu_solver.ffi_targets ());
            true
          with Pjrt.Abi.Error _ -> false
        in
        gpu_solver_registered := Some ready;
        ready

  let () = Stablehlo.Emit.gpu_solver_ready := gpu_solver_targets_ready
  let clients = Hashtbl.create 2

  let client_for platform =
    match Hashtbl.find_opt clients platform with
    | Some client -> client
    | None ->
        let plugin = plugin_for platform in
        let client =
          match platform with
          | "cuda" -> Pjrt.Client.create ~options:(gpu_client_options ()) plugin
          | _ -> (
              let first = Pjrt.Client.create plugin in
              match forced_cpu_devices () with
              | Some wanted
                when Pjrt.Client.platform first = "cpu"
                     && wanted <> Array.length (Pjrt.Client.devices first) ->
                  Pjrt.Client.destroy first;
                  Pjrt.Client.create ~cpu_devices:wanted plugin
              | _ -> first)
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

  let compile cj =
    compile_text (Stablehlo.Emit.emit_jax_text ~name:"_lambda" cj)

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

let verifier_diagnostic reason =
  let line =
    match String.index_opt reason '\n' with
    | Some i -> String.sub reason 0 i
    | None -> reason
  in
  let marker = ": error: " in
  let n = String.length marker in
  let rec find i =
    if i + n > String.length line then None
    else if String.sub line i n = marker then
      Some (String.sub line (i + n) (String.length line - i - n))
    else find (i + 1)
  in
  if String.starts_with ~prefix:"-:" line then
    match find 0 with
    | Some message when not (String.starts_with ~prefix:"expected " message) ->
        Some message
    | _ -> None
  else None

let compile_failure reason =
  match verifier_diagnostic reason with
  | Some message ->
      Py_exn.value_error
        ("Cannot lower jaxpr with verifier errors:\n\t" ^ message)
  | None -> runtime_error reason

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

let callback_failure : exn option ref = ref None

let dtype_of_ffi = function
  | 1 -> Dtype.Bool
  | 2 -> Dtype.I8
  | 3 -> Dtype.I16
  | 4 -> Dtype.I32
  | 5 -> Dtype.I64
  | 6 -> Dtype.Uint8
  | 7 -> Dtype.Uint16
  | 8 -> Dtype.Uint32
  | 9 -> Dtype.Uint64
  | 10 -> Dtype.F16
  | 11 -> Dtype.F32
  | 12 -> Dtype.F64
  | 15 -> Dtype.Complex64
  | 16 -> Dtype.Bf16
  | 18 -> Dtype.Complex128
  | code -> failwith (Printf.sprintf "host callback: dtype code %d" code)

let dispatch_host_callback id args rets =
  try
    let host, _ =
      match Core.custom_rule id with
      | Core.Callback_rule { host; result_avals; _ } -> (host, result_avals)
      | _ -> failwith "host callback: not a callback rule"
    in
    let inputs =
      Array.to_list
        (Array.map
           (fun (code, dims, data) ->
             Concrete (Pjrt.Buffer.host_of_bytes (dtype_of_ffi code) dims data))
           args)
    in
    let outputs = host inputs in
    List.iteri
      (fun i out ->
        let code, _, target = rets.(i) in
        let nd =
          match out with
          | Concrete nd -> nd
          | Device b -> Pjrt.Buffer.to_host b
          | Tracer _ -> failwith "host callback: tracer result"
        in
        let nd = Ndarray.canonicalize (dtype_of_ffi code) nd in
        Bigarray.Array1.blit (Pjrt.Buffer.host_bytes nd) target)
      outputs
  with error ->
    if !callback_failure = None then callback_failure := Some error

let () =
  ignore (Thread.self ());
  Callback.register "ojax_host_callback" dispatch_host_callback

let raise_callback_failure () =
  match !callback_failure with
  | Some error ->
      callback_failure := None;
      raise error
  | None -> ()

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
      raise_callback_failure ();
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
  let lowered = Stablehlo.Emit.lower ~name:"_lambda" cj in
  Monitoring.record_event_duration_secs
    ~kwargs:[ ("fun_name", Monitoring.Text "jit") ]
    "/jax/core/compile/jaxpr_to_mlir_module_duration"
    (Sys.time () -. start);
  lowered

let select_kept kept args =
  let table = Array.of_list args in
  List.map (fun i -> table.(i)) kept

module PB = Pjrt.Buffer
module PE = Pjrt.Executable

let device_tables = Hashtbl.create 2

let local_devices_for platform =
  match Hashtbl.find_opt device_tables platform with
  | Some table -> table
  | None ->
      let client = Xla.client_for platform in
      let all = Pjrt.Client.devices client in
      let table =
        if
          Pjrt.Client.platform client = "cpu"
          && Xla.forced_cpu_devices () = None
        then Array.sub all 0 1
        else all
      in
      Hashtbl.replace device_tables platform table;
      table

let host_device = { Pjrt.Client.id = 0; process_index = 0; kind = "cpu" }

let local_devices () =
  if use_xla () then local_devices_for (current_platform ())
  else [| host_device |]

let devices ?backend () =
  let platform =
    match backend with
    | Some name -> expand_platform name
    | None -> current_platform ()
  in
  let table =
    match (platform, Lazy.force backend_mode) with
    | "cuda", _ -> local_devices_for "cuda"
    | _, `Interpreter -> [| host_device |]
    | _, `Xla -> local_devices_for "cpu"
    | _, `Auto ->
        if Lazy.force cpu_uses_xla then local_devices_for "cpu"
        else [| host_device |]
  in
  Array.to_list
    (Array.map
       (fun (d : Pjrt.Client.device) -> Printf.sprintf "%s:%d" platform d.id)
       table)

let multi_device_available () = use_xla ()

let platform () =
  if use_xla () then Pjrt.Client.platform (Xla.client ()) else "cpu"

let mesh_coordinates mesh pos =
  let rem = ref pos in
  List.rev_map
    (fun (axis, size) ->
      let c = !rem mod size in
      rem := !rem / size;
      (axis, c))
    (List.rev mesh)

let check_divisible ~shape ~mesh ~spec =
  Array.iteri
    (fun d axes ->
      let tiles =
        List.fold_left (fun acc a -> acc * List.assoc a mesh) 1 axes
      in
      if tiles > 1 && shape.(d) mod tiles <> 0 then
        invalid_arg
          (Printf.sprintf
             "Sharding implies that array axis %d is partitioned %d times, but \
              the dimension size is %d (full shape: (%s), per-dimension tiling \
              factors should evenly divide the shape)"
             d tiles shape.(d)
             (String.concat ", "
                (Array.to_list (Array.map string_of_int shape)))))
    spec

let named_index ~shape ~mesh ~spec pos =
  let coords = mesh_coordinates mesh pos in
  Array.mapi
    (fun d axes ->
      let tile, tiles =
        List.fold_left
          (fun (tile, tiles) a ->
            let size = List.assoc a mesh in
            ((tile * size) + List.assoc a coords, tiles * size))
          (0, 1) axes
      in
      let chunk = shape.(d) / tiles in
      (tile * chunk, (tile + 1) * chunk))
    spec

let replica_ids indices =
  let seen = Hashtbl.create 8 in
  Array.map
    (fun index ->
      let n = Option.value ~default:0 (Hashtbl.find_opt seen index) in
      Hashtbl.replace seen index (n + 1);
      n)
    indices

let placed_of ~shape ~dtype ~devices ~indices ~layout raws =
  let ids = replica_ids indices in
  Device
    (PB.Placed
       {
         global_shape = shape;
         dtype;
         layout;
         shards =
           Array.mapi
             (fun p raw ->
               {
                 PB.device = devices.(p);
                 index = indices.(p);
                 replica_id = ids.(p);
                 data = raw;
               })
             raws;
       })

let host_of = function
  | Concrete nd -> nd
  | Device b -> PB.to_host b
  | Tracer _ -> failwith "Backend: placement requires concrete arguments"

let place_named nd ~mesh ~devices ~spec =
  let shape = Ndarray.shape nd in
  check_divisible ~shape ~mesh ~spec;
  let client = Xla.client () in
  let indices =
    Array.mapi (fun pos _ -> named_index ~shape ~mesh ~spec pos) devices
  in
  let raws =
    PB.shards_of_host client nd
      (Array.mapi (fun pos device -> (indices.(pos), device)) devices)
  in
  placed_of ~shape ~dtype:(Ndarray.dtype nd) ~devices ~indices
    ~layout:(PB.Named { mesh_axes = mesh; mesh_devices = devices; spec })
    raws

let place_on nd device =
  let shape = Ndarray.shape nd in
  let index = Array.map (fun d -> (0, d)) shape in
  let raws = PB.shards_of_host (Xla.client ()) nd [| (index, device) |] in
  placed_of ~shape ~dtype:(Ndarray.dtype nd) ~devices:[| device |]
    ~indices:[| index |] ~layout:(PB.On_device device) raws

let replicated_spec rank = Array.make rank []

type plan = {
  mesh : (string * int) list;
  devices : int array;
  arg_specs : string list array list;
  out_specs : string list array option list;
}

type placed_program = {
  program : PE.t;
  kept : int list option;
  tilings : PE.output_sharding list Lazy.t;
}

let placed_programs : (int * plan, placed_program option) Hashtbl.t =
  Hashtbl.create 16

let compile_plan (cj : closed_jaxpr) plan =
  match Hashtbl.find_opt placed_programs (cj.jid, plan) with
  | Some entry -> entry
  | None ->
      let entry =
        match
          if Array.length plan.devices = 1 then
            let lowered = Stablehlo.Emit.lower ~name:"_lambda" cj in
            (lowered.text, Some lowered.kept)
          else
            let lowered =
              Stablehlo.Emit.lower_partitioned ~name:"_lambda"
                {
                  pt_mesh = plan.mesh;
                  pt_devices = plan.devices;
                  pt_in = plan.arg_specs;
                  pt_out = plan.out_specs;
                }
                cj
            in
            (lowered.text, Some lowered.kept)
        with
        | text, _ when mentions_unsupported_type text ->
            host_fallbacks := "f6 dtypes have no XLA type" :: !host_fallbacks;
            None
        | text, kept ->
            let propagate_outputs =
              Array.length plan.devices > 1
              && List.exists Option.is_none plan.out_specs
            in
            let program =
              try
                PE.compile_placed (Xla.client ()) text ~devices:plan.devices
                  ~propagate_outputs
              with Pjrt.Abi.Error reason -> runtime_error reason
            in
            incr compiled_on_xla;
            Some { program; kept; tilings = lazy (PE.output_shardings program) }
        | exception (Failure reason | Invalid_argument reason) ->
            host_fallbacks := reason :: !host_fallbacks;
            None
      in
      Hashtbl.replace placed_programs (cj.jid, plan) entry;
      entry

let raws_for_arg plan arg spec =
  match arg with
  | Device (PB.Placed { layout = PB.Named n; shards; _ })
    when n.mesh_axes = plan.mesh
         && n.mesh_devices = plan.devices
         && n.spec = spec ->
      Array.map
        (fun device ->
          let s =
            List.find
              (fun (s : PB.shard) -> s.device = device)
              (Array.to_list shards)
          in
          s.data)
        plan.devices
  | _ ->
      let nd = host_of arg in
      let shape = Ndarray.shape nd in
      check_divisible ~shape ~mesh:plan.mesh ~spec;
      let client = Xla.client () in
      PB.shards_of_host client nd
        (Array.mapi
           (fun pos device ->
             (named_index ~shape ~mesh:plan.mesh ~spec pos, device))
           plan.devices)

let raw_on_device device = function
  | Device (PB.Placed { shards = [| s |]; _ }) when s.device = device -> s.data
  | Device (PB.Single raw) when PB.raw_device raw = device -> raw
  | arg ->
      let nd = host_of arg in
      (PB.shards_of_host (Xla.client ()) nd
         [| (Array.map (fun d -> (0, d)) (Ndarray.shape nd), device) |]).(0)

let indices_of_tiling shape devices tiling =
  let full = Array.map (fun d -> (0, d)) shape in
  match tiling with
  | PE.Replicated -> Array.map (fun _ -> full) devices
  | PE.Tiled { dims; grid } ->
      let out = Array.map (fun _ -> full) devices in
      Array.iteri
        (fun flat partition ->
          let rem = ref flat in
          let coords = Array.make (Array.length dims) 0 in
          for i = Array.length dims - 1 downto 0 do
            coords.(i) <- !rem mod dims.(i);
            rem := !rem / dims.(i)
          done;
          out.(partition) <-
            Array.mapi
              (fun d extent ->
                let chunk = extent / dims.(d) in
                (coords.(d) * chunk, (coords.(d) + 1) * chunk))
              shape)
        grid;
      out

let tiles_of_tiling rank = function
  | PE.Replicated -> Array.make rank 1
  | PE.Tiled { dims; _ } -> Array.sub dims 0 rank

let derive_spec ~shape ~mesh indices =
  let rank = Array.length shape in
  let tiles_wanted =
    Array.mapi
      (fun d extent ->
        let lo, hi = indices.(0).(d) in
        if extent = 0 then 1 else extent / max 1 (hi - lo))
      shape
  in
  let rec choose_axes unused product =
    if product = 1 then [ [] ]
    else
      List.concat_map
        (fun (axis, size) ->
          if size > 1 && product mod size = 0 then
            List.map
              (fun rest -> axis :: rest)
              (choose_axes
                 (List.filter (fun (a, _) -> a <> axis) unused)
                 (product / size))
          else [])
        unused
  in
  let rec assign d unused =
    if d = rank then [ [] ]
    else
      List.concat_map
        (fun axes ->
          let remaining =
            List.filter (fun (a, _) -> not (List.mem a axes)) unused
          in
          List.map (fun rest -> axes :: rest) (assign (d + 1) remaining))
        (choose_axes unused tiles_wanted.(d))
  in
  List.find_map
    (fun candidate ->
      let spec = Array.of_list candidate in
      let matches =
        Array.for_all Fun.id
          (Array.mapi
             (fun pos index -> named_index ~shape ~mesh ~spec pos = index)
             indices)
      in
      if matches then Some spec else None)
    (assign 0 mesh)

let run_program plan placed args =
  let single = Array.length plan.devices = 1 in
  let args =
    match placed.kept with Some kept -> select_kept kept args | None -> args
  in
  let lists =
    if single then
      [| Array.of_list (List.map (raw_on_device plan.devices.(0)) args) |]
    else
      let per_arg =
        let specs =
          match placed.kept with
          | Some kept -> select_kept kept plan.arg_specs
          | None -> plan.arg_specs
        in
        Array.of_list (List.map2 (raws_for_arg plan) args specs)
      in
      Array.mapi
        (fun pos _ -> Array.map (fun raws -> raws.(pos)) per_arg)
        plan.devices
  in
  let outs =
    try PE.execute_devices placed.program lists
    with Pjrt.Abi.Error reason -> runtime_error reason
  in
  let n_out = if Array.length outs = 0 then 0 else Array.length outs.(0) in
  List.init n_out (fun k ->
      let raws = Array.map (fun row -> row.(k)) outs in
      let probe = PB.Single raws.(0) in
      let dtype = PB.element_type probe in
      if single then
        let shape = PB.raw_dimensions raws.(0) in
        placed_of ~shape ~dtype ~devices:plan.devices
          ~indices:[| Array.map (fun d -> (0, d)) shape |]
          ~layout:(PB.On_device plan.devices.(0))
          raws
      else
        let tiling =
          match List.nth_opt plan.out_specs k with
          | Some (Some _) -> None
          | _ -> Some (List.nth (Lazy.force placed.tilings) k)
        in
        let local = PB.raw_dimensions raws.(0) in
        let shape =
          match (List.nth_opt plan.out_specs k, tiling) with
          | Some (Some spec), _ ->
              Array.mapi
                (fun d extent ->
                  List.fold_left
                    (fun acc a -> acc * List.assoc a plan.mesh)
                    extent spec.(d))
                local
          | _, Some t ->
              let tiles = tiles_of_tiling (Array.length local) t in
              Array.mapi (fun d extent -> extent * tiles.(d)) local
          | _ -> local
        in
        let indices, layout =
          match (List.nth_opt plan.out_specs k, tiling) with
          | Some (Some spec), _ ->
              ( Array.mapi
                  (fun pos _ -> named_index ~shape ~mesh:plan.mesh ~spec pos)
                  plan.devices,
                PB.Named
                  { mesh_axes = plan.mesh; mesh_devices = plan.devices; spec }
              )
          | _, Some t -> (
              let indices = indices_of_tiling shape plan.devices t in
              match derive_spec ~shape ~mesh:plan.mesh indices with
              | Some spec ->
                  ( indices,
                    PB.Named
                      {
                        mesh_axes = plan.mesh;
                        mesh_devices = plan.devices;
                        spec;
                      } )
              | None -> (indices, PB.Tiled))
          | _ -> failwith "Backend: output has no sharding"
        in
        placed_of ~shape ~dtype ~devices:plan.devices ~indices ~layout raws)

let rank_of v = Array.length (Core.get_aval v).shape

let incompatible () =
  invalid_arg "Received incompatible devices for jitted computation."

let plan_of_args args =
  let placed =
    List.filter_map (function Device (PB.Placed p) -> Some p | _ -> None) args
  in
  let device_list (p : PB.placed) =
    Array.map (fun (s : PB.shard) -> s.device) p.shards
  in
  let one_device =
    List.for_all (fun (p : PB.placed) -> Array.length p.shards = 1) placed
  in
  match placed with
  | [] -> None
  | first :: _ when one_device ->
      let device = (device_list first).(0) in
      if List.exists (fun p -> (device_list p).(0) <> device) placed then
        incompatible ();
      Some
        {
          mesh = [];
          devices = [| device |];
          arg_specs = List.map (fun v -> replicated_spec (rank_of v)) args;
          out_specs = [];
        }
  | _ ->
      let named =
        List.filter_map
          (fun (p : PB.placed) ->
            match p.layout with PB.Named n -> Some n | _ -> None)
          placed
      in
      let mesh, devices =
        match named with
        | n :: _ -> (n.mesh_axes, n.mesh_devices)
        | [] ->
            let d = device_list (List.hd placed) in
            ([ ("_tiled", Array.length d) ], d)
      in
      List.iter
        (fun (p : PB.placed) ->
          match p.layout with
          | PB.Named n when n.mesh_axes = mesh && n.mesh_devices = devices -> ()
          | PB.Tiled when device_list p = devices -> ()
          | _ -> incompatible ())
        placed;
      Some
        {
          mesh;
          devices;
          arg_specs =
            List.map
              (function
                | Device (PB.Placed { layout = PB.Named n; _ }) -> n.spec
                | v -> replicated_spec (rank_of v))
              args;
          out_specs = [];
        }

let plan_for_mesh mesh devices args =
  {
    mesh;
    devices;
    arg_specs =
      List.map
        (function
          | Device (PB.Placed { layout = PB.Named n; _ })
            when n.mesh_axes = mesh && n.mesh_devices = devices ->
              n.spec
          | v -> replicated_spec (rank_of v))
        args;
    out_specs = [];
  }

let placed_text ~name ?(out_specs = []) cj args =
  match plan_of_args args with
  | None -> Stablehlo.Emit.emit_jax_text ~name cj
  | Some plan when Array.length plan.devices = 1 ->
      Stablehlo.Emit.emit_jax_text ~name cj
  | Some plan ->
      Stablehlo.Mlir_names.renumber_callbacks
        (Stablehlo.Emit.lower_partitioned ~name
           {
             pt_mesh = plan.mesh;
             pt_devices = plan.devices;
             pt_in = plan.arg_specs;
             pt_out = out_specs;
           }
           cj)
          .text

let run_plan cj plan args =
  match compile_plan cj plan with
  | Some placed -> Some (run_program plan placed args)
  | None -> None

let () =
  Core.placed_eval :=
    fun prim args ->
      match plan_of_args args with
      | None -> None
      | Some plan -> (
          let avals = List.map Core.get_aval args in
          match Jaxpr.make_jaxpr avals (fun vs -> Core.bind prim vs) with
          | exception (Failure _ | Invalid_argument _) -> None
          | cj -> run_plan cj plan args)

type eager_program =
  | Eager_compiled of Pjrt.Executable.t * aval list * int list
  | Eager_host

type eager_key = {
  ek_prim : primitive;
  ek_avals : aval list;
  ek_platform : string;
  ek_context : (string * string) list;
}

module Eager_cache = Hashtbl.Make (struct
  type t = eager_key

  let equal a b = compare a b = 0
  let hash key = Hashtbl.hash_param 64 256 key
end)

type eager_stats = {
  eager_compiled : int;
  eager_cache_hits : int;
  eager_executions : int;
  eager_fallbacks : (string * string * int) list;
}

let eager_programs = Eager_cache.create 512

let eager_texts : (string * string, Pjrt.Executable.t) Hashtbl.t =
  Hashtbl.create 512

let eager_compiled = ref 0
let eager_cache_hits = ref 0
let eager_executions = ref 0
let eager_fallback_table : (string * string, int) Hashtbl.t = Hashtbl.create 16

let note_eager_fallback_named name reason =
  let key = (name, reason) in
  let seen =
    Option.value ~default:0 (Hashtbl.find_opt eager_fallback_table key)
  in
  Hashtbl.replace eager_fallback_table key (seen + 1)

let note_eager_fallback prim reason =
  note_eager_fallback_named (Pretty_printer.primitive_name prim) reason

let eager_stats () =
  {
    eager_compiled = !eager_compiled;
    eager_cache_hits = !eager_cache_hits;
    eager_executions = !eager_executions;
    eager_fallbacks =
      List.sort compare
        (Hashtbl.fold
           (fun (name, reason) count acc -> (name, reason, count) :: acc)
           eager_fallback_table []);
  }

let eager_runs_impl = function
  | Custom_jvp_call _ | Custom_vjp_call _ | Custom_lin _ | Custom_linear_solve _
  | Xla_call _ | Stop_gradient ->
      true
  | _ -> false

let eager_holds_jaxpr = function
  | Reduce _ | Composite _ | Reduce_window _ | Select_and_scatter _ | Cond _
  | Scan _ | While _ ->
      true
  | _ -> false

let single_primitive_jaxpr prim avals_in avals_out =
  let fresh vaval = { vid = Core.fresh_id (); vaval } in
  let in_binders = List.map fresh avals_in in
  let outs = List.map fresh avals_out in
  {
    jid = Core.fresh_id ();
    consts = [];
    jaxpr =
      {
        in_binders;
        eqns =
          [
            {
              prim;
              inputs = List.map (fun v -> A_var v) in_binders;
              outs;
              multiple_results = List.length outs > 1;
            };
          ];
        outs = List.map (fun v -> A_var v) outs;
      };
  }

let compile_eager platform text =
  match Hashtbl.find_opt eager_texts (platform, text) with
  | Some exec ->
      incr eager_cache_hits;
      exec
  | None ->
      let exec = Xla.compile_text text in
      incr eager_compiled;
      Hashtbl.replace eager_texts (platform, text) exec;
      exec

let eager_rejection = function
  | Stack_overflow | Out_of_memory -> false
  | _ -> true

let build_eager_program prim avals_in platform =
  let fall_back reason =
    note_eager_fallback prim reason;
    Eager_host
  in
  match Core.rules.abstract_eval prim avals_in with
  | exception ((Failure _ | Invalid_argument _ | Not_found) as error) ->
      fall_back (Printexc.to_string error)
  | exception error when eager_rejection error ->
      fall_back (Printexc.to_string error)
  | avals_out -> (
      match
        Stablehlo.Emit.lower
          ~name:(Pretty_printer.primitive_name prim)
          (single_primitive_jaxpr prim avals_in avals_out)
      with
      | { text; _ } when mentions_unsupported_type text ->
          fall_back "f6 dtypes have no XLA type"
      | exception ((Failure _ | Invalid_argument _ | Not_found) as error) ->
          fall_back (Printexc.to_string error)
      | { text; kept } -> (
          match compile_eager platform text with
          | exec -> Eager_compiled (exec, avals_out, kept)
          | exception Pjrt.Abi.Error reason -> fall_back reason))

let eager_program prim avals_in =
  let platform = current_platform () in
  if eager_holds_jaxpr prim then build_eager_program prim avals_in platform
  else
    let key =
      {
        ek_prim = prim;
        ek_avals = avals_in;
        ek_platform = platform;
        ek_context = Config.trace_context ();
      }
    in
    match Eager_cache.find_opt eager_programs key with
    | Some program ->
        incr eager_cache_hits;
        program
    | None ->
        let program = build_eager_program prim avals_in platform in
        Eager_cache.replace eager_programs key program;
        program

let output_matches outs avals_out =
  List.compare_lengths outs avals_out = 0
  && List.for_all2
       (fun out (aval : aval) ->
         match out with
         | Concrete nd ->
             Ndarray.dtype nd = aval.dtype && Ndarray.shape nd = aval.shape
         | Device _ | Tracer _ -> false)
       outs avals_out

let eager_eval prim args =
  if eager_runs_impl prim || Stablehlo.Emit.emitting () || not (use_xla ()) then
    None
  else
    let avals_in = List.map Core.get_aval args in
    match eager_program prim avals_in with
    | Eager_host -> None
    | Eager_compiled (exec, avals_out, kept) -> (
        match xla_run exec (select_kept kept args) false with
        | outs when output_matches outs avals_out ->
            incr eager_executions;
            Some outs
        | _ ->
            note_eager_fallback prim "compiled output type differs";
            None)

let () = Core.eager_eval := eager_eval

let compile_region name cj =
  match Stablehlo.Emit.lower ~name cj with
  | { text; _ } when mentions_unsupported_type text ->
      Error "f6 dtypes have no XLA type"
  | { text; kept } -> (
      match compile_eager (current_platform ()) text with
      | exec -> Ok (exec, kept)
      | exception Pjrt.Abi.Error reason -> Error reason)
  | exception ((Failure _ | Invalid_argument _ | Not_found) as error) ->
      Error (Printexc.to_string error)

type region_key = {
  rk_name : string;
  rk_statics : Obj.t;
  rk_avals : aval list;
  rk_platform : string;
  rk_context : (string * string) list;
}

module Region_cache = Hashtbl.Make (struct
  type t = region_key

  let equal a b = compare a b = 0
  let hash key = Hashtbl.hash_param 64 256 key
end)

let region_programs = Region_cache.create 512

let atom_output_aval = function
  | A_var (v : var) -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar aval -> aval

let build_region name f avals =
  let fall_back reason =
    note_eager_fallback_named name reason;
    None
  in
  match Jaxpr.make_jaxpr avals f with
  | exception error when Py_exn.class_name error <> None -> raise error
  | exception error when eager_rejection error ->
      fall_back (Printexc.to_string error)
  | cj -> (
      match compile_region name cj with
      | Ok (exec, kept) ->
          Some (exec, List.map atom_output_aval cj.jaxpr.outs, kept)
      | Error reason -> fall_back reason)

let inline_call_cache : (region_key, closed_jaxpr option) Hashtbl.t =
  Hashtbl.create 512

let stage_inline_call ~inline key name f avals =
  match Hashtbl.find_opt inline_call_cache key with
  | Some c -> c
  | None ->
      let c =
        match Jaxpr.make_jaxpr_hoisted avals f with
        | c, [] ->
            let short =
              match String.rindex_opt name '.' with
              | Some i -> String.sub name (i + 1) (String.length name - i - 1)
              | None -> name
            in
            Pretty_printer.register_call_name c.jid short;
            let wraps_one_call =
              match c.jaxpr.eqns with
              | [ { prim = Xla_call _; _ } ] -> true
              | _ -> false
            in
            if inline || wraps_one_call then Core.mark_inline_call c.jid;
            Some c
        | _ -> None
        | exception error when eager_rejection error -> None
      in
      Hashtbl.replace inline_call_cache key c;
      c

let rec jitted_keyed ?(inline = true) (name, statics) f args =
  match (Core.find_top_trace args).kind with
  | (KJVP | KBatch | KPE)
    when not
           (Config.value Config.disable_jit
           || Config.value Config.no_execution
           || Stablehlo.Emit.emitting ()) -> (
      let avals = List.map Core.get_aval args in
      let key =
        {
          rk_name = name;
          rk_statics = statics;
          rk_avals = avals;
          rk_platform = "";
          rk_context = Config.trace_context ();
        }
      in
      match stage_inline_call ~inline key name f avals with
      | Some c -> Core.bind (Xla_call c) args
      | None -> f args)
  | _ -> jitted_eval (name, statics) f args

and jitted_eval (name, statics) f args =
  if
    Config.value Config.disable_jit
    || Config.value Config.no_execution
    || Core.tracing_dynamically ()
    || Stablehlo.Emit.emitting ()
    || (not (List.for_all is_hostable args))
    || List.exists Core.is_placed args
    || not (use_xla ())
  then f args
  else
    let avals = List.map Core.get_aval args in
    let key =
      {
        rk_name = name;
        rk_statics = statics;
        rk_avals = avals;
        rk_platform = current_platform ();
        rk_context = Config.trace_context ();
      }
    in
    let program =
      match Region_cache.find_opt region_programs key with
      | Some program ->
          incr eager_cache_hits;
          program
      | None ->
          let program = build_region name f avals in
          Region_cache.replace region_programs key program;
          program
    in
    match program with
    | None -> f args
    | Some (exec, wanted, kept) -> (
        match xla_run exec (select_kept kept args) false with
        | outs when output_matches outs wanted ->
            incr eager_executions;
            outs
        | _ -> f args)

let () =
  Jaxpr.eager_jit :=
    fun name key f args ->
      jitted_keyed ("named_jit." ^ name, Obj.repr key) f args

let () =
  Core.call_inline :=
    fun c args ->
      jitted_keyed
        ("xla_call." ^ Pretty_printer.call_name c.jid, Obj.repr c.jid)
        (fun a -> Jaxpr.eval_closed_jaxpr c a)
        args

let executor ?options (cj : closed_jaxpr) : value list -> value list =
  let precision = Config.value Config.default_matmul_precision in
  let per_platform = Hashtbl.create 2 in
  let artifacts_for platform =
    match Hashtbl.find_opt per_platform platform with
    | Some artifacts -> artifacts
    | None ->
        let emitted =
          Config.with_value Config.default_matmul_precision precision (fun () ->
              match emit_timed cj with
              | { text; _ } when mentions_unsupported_type text ->
                  host_fallbacks :=
                    "f6 dtypes have no XLA type" :: !host_fallbacks;
                  (None, None)
              | lowered -> (Some lowered, Stablehlo.Emit.shard_mesh_seen ())
              | exception Invalid_argument reason
                when String.starts_with ~prefix:"The precision" reason ->
                  invalid_arg reason
              | exception (Failure reason | Invalid_argument reason) ->
                  host_fallbacks := reason :: !host_fallbacks;
                  (None, None))
        in
        let exec =
          lazy
            (match fst emitted with
            | None -> None
            | Some { Stablehlo.Emit.text; kept } -> (
                incr compiled_on_xla;
                match Xla.compile_text ?options text with
                | exec -> Some (exec, kept)
                | exception Pjrt.Abi.Error reason -> compile_failure reason))
        in
        Hashtbl.replace per_platform platform (emitted, exec);
        (emitted, exec)
  in
  (match options with
  | Some (_ :: _) when not (use_xla ()) ->
      runtime_error
        "UNIMPLEMENTED: compiler options need the PJRT backend; set \
         OJAX_PJRT_PLUGIN"
  | Some _ when use_xla () ->
      ignore (Lazy.force (snd (artifacts_for (current_platform ()))))
  | _ -> ());
  fun args ->
    if use_xla () then (
      if Config.value Config.no_execution then
        failwith "compiled execution is disabled by no_execution";
      guard_implicit_args args;
      let emitted, exec = artifacts_for (current_platform ()) in
      let plain args =
        match Lazy.force exec with
        | Some (compiled, kept) when List.for_all is_hostable args ->
            xla_run compiled (select_kept kept args)
              (List.exists is_device args)
        | Some _ | None -> Jaxpr.eval_closed_jaxpr cj args
      in
      if not (List.for_all is_hostable args) then plain args
      else
        let via plan =
          match run_plan cj plan args with
          | Some outs -> outs
          | None -> Jaxpr.eval_closed_jaxpr cj args
        in
        match snd emitted with
        | Some (mesh, devices) when Array.length devices > 1 ->
            via (plan_for_mesh mesh devices args)
        | _ -> (
            match plan_of_args args with
            | Some plan -> via plan
            | None -> plain args))
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

let client () = Xla.client ()
let clear_placed_programs () = Hashtbl.reset placed_programs
