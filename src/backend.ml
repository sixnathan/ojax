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

module Xla = struct
  type buffer = Pjrt.Buffer.t
  type compiled = Pjrt.Executable.t

  let plugin = lazy (Pjrt.Abi.open_plugin (Pjrt.Discover.preflight ()))
  let client = lazy (Pjrt.Client.create (Lazy.force plugin))
  let of_host nd = Pjrt.Buffer.of_host (Lazy.force client) nd
  let to_host buf = Pjrt.Buffer.to_host buf

  let compile_text ?options text =
    let start = Sys.time () in
    let exec = Pjrt.Executable.compile ?options (Lazy.force client) text in
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

let use_xla =
  lazy
    (match Sys.getenv_opt "OJAX_BACKEND" with
    | Some "interpreter" -> false
    | Some "xla" -> true
    | None -> plugin_available ()
    | Some other ->
        invalid_arg
          (Printf.sprintf
             "OJAX_BACKEND: unknown backend %S (expected \"interpreter\" or \
              \"xla\")"
             other))

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
  | Device b -> (b, false)
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
  if Lazy.force use_xla then (
    let build () =
      Config.with_value Config.default_matmul_precision precision (fun () ->
          match emit_timed cj with
          | text when mentions_unsupported_type text ->
              host_fallbacks := "f6 dtypes have no XLA type" :: !host_fallbacks;
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
    let exec =
      match options with
      | Some _ -> Lazy.from_val (build ())
      | None -> Lazy.from_fun build
    in
    fun args ->
      if Config.value Config.no_execution then
        failwith "compiled execution is disabled by no_execution";
      guard_implicit_args args;
      match Lazy.force exec with
      | Some compiled when List.for_all is_hostable args ->
          xla_run compiled args (List.exists is_device args)
      | Some _ | None -> Jaxpr.eval_closed_jaxpr cj args)
  else (
    (match options with
    | Some (_ :: _) ->
        runtime_error
          "UNIMPLEMENTED: compiler options need the PJRT backend; set \
           OJAX_PJRT_PLUGIN"
    | _ -> ());
    fun args ->
      guard_implicit_args args;
      Jaxpr.eval_closed_jaxpr cj args)

let run_text text args =
  if Lazy.force use_xla then (
    guard_implicit_args args;
    let exec =
      try Xla.compile_text text
      with Pjrt.Abi.Error reason -> runtime_error reason
    in
    xla_run exec args (List.exists is_device args))
  else
    failwith
      "Running an exported module needs the PJRT plugin; set OJAX_PJRT_PLUGIN"

let xla_plugin () = Lazy.force Xla.plugin
let xla_client () = Lazy.force Xla.client
let forced_cache : (string, Pjrt.Executable.t) Hashtbl.t = Hashtbl.create 16

let run_text_forced text args =
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
  xla_run exec args false

let of_host_value v =
  (match v with
  | Concrete nd -> guard_host_to_device ~explicit:true nd
  | Device _ | Tracer _ -> ());
  if Lazy.force use_xla then
    match v with
    | Concrete nd -> track_device (Xla.of_host nd)
    | Device _ | Tracer _ -> v
  else v

let to_host_value = function
  | Device b -> Concrete (Pjrt.Buffer.to_host b)
  | (Concrete _ | Tracer _) as v -> v
