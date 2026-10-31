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

  let plugin =
    lazy
      (let path = Pjrt.Discover.preflight () in
       Pjrt.Abi.open_plugin path)

  let client = lazy (Pjrt.Client.create (Lazy.force plugin))
  let of_host nd = Pjrt.Buffer.of_host (Lazy.force client) nd
  let to_host buf = Pjrt.Buffer.to_host buf
  let compile_text text = Pjrt.Executable.compile (Lazy.force client) text
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
      let outs = Xla.execute exec (List.map fst inputs) in
      if resident then List.map (fun b -> Device b) outs
      else
        Fun.protect
          ~finally:(fun () -> List.iter Pjrt.Buffer.destroy outs)
          (fun () -> List.map (fun b -> Concrete (Xla.to_host b)) outs))

let compiled_on_xla = ref 0
let host_fallbacks : string list ref = ref []
let stats () = (!compiled_on_xla, List.rev !host_fallbacks)

let executor (cj : closed_jaxpr) : value list -> value list =
  if Lazy.force use_xla then (
    let exec =
      lazy
        (match Stablehlo.Emit.emit_closed_jaxpr cj with
        | text ->
            incr compiled_on_xla;
            Some (Xla.compile_text text)
        | exception (Failure reason | Invalid_argument reason) ->
            host_fallbacks := reason :: !host_fallbacks;
            None)
    in
    fun args ->
      if Config.value Config.no_execution then
        failwith "compiled execution is disabled by no_execution";
      match Lazy.force exec with
      | Some compiled when List.for_all is_hostable args ->
          xla_run compiled args (List.exists is_device args)
      | Some _ | None -> Jaxpr.eval_closed_jaxpr cj args)
  else fun args -> Jaxpr.eval_closed_jaxpr cj args

let run_text text args =
  if Lazy.force use_xla then
    let exec = Xla.compile_text text in
    xla_run exec args (List.exists is_device args)
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
        let e = Xla.compile_text text in
        Hashtbl.replace forced_cache text e;
        e
  in
  xla_run exec args false

let of_host_value v =
  if Lazy.force use_xla then
    match v with
    | Concrete nd -> Device (Xla.of_host nd)
    | Device _ | Tracer _ -> v
  else v

let to_host_value = function
  | Device b -> Concrete (Pjrt.Buffer.to_host b)
  | (Concrete _ | Tracer _) as v -> v
