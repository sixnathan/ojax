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
    | Tracer _ -> failwith "Backend.Interpreter: unexpected tracer result"

  let execute cj bufs =
    List.map concrete
      (Jaxpr.eval_closed_jaxpr cj (List.map (fun nd -> Concrete nd) bufs))
end

module Xla = struct
  type buffer = Pjrt.Buffer.t
  type compiled = Pjrt.Executable.t

  let client =
    lazy
      (let path = Pjrt.Discover.preflight () in
       Pjrt.Client.create (Pjrt.Abi.open_plugin path))

  let of_host nd = Pjrt.Buffer.of_host (Lazy.force client) nd
  let to_host buf = Pjrt.Buffer.to_host buf

  let compile cj =
    Pjrt.Executable.compile (Lazy.force client)
      (Stablehlo.Emit.emit_closed_jaxpr cj)

  let execute exec bufs =
    Array.to_list (Pjrt.Executable.execute exec (Array.of_list bufs))
end

let plugin_available () =
  match Sys.getenv_opt Pjrt.Discover.env_var with
  | Some path -> (not (Filename.is_relative path)) && Sys.file_exists path
  | None -> false

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

let concrete_arg = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "Backend: xla execution requires concrete arguments"

let all_concrete =
  List.for_all (function Concrete _ -> true | Tracer _ -> false)

let xla_run exec args =
  let bufs = List.map (fun v -> Xla.of_host (concrete_arg v)) args in
  let outs = Xla.execute exec bufs in
  let results = List.map (fun b -> Concrete (Xla.to_host b)) outs in
  List.iter Pjrt.Buffer.destroy bufs;
  List.iter Pjrt.Buffer.destroy outs;
  results

let executor (cj : closed_jaxpr) : value list -> value list =
  if Lazy.force use_xla then
    let exec = lazy (Xla.compile cj) in
    fun args ->
      if all_concrete args then xla_run (Lazy.force exec) args
      else Jaxpr.eval_closed_jaxpr cj args
  else fun args -> Jaxpr.eval_closed_jaxpr cj args
