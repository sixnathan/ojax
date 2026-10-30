let trace_annotation _name thunk = thunk ()
let step_trace_annotation _name ~step_num:_ thunk = thunk ()
let annotate_function f = f
let start_trace _log_dir = ()
let stop_trace () = ()

let trace log_dir thunk =
  start_trace log_dir;
  Fun.protect ~finally:stop_trace thunk

let device_memory_profile () = ""

let save_device_memory_profile path =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr oc)
    (fun () -> output_string oc (device_memory_profile ()))
