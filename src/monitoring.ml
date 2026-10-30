type event_listener = string -> unit
type duration_listener = string -> float -> unit
type scalar_listener = string -> float -> unit

let event_listeners : event_listener list ref = ref []
let duration_listeners : duration_listener list ref = ref []
let scalar_listeners : scalar_listener list ref = ref []
let register_event_listener f = event_listeners := f :: !event_listeners

let register_event_duration_secs_listener f =
  duration_listeners := f :: !duration_listeners

let register_scalar_listener f = scalar_listeners := f :: !scalar_listeners

let clear_event_listeners () =
  event_listeners := [];
  duration_listeners := [];
  scalar_listeners := []

let record_event name = List.iter (fun f -> f name) !event_listeners

let record_event_duration_secs name duration =
  List.iter (fun f -> f name duration) !duration_listeners

let record_scalar name value =
  List.iter (fun f -> f name value) !scalar_listeners

let record_event_time_span name thunk =
  let start = Sys.time () in
  Fun.protect
    ~finally:(fun () -> record_event_duration_secs name (Sys.time () -. start))
    thunk
