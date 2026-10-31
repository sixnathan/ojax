type metadata = Text of string | Number of int
type kwargs = (string * metadata) list
type event_listener = string -> kwargs -> unit
type duration_listener = string -> float -> kwargs -> unit
type time_span_listener = string -> float -> float -> kwargs -> unit
type scalar_listener = string -> float -> kwargs -> unit

let event_listeners : event_listener list ref = ref []
let duration_listeners : duration_listener list ref = ref []
let time_span_listeners : time_span_listener list ref = ref []
let scalar_listeners : scalar_listener list ref = ref []
let register store f = store := !store @ [ f ]
let register_event_listener f = register event_listeners f
let register_event_duration_secs_listener f = register duration_listeners f
let register_event_time_span_listener f = register time_span_listeners f
let register_scalar_listener f = register scalar_listeners f

let unregister store f =
  assert (List.memq f !store);
  let rec drop = function
    | [] -> []
    | x :: rest -> if x == f then rest else x :: drop rest
  in
  store := drop !store

let unregister_event_listener f = unregister event_listeners f
let unregister_event_duration_listener f = unregister duration_listeners f
let unregister_event_time_span_listener f = unregister time_span_listeners f
let unregister_scalar_listener f = unregister scalar_listeners f
let get_event_listeners () = !event_listeners
let get_event_duration_listeners () = !duration_listeners
let get_event_time_span_listeners () = !time_span_listeners
let get_scalar_listeners () = !scalar_listeners

let clear_event_listeners () =
  event_listeners := [];
  duration_listeners := [];
  time_span_listeners := []

let record_event ?(kwargs = []) name =
  List.iter (fun f -> f name kwargs) !event_listeners

let record_event_duration_secs ?(kwargs = []) name duration =
  List.iter (fun f -> f name duration kwargs) !duration_listeners

let record_event_time_span ?(kwargs = []) name start_time end_time =
  List.iter (fun f -> f name start_time end_time kwargs) !time_span_listeners

let record_scalar ?(kwargs = []) name value =
  List.iter (fun f -> f name value kwargs) !scalar_listeners
