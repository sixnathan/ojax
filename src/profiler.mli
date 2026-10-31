type stat_value =
  | Stat_int of int64
  | Stat_uint of int64
  | Stat_float of float
  | Stat_text of string
  | Stat_bytes of string

type profile_event = {
  event_name : string;
  start_ns : float;
  end_ns : float;
  duration_ns : float;
  event_stats : (string * stat_value) list;
}

type profile_line = { line_name : string; events : profile_event list }

type profile_plane = {
  plane_name : string;
  lines : profile_line list;
  plane_stats : (string * stat_value) list;
}

type profile_data = { planes : profile_plane list }

type profile_options = {
  host_tracer_level : int;
  python_tracer_level : int;
  device_tracer_level : int;
  enable_hlo_proto : bool;
  include_dataset_ops : bool;
  raise_error_on_start_failure : bool;
  start_timestamp_ns : int;
  duration_ms : int;
  repository_path : string;
  session_id : string;
  advanced_configuration : (string * string) list;
}

type profiler_server

val profile_options : unit -> profile_options
val profile_data_from_serialized_xspace : string -> profile_data
val profile_data_from_file : string -> profile_data
val find_plane_with_name : profile_data -> string -> profile_plane option

val trace_annotation :
  ?stats:(string * stat_value) list -> string -> (unit -> 'a) -> 'a

val step_trace_annotation : string -> step_num:int -> (unit -> 'a) -> 'a
val annotate_function : 'a -> 'a
val start_trace : ?profiler_options:profile_options -> string -> unit
val stop_trace : unit -> unit
val trace : ?profiler_options:profile_options -> string -> (unit -> 'a) -> 'a
val start_server : ?requires_backend:bool -> int -> profiler_server
val stop_server : unit -> unit
val server_port : profiler_server -> int
val register_subprocess : int -> int -> unit -> unit
val device_memory_profile : unit -> string
val save_device_memory_profile : string -> unit
