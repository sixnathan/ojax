val initialize :
  ?coordinator_address:string ->
  ?num_processes:int ->
  ?process_id:int ->
  unit ->
  unit

val is_initialized : unit -> bool
val shutdown : unit -> unit
