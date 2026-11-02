val needs_check : unit -> bool
val special_of : Ndarray.t -> string option
val check : string -> Ndarray.t list -> unit
val notice_sink : (string -> unit) ref
val check_jit : name:string -> rerun:(unit -> unit) -> Ndarray.t list -> unit
