val sink : (string -> string -> unit) ref
val warn : ?category:string -> string -> unit
val catch : (unit -> 'a) -> 'a * (string * string) list
