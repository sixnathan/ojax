type builder

val create : int -> builder
val start_object : builder -> int -> unit
val end_object : builder -> int
val add_int_slot : builder -> int -> int -> int -> unit
val add_offset_slot : builder -> int -> int -> unit
val prepend_u8 : builder -> int -> unit
val prepend_u16 : builder -> int -> unit
val prepend_u32 : builder -> int -> unit
val start_vector : builder -> int -> int -> int -> int
val end_vector : builder -> int
val create_string : builder -> string -> int
val create_byte_vector : builder -> string -> int
val create_uint_vector : builder -> int -> int list -> int
val offsets_vector : builder -> int list -> int
val finish : builder -> int -> unit
val output : builder -> string

type table

val root : string -> table
val has : table -> int -> bool
val get_int : table -> int -> int -> default:int -> int
val get_int8 : table -> int -> int
val vector_length : table -> int -> int
val get_string : table -> int -> string option
val get_bytes : table -> int -> string
val get_table : table -> int -> table option
val tables : table -> int -> table list
val strings : table -> int -> string list
val uints : table -> int -> int -> int list
