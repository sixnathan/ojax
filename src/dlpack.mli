open Types

type capsule
type placement = On_device of string | With_sharding of Sharding.t

type external_array = {
  dlpack : unit -> capsule;
  dlpack_device : unit -> int * int;
}

val dlpack_version : int * int
val min_dlpack_version : int * int
val is_supported_dtype : Dtype.t -> bool
val is_supported_dtype_name : string -> bool
val capsule_of_pointer : nativeint -> capsule
val capsule_pointer : capsule -> nativeint
val capsule_used : capsule -> bool

val to_dlpack :
  ?dl_device:int * int ->
  ?max_version:int * int ->
  ?copy:bool ->
  value ->
  capsule

val external_of_value : value -> external_array
val from_dlpack : ?device:placement -> ?copy:bool -> external_array -> value
val unsafe_buffer_pointer : value -> nativeint
