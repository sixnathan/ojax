type raw

type named = {
  mesh_axes : (string * int) list;
  mesh_devices : int array;
  spec : string list array;
}

type layout = On_device of int | Named of named | Tiled

type shard = {
  device : int;
  index : (int * int) array;
  replica_id : int;
  data : raw;
}

type placed = {
  global_shape : int array;
  dtype : Dtype.t;
  shards : shard array;
  layout : layout;
}

type t = Single of raw | Placed of placed

val of_host : Client.t -> Ndarray.t -> t
val to_host : t -> Ndarray.t
val dimensions : t -> int array
val element_type : t -> Dtype.t
val destroy : t -> unit
val to_dlpack : t -> nativeint
val of_dlpack : Client.t -> nativeint -> int -> t
val dlpack_delete : nativeint -> unit
val unsafe_pointer : t -> nativeint
val belongs_to : t -> Client.t -> bool
val raw_dimensions : raw -> int array
val raw_device : raw -> int
val raw_to_host : raw -> Ndarray.t

val shards_of_host :
  Client.t -> Ndarray.t -> ((int * int) array * int) array -> raw array

val single_raw : t -> raw
