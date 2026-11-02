open Types

val nonzero : ?size:int -> ?fill_value:value -> value -> value list
val fancy_take : value -> value list -> value
val flatnonzero : ?size:int -> ?fill_value:value -> value -> value
val argwhere : ?size:int -> ?fill_value:value -> value -> value

type unique_result = {
  values : value;
  indices : value;
  inverse : value;
  counts : value;
}

val unique_values : ?size:int -> ?fill_value:float -> value -> value
val unique_counts : ?size:int -> ?fill_value:float -> value -> value * value
val unique_inverse : ?size:int -> ?fill_value:float -> value -> value * value
val unique_all : ?size:int -> ?fill_value:float -> value -> unique_result

val unique :
  ?return_index:bool ->
  ?return_inverse:bool ->
  ?return_counts:bool ->
  ?axis:int ->
  ?equal_nan:bool ->
  ?size:int ->
  ?fill_value:float ->
  ?sorted:bool ->
  value ->
  value list

val union1d : ?size:int -> ?fill_value:float -> value -> value -> value

val intersect1d :
  ?assume_unique:bool ->
  ?size:int ->
  ?fill_value:float ->
  value ->
  value ->
  value

val intersect1d_indices :
  ?assume_unique:bool -> value -> value -> value * value * value

val setdiff1d :
  ?assume_unique:bool ->
  ?size:int ->
  ?fill_value:float ->
  value ->
  value ->
  value

val setxor1d :
  ?assume_unique:bool ->
  ?size:int ->
  ?fill_value:float ->
  value ->
  value ->
  value

val bincount : ?weights:value -> ?minlength:int -> ?length:int -> value -> value

val histogram_bin_edges :
  ?bins:int -> ?range:float * float -> ?weights:value -> value -> value

val histogram :
  ?bins:int ->
  ?range:float * float ->
  ?weights:value ->
  ?density:bool ->
  value ->
  value * value

val histogram_edges :
  ?weights:value -> ?density:bool -> value -> value -> value * value

val ravel_multi_index :
  ?mode:string ->
  ?order:string ->
  ?dtype:Dtype.t ->
  value list ->
  int array ->
  value

val histogramdd :
  ?bins:int array ->
  ?range:(float * float) array ->
  ?weights:value ->
  ?density:bool ->
  value ->
  value * value list

val histogramdd_uniform :
  ?bins:int ->
  ?range:(float * float) array ->
  ?weights:value ->
  ?density:bool ->
  value ->
  value * value list

val histogram2d :
  ?bins:int array ->
  ?range:(float * float) array ->
  ?weights:value ->
  ?density:bool ->
  value ->
  value ->
  value * value * value

val interp :
  ?left:float ->
  ?right:float ->
  ?period:float ->
  value ->
  value ->
  value ->
  value

val gradient :
  ?spacing:float -> ?axis:int list -> ?edge_order:int -> value -> value list

val choose : ?mode:string -> value -> value list -> value

val compress :
  ?axis:int -> ?size:int -> ?fill_value:float -> value -> value -> value

val extract : ?size:int -> ?fill_value:float -> value -> value -> value
val delete : ?axis:int -> value -> int list -> value

val delete_array :
  ?axis:int -> ?assume_unique_indices:bool -> value -> value -> value

val insert : ?axis:int -> value -> int -> value -> value
val place : ?inplace:bool -> value -> value -> value -> value
val piecewise : value -> value list -> (value -> value) list -> value
val fill_diagonal : ?wrap:bool -> ?inplace:bool -> value -> value -> value
val tril_indices : ?k:int -> ?m:int -> int -> value * value
val triu_indices : ?k:int -> ?m:int -> int -> value * value
val tril_indices_from : ?k:int -> value -> value * value
val triu_indices_from : ?k:int -> value -> value * value

val mask_indices :
  ?k:int -> ?size:int -> int -> (value -> int -> value) -> value * value

val trim_zeros : ?trim:string -> ?axis:int list -> value -> value
val sort_complex : value -> value
val bitwise_count : value -> value
val frexp : value -> value * value
val ldexp : value -> value -> value
val argpartition : ?axis:int -> value -> 'a -> value
val apply_along_axis : (value -> value) -> int -> value -> value
val apply_over_axes : (value -> int -> value) -> value -> int list -> value

type nested = Item of value | Group of nested list

val block : nested -> value
val promote_types : Dtype.t -> Dtype.t -> Dtype.t
val can_cast : ?casting:string -> Dtype.t -> Dtype.t -> bool
val shape_of : value -> int array
val size : ?axis:int array -> value -> int
val ndim : value -> int
val polydiv : ?trim_leading_zeros:bool -> value -> value -> value * value
val polyfit : ?rcond:float -> ?w:value -> value -> value -> int -> value

val polyfit_full :
  ?rcond:float ->
  ?w:value ->
  value ->
  value ->
  int ->
  value * value * value * value * value

val polyfit_cov :
  ?rcond:float ->
  ?w:value ->
  ?unscaled:bool ->
  value ->
  value ->
  int ->
  value * value

val roots : ?strip_zeros:bool -> value -> value
val fromfunction : (value list -> 'a) -> int array -> dtype:Dtype.t -> 'a
val pi : float
val e : float
val euler_gamma : float
val inf : float
val nan : float

type printoptions = {
  precision : int;
  threshold : int;
  edgeitems : int;
  suppress : bool;
}

val get_printoptions : unit -> printoptions

val set_printoptions :
  ?precision:int ->
  ?threshold:int ->
  ?edgeitems:int ->
  ?suppress:bool ->
  unit ->
  unit

val with_printoptions :
  ?precision:int ->
  ?threshold:int ->
  ?edgeitems:int ->
  ?suppress:bool ->
  (unit -> 'a) ->
  'a

val array_str : value -> string

val array_repr :
  ?max_line_width:int ->
  ?precision:int ->
  ?suppress_small:bool ->
  value ->
  string

val numpy_array_repr :
  ?max_line_width:int ->
  ?precision:int ->
  ?suppress_small:bool ->
  value ->
  string

val array_str_jax :
  ?max_line_width:int ->
  ?precision:int ->
  ?suppress_small:bool ->
  value ->
  string

val packbits : ?axis:int -> ?bitorder:string -> value -> value
val unpackbits : ?axis:int -> ?count:int -> ?bitorder:string -> value -> value
val shape : value -> int array
val broadcast_shapes : int array list -> int array
val iterable : value -> bool
val complex_ : value -> value
val dtype : ?align:bool -> string -> Dtype.t

type ndarray = value
type ufunc = Ufunc.t
type complex_warning = Complex_warning of string
