open Types

val nonzero : ?size:int -> ?fill_value:int -> value -> value list
val flatnonzero : ?size:int -> ?fill_value:int -> value -> value
val argwhere : ?size:int -> ?fill_value:int -> value -> value

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
  ?size:int ->
  ?fill_value:float ->
  value ->
  value list

val union1d : value -> value -> value
val intersect1d : value -> value -> value
val setdiff1d : value -> value -> value
val setxor1d : value -> value -> value
val bincount : ?weights:value -> ?length:int -> value -> value
val histogram_bin_edges : ?bins:int -> ?range:float * float -> value -> value

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
  ?mode:string -> ?order:string -> value list -> int array -> value

val histogramdd :
  ?bins:int array ->
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

val gradient : ?spacing:float -> ?axis:int list -> value -> value list
val choose : ?mode:string -> value -> value list -> value
val compress : ?axis:int -> value -> value -> value
val extract : value -> value -> value
val delete : ?axis:int -> value -> int list -> value
val insert : ?axis:int -> value -> int -> value -> value
val place : value -> value -> value -> value
val piecewise : value -> value list -> (value -> value) list -> value
val fill_diagonal : ?wrap:bool -> value -> value -> value
val tril_indices : ?k:int -> ?m:int -> int -> value * value
val triu_indices : ?k:int -> ?m:int -> int -> value * value
val tril_indices_from : ?k:int -> value -> value * value
val triu_indices_from : ?k:int -> value -> value * value
val mask_indices : ?k:int -> int -> (value -> int -> value) -> value * value
val trim_zeros : ?trim:string -> value -> value
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
val can_cast : Dtype.t -> Dtype.t -> bool
val shape_of : value -> int array
val size : value -> int
val ndim : value -> int
val polydiv : value -> value -> value * value
val polyfit : ?rcond:float -> value -> value -> int -> value
val roots : ?strip_zeros:bool -> value -> value
val fromfunction : (value list -> 'a) -> int array -> dtype:Dtype.t -> 'a
val pi : float
val e : float
val euler_gamma : float
val inf : float
val nan : float
val array_str : value -> string
val array_repr : value -> string
