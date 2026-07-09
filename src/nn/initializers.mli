type t = Types.value -> shape:int array -> Types.value
type mode = Fan_in | Fan_out | Fan_avg | Fan_geo_avg
type distribution = Truncated_normal | Normal | Uniform

val zeros : t
val ones : t
val constant : float -> t
val uniform : ?scale:float -> unit -> t
val normal : ?stddev:float -> unit -> t

val truncated_normal :
  ?stddev:float -> ?lower:float -> ?upper:float -> unit -> t

val variance_scaling :
  float ->
  mode ->
  distribution ->
  ?in_axis:int ->
  ?out_axis:int ->
  ?batch_axis:int array ->
  unit ->
  t

val glorot_uniform :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val glorot_normal :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val lecun_uniform :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val lecun_normal :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val he_uniform :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val he_normal :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val orthogonal : ?scale:float -> ?column_axis:int -> unit -> t
val delta_orthogonal : ?scale:float -> ?column_axis:int -> unit -> t
