type t = Types.value -> shape:int array -> Types.value
type initializer_ = t
type mode = Fan_in | Fan_out | Fan_avg | Fan_geo_avg
type distribution = Truncated_normal | Normal | Uniform

val zeros : t
val ones : t

val zeros_dtype :
  ?dtype:Dtype.t -> Types.value -> shape:int array -> Types.value

val ones_dtype : ?dtype:Dtype.t -> Types.value -> shape:int array -> Types.value
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

val xavier_uniform :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val xavier_normal :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val kaiming_uniform :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val kaiming_normal :
  ?in_axis:int -> ?out_axis:int -> ?batch_axis:int array -> unit -> t

val orthogonal : ?scale:float -> ?column_axis:int -> unit -> t
val delta_orthogonal : ?scale:float -> ?column_axis:int -> unit -> t
val with_dtype : Dtype.t -> t -> t
