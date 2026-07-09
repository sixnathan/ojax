type resize_method =
  | Nearest
  | Linear
  | Lanczos3
  | Lanczos5
  | Cubic
  | Cubic_pytorch
  | Area

type kernel = Triangle | Lanczos of float | Keys_cubic | Area_kernel

val from_string : string -> resize_method
val kernels : resize_method -> int * kernel

val compute_weight_mat :
  input_size:int ->
  output_size:int ->
  scale:float ->
  translation:float ->
  kernel:kernel ->
  antialias:bool ->
  edge_padding:bool ->
  radius:int ->
  Types.value

val scale_and_translate :
  Types.value ->
  shape:int array ->
  spatial_dims:int array ->
  scale:float array ->
  translation:float array ->
  method_:resize_method ->
  ?antialias:bool ->
  unit ->
  Types.value

val resize :
  Types.value ->
  shape:int array ->
  method_:resize_method ->
  ?antialias:bool ->
  unit ->
  Types.value
