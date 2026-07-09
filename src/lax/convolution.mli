open Types

val conv_shape :
  conv_dims ->
  int array ->
  (int * int) array ->
  int array ->
  int array ->
  int ->
  int ->
  int array ->
  int array ->
  int array

val conv_impl :
  conv_dims ->
  int array ->
  (int * int) array ->
  int array ->
  int array ->
  int ->
  int ->
  Ndarray.t ->
  Ndarray.t ->
  Ndarray.t
