val segment_sum :
  ?num_segments:int ->
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?bucket_size:int ->
  ?mode:string ->
  Types.value ->
  Types.value ->
  Types.value

val segment_prod :
  ?num_segments:int ->
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?bucket_size:int ->
  ?mode:string ->
  Types.value ->
  Types.value ->
  Types.value

val segment_max :
  ?num_segments:int ->
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?bucket_size:int ->
  ?mode:string ->
  Types.value ->
  Types.value ->
  Types.value

val segment_min :
  ?num_segments:int ->
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?bucket_size:int ->
  ?mode:string ->
  Types.value ->
  Types.value ->
  Types.value
