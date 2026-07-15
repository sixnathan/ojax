val dct :
  ?type_:int ->
  ?n:int ->
  ?axis:int ->
  ?norm:string ->
  Types.value ->
  Types.value

val dctn :
  ?type_:int ->
  ?s:int array ->
  ?axes:int array ->
  ?norm:string ->
  Types.value ->
  Types.value

val idct :
  ?type_:int ->
  ?n:int ->
  ?axis:int ->
  ?norm:string ->
  Types.value ->
  Types.value

val idctn :
  ?type_:int ->
  ?s:int array ->
  ?axes:int array ->
  ?norm:string ->
  Types.value ->
  Types.value
