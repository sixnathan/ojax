val logsumexp :
  ?axis:int array ->
  ?b:Types.value ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val logsumexp_sign :
  ?axis:int array ->
  ?b:Types.value ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value * Types.value
