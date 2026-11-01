type options = {
  precision : int;
  threshold : int;
  edgeitems : int;
  suppress : bool;
  linewidth : int;
}

val default_options : options

val array2string :
  ?options:options ->
  ?separator:string ->
  ?prefix:string ->
  ?suffix:string ->
  Ndarray.t ->
  string

val array_repr : ?options:options -> Ndarray.t -> string
val array_str : ?options:options -> Ndarray.t -> string
