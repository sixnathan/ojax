open Types

type exported = {
  mlir_module : string;
  in_avals : aval list;
  out_avals : aval list;
  platforms : string list;
}

val export : (value list -> value list) -> aval list -> exported
val serialize : exported -> string
val deserialize : string -> exported
val call : exported -> value list -> value list
