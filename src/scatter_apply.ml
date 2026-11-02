let table : (int, Obj.t * Types.closed_jaxpr * (float -> float)) Hashtbl.t =
  Hashtbl.create 16

let mark prim jaxpr host =
  let o = Obj.repr prim in
  Hashtbl.add table (Hashtbl.hash o) (o, jaxpr, host);
  prim

let find prim =
  let o = Obj.repr prim in
  List.find_map
    (fun (other, jaxpr, host) ->
      if other == o then Some (jaxpr, host) else None)
    (Hashtbl.find_all table (Hashtbl.hash o))
