let table : (int, int array * (int -> bool)) Hashtbl.t = Hashtbl.create 256

let mark_where predicate arr =
  if Array.length arr > 0 then
    Hashtbl.add table (Hashtbl.hash arr) (arr, predicate);
  arr

let mark arr = mark_where (fun _ -> true) arr

let is_marked arr i =
  List.exists
    (fun (other, predicate) -> other == arr && predicate i)
    (Hashtbl.find_all table (Hashtbl.hash arr))

let marked_prims : (int, Obj.t) Hashtbl.t = Hashtbl.create 64

let mark_prim prim =
  let o = Obj.repr prim in
  Hashtbl.add marked_prims (Hashtbl.hash o) o;
  prim

let prim_marked prim =
  let o = Obj.repr prim in
  List.exists
    (fun other -> other == o)
    (Hashtbl.find_all marked_prims (Hashtbl.hash o))

let sorted_scatters : (int, Obj.t) Hashtbl.t = Hashtbl.create 64

let mark_scatter prim =
  let o = Obj.repr prim in
  Hashtbl.add sorted_scatters (Hashtbl.hash o) o;
  prim

let scatter_marked prim =
  let o = Obj.repr prim in
  List.exists
    (fun other -> other == o)
    (Hashtbl.find_all sorted_scatters (Hashtbl.hash o))

let gather_fills : (int, Obj.t * string * float option) Hashtbl.t =
  Hashtbl.create 64

let mark_gather_fill prim repr fill =
  let o = Obj.repr prim in
  Hashtbl.add gather_fills (Hashtbl.hash o) (o, repr, fill);
  prim

let gather_fill_of prim =
  let o = Obj.repr prim in
  List.find_map
    (fun (other, repr, fill) -> if other == o then Some (repr, fill) else None)
    (Hashtbl.find_all gather_fills (Hashtbl.hash o))
