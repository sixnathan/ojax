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

let marked_triples :
    (int, (int * int * int) array * (int -> int -> bool)) Hashtbl.t =
  Hashtbl.create 64

let mark_triples_where predicate arr =
  if Array.length arr > 0 then
    Hashtbl.add marked_triples (Hashtbl.hash arr) (arr, predicate);
  arr

let triple_marked arr i j =
  List.exists
    (fun (other, predicate) -> other == arr && predicate i j)
    (Hashtbl.find_all marked_triples (Hashtbl.hash arr))

let sorted_unique : (int, Obj.t) Hashtbl.t = Hashtbl.create 16

let mark_sorted_unique dims =
  let o = Obj.repr dims in
  Hashtbl.add sorted_unique (Hashtbl.hash o) o;
  dims

let is_sorted_unique dims =
  let o = Obj.repr dims in
  List.exists
    (fun other -> other == o)
    (Hashtbl.find_all sorted_unique (Hashtbl.hash o))

let clip_gathers : (int, Obj.t) Hashtbl.t = Hashtbl.create 16

let mark_clip dims =
  let o = Obj.repr dims in
  Hashtbl.add clip_gathers (Hashtbl.hash o) o;
  dims

let is_clip dims =
  let o = Obj.repr dims in
  List.exists
    (fun other -> other == o)
    (Hashtbl.find_all clip_gathers (Hashtbl.hash o))
