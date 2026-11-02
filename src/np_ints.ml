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

let triples : (int, (int * int * int) array * (int -> int -> bool)) Hashtbl.t =
  Hashtbl.create 64

let mark_triples predicate arr =
  if Array.length arr > 0 then
    Hashtbl.add triples (Hashtbl.hash arr) (arr, predicate);
  arr

let triple_marked arr i j =
  List.exists
    (fun (other, predicate) -> other == arr && predicate i j)
    (Hashtbl.find_all triples (Hashtbl.hash arr))

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

let clip_gathers : (int, Obj.t * bool) Hashtbl.t = Hashtbl.create 64

let mark_gather_clip ~sorted prim =
  let o = Obj.repr prim in
  Hashtbl.add clip_gathers (Hashtbl.hash o) (o, sorted);
  prim

let gather_clip_of prim =
  let o = Obj.repr prim in
  List.find_map
    (fun (other, sorted) -> if other == o then Some sorted else None)
    (Hashtbl.find_all clip_gathers (Hashtbl.hash o))

let gather_flags : (int, Obj.t * bool * bool) Hashtbl.t = Hashtbl.create 16

let mark_gather_flags ~sorted ~unique prim =
  let o = Obj.repr prim in
  Hashtbl.add gather_flags (Hashtbl.hash o) (o, sorted, unique);
  prim

let gather_flags_of prim =
  let o = Obj.repr prim in
  List.find_map
    (fun (other, sorted, unique) ->
      if other == o then Some (sorted, unique) else None)
    (Hashtbl.find_all gather_flags (Hashtbl.hash o))

let scatter_flags : (int, Obj.t * bool * bool) Hashtbl.t = Hashtbl.create 16

let mark_scatter_flags ~sorted ~unique prim =
  let o = Obj.repr prim in
  Hashtbl.add scatter_flags (Hashtbl.hash o) (o, sorted, unique);
  prim

let scatter_flags_of prim =
  let o = Obj.repr prim in
  List.find_map
    (fun (other, sorted, unique) ->
      if other == o then Some (sorted, unique) else None)
    (Hashtbl.find_all scatter_flags (Hashtbl.hash o))

let one_hot_scatters : (int, Obj.t) Hashtbl.t = Hashtbl.create 16

let mark_scatter_one_hot prim =
  let o = Obj.repr prim in
  Hashtbl.add one_hot_scatters (Hashtbl.hash o) o;
  prim

let scatter_one_hot prim =
  let o = Obj.repr prim in
  List.exists
    (fun other -> other == o)
    (Hashtbl.find_all one_hot_scatters (Hashtbl.hash o))

let copy_scatter_marks ~from prim =
  let prim =
    match scatter_flags_of from with
    | Some (sorted, unique) -> mark_scatter_flags ~sorted ~unique prim
    | None -> prim
  in
  let prim = if scatter_marked from then mark_scatter prim else prim in
  let prim = if prim_marked from then mark_prim prim else prim in
  if scatter_one_hot from then mark_scatter_one_hot prim else prim
