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
