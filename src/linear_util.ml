exception Store_exception of string

type 'a store = { mutable v : 'a option }

let new_store () : 'a store = { v = None }

let store (s : 'a store) (x : 'a) : unit =
  match s.v with
  | Some _ -> raise (Store_exception "Store occupied")
  | None -> s.v <- Some x

let store_val (s : 'a store) : 'a =
  match s.v with Some x -> x | None -> raise (Store_exception "Store empty")

let reset (s : 'a store) : unit = s.v <- None

type ('a, 'b) t = { f_transformed : 'a -> 'b }

let wrap_init (f : 'a -> 'b) : ('a, 'b) t = { f_transformed = f }
let call_wrapped (w : ('a, 'b) t) (x : 'a) : 'b = w.f_transformed x

let transformation2 (gen : ('a -> 'b) -> 'c -> 'd) (w : ('a, 'b) t) : ('c, 'd) t
    =
  { f_transformed = gen w.f_transformed }

let transformation_with_aux2 (gen : ('a -> 'b) -> 'aux store -> 'c -> 'd)
    (w : ('a, 'b) t) : ('c, 'd) t * (unit -> 'aux) =
  let out_store = new_store () in
  ( { f_transformed = gen w.f_transformed out_store },
    fun () -> store_val out_store )

let merge_linear_aux (aux1 : unit -> 'a) (aux2 : unit -> 'a) : bool * 'a =
  match aux1 () with
  | exception Store_exception _ -> (
      match aux2 () with
      | exception Store_exception _ ->
          raise (Store_exception "neither store occupied")
      | out2 -> (false, out2))
  | out1 -> (
      match aux2 () with
      | exception Store_exception _ -> (true, out1)
      | _ -> raise (Store_exception "both stores occupied"))
