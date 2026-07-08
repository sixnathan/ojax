open Types

let prod a = Array.fold_left ( * ) 1 a

let strides shape =
  let n = Array.length shape in
  let s = Array.make (max n 1) 1 in
  for i = n - 2 downto 0 do
    s.(i) <- s.(i + 1) * shape.(i + 1)
  done;
  if n = 0 then [||] else Array.sub s 0 n

let decode flat shape =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let r = ref flat in
  for d = n - 1 downto 0 do
    idx.(d) <- !r mod shape.(d);
    r := !r / shape.(d)
  done;
  idx

let mem arr x = Array.exists (fun a -> a = x) arr

let free_axes ndim batch contract =
  let out = ref [] in
  for d = ndim - 1 downto 0 do
    if (not (mem batch d)) && not (mem contract d) then out := d :: !out
  done;
  Array.of_list !out

let reduce_shape shape axes =
  let n = Array.length shape in
  let out = ref [] in
  for d = n - 1 downto 0 do
    if not (mem axes d) then out := shape.(d) :: !out
  done;
  Array.of_list !out

let dot_general_shape (dd : dot_dims) lhs_shape rhs_shape =
  let lhs_free =
    free_axes (Array.length lhs_shape) dd.lhs_batch dd.lhs_contract
  in
  let rhs_free =
    free_axes (Array.length rhs_shape) dd.rhs_batch dd.rhs_contract
  in
  let batch = Array.map (fun a -> lhs_shape.(a)) dd.lhs_batch in
  let lf = Array.map (fun a -> lhs_shape.(a)) lhs_free in
  let rf = Array.map (fun a -> rhs_shape.(a)) rhs_free in
  Array.concat [ batch; lf; rf ]

let all_weak avals = List.for_all (fun a -> a.weak_type) avals
let dilate_dim d f = if d = 0 then 0 else ((d - 1) * f) + 1

let pad_shape (cfg : (int * int * int) array) in_shape =
  Array.mapi
    (fun i d ->
      let lo, hi, interior = cfg.(i) in
      lo + hi + dilate_dim d (interior + 1))
    in_shape

let concatenate_shape dim shapes =
  match shapes with
  | [] -> [||]
  | first :: _ ->
      let total = List.fold_left (fun acc s -> acc + s.(dim)) 0 shapes in
      Array.mapi (fun i d -> if i = dim then total else d) first

let insert_int arr idx v =
  let n = Array.length arr in
  Array.init (n + 1) (fun i ->
      if i < idx then arr.(i) else if i = idx then v else arr.(i - 1))

let remove_int arr idx =
  let n = Array.length arr in
  Array.init (n - 1) (fun i -> if i < idx then arr.(i) else arr.(i + 1))

let stack_shape axis n in_shape = insert_int in_shape axis n
let tile_shape reps in_shape = Array.mapi (fun i d -> d * reps.(i)) in_shape
let transpose_shape perm in_shape = Array.map (fun p -> in_shape.(p)) perm

let squeeze_shape dims in_shape =
  let is_dropped = Array.make (Array.length in_shape) false in
  Array.iter (fun d -> is_dropped.(d) <- true) dims;
  let out = ref [] in
  for i = Array.length in_shape - 1 downto 0 do
    if not is_dropped.(i) then out := in_shape.(i) :: !out
  done;
  Array.of_list !out

let split_shapes sizes axis in_shape =
  Array.to_list
    (Array.map
       (fun size ->
         Array.mapi (fun i d -> if i = axis then size else d) in_shape)
       sizes)

let unstack_shapes axis in_shape =
  let sub = remove_int in_shape axis in
  List.init in_shape.(axis) (fun _ -> Array.copy sub)

let argsort perm =
  let n = Array.length perm in
  let out = Array.make n 0 in
  Array.iteri (fun i p -> out.(p) <- i) perm;
  out
