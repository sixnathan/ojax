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
