open Types

let prod a = Array.fold_left ( * ) 1 a

let py_tuple a =
  match Array.to_list a with
  | [ x ] -> Printf.sprintf "(%d,)" x
  | l -> "(" ^ String.concat ", " (List.map string_of_int l) ^ ")"

let type_error msg = raise (Errors.Jax_type_error msg)

let check_conv_window (dn : conv_dims) rhs_shape =
  let n = Array.length dn.rhs_spec - 2 in
  for i = 0 to n - 1 do
    let w = rhs_shape.(dn.rhs_spec.(i + 2)) in
    if w <= 0 then
      invalid_arg
        (Printf.sprintf
           "Cannot lower jaxpr with verifier errors:\n\
            \texpects window to have positive value for %d-th window \
            dimension, but got %d."
           i w)
  done

let check_shapelike ?(non_zero_shape = false) fun_name arg_name obj =
  let lower, bound =
    if non_zero_shape then (1, "strictly positive") else (0, "nonnegative")
  in
  if Array.exists (fun d -> d < lower) obj then
    type_error
      (Printf.sprintf "%s %s must have every element be %s, got %s." fun_name
         arg_name bound (py_tuple obj))

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

let distinct a =
  List.length (List.sort_uniq compare (Array.to_list a)) = Array.length a

let overlap a b = Array.exists (fun x -> Array.mem x b) a

let check_dot_general (dd : dot_dims) lhs_shape rhs_shape =
  let lhs_ndim = Array.length lhs_shape and rhs_ndim = Array.length rhs_shape in
  let out_of_range ndim d = Array.exists (fun x -> x < 0 || x >= ndim) d in
  let fail fmt = Printf.ksprintf type_error fmt in
  let t = py_tuple in
  let lc = dd.lhs_contract
  and rc = dd.rhs_contract
  and lb = dd.lhs_batch
  and rb = dd.rhs_batch in
  if out_of_range lhs_ndim lc || out_of_range lhs_ndim lb then
    fail
      "dot_general requires lhs dimension numbers to be nonnegative and less \
       than the number of axes of the lhs value, got lhs_batch of %s and \
       lhs_contracting of %s for lhs of rank %d"
      (t lb) (t lc) lhs_ndim;
  if out_of_range rhs_ndim rc || out_of_range rhs_ndim rb then
    fail
      "dot_general requires rhs dimension numbers to be nonnegative and less \
       than the number of axes of the rhs value, got rhs_batch of %s and \
       rhs_contracting of %s for rhs of rank %d"
      (t rb) (t rc) rhs_ndim;
  if Array.length lb <> Array.length rb then
    fail
      "dot_general requires equal numbers of lhs_batch and rhs_batch \
       dimensions, got lhs_batch %s and rhs_batch %s."
      (t lb) (t rb);
  if not (distinct lb) then
    fail
      "dot_general requires lhs batch dimensions to be distinct, got lhs_batch \
       %s."
      (t lb);
  if not (distinct rb) then
    fail
      "dot_general requires rhs batch dimensions to be distinct, got rhs_batch \
       %s."
      (t rb);
  if not (distinct lc) then
    fail
      "dot_general requires lhs contracting dimensions to be distinct, got \
       lhs_contracting %s."
      (t lc);
  if not (distinct rc) then
    fail
      "dot_general requires rhs contracting dimensions to be distinct, got \
       rhs_contracting %s."
      (t rc);
  if overlap lc lb then
    fail
      "dot_general requires lhs batch dimensions to be disjoint from \
       contracting dimensions, got lhs_batch %s and lhs_contracting %s."
      (t lb) (t lc);
  if overlap rc rb then
    fail
      "dot_general requires rhs batch dimensions to be disjoint from \
       contracting dimensions, got rhs_batch %s and rhs_contracting %s."
      (t rb) (t rc);
  let pick sh = Array.map (fun i -> sh.(i)) in
  let lbs = pick lhs_shape lb and rbs = pick rhs_shape rb in
  if lbs <> rbs then
    fail
      "dot_general requires lhs batch dimensions and rhs batch dimensions to \
       have the same shape, got %s and %s."
      (t lbs) (t rbs);
  let lcs = pick lhs_shape lc and rcs = pick rhs_shape rc in
  if lcs <> rcs then
    fail
      "dot_general requires contracting dimensions to have the same shape, got \
       %s and %s."
      (t lcs) (t rcs)

let dot_general_shape (dd : dot_dims) lhs_shape rhs_shape =
  check_dot_general dd lhs_shape rhs_shape;
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

let complex_bilinear f a b =
  let is_complex v =
    match (Core.get_aval v).dtype with
    | Dtype.Complex64 | Dtype.Complex128 -> true
    | _ -> false
  in
  if not (is_complex a || is_complex b) then f a b
  else
    let parts v =
      if is_complex v then (Core.bind1 Real [ v ], Some (Core.bind1 Imag [ v ]))
      else (v, None)
    in
    let ar, ai = parts a and br, bi = parts b in
    let rr = f ar br in
    let real =
      match (ai, bi) with
      | Some x, Some y -> Core.bind1 Sub [ rr; f x y ]
      | _ -> rr
    in
    let cross =
      match (ai, bi) with
      | Some x, Some y -> Core.bind1 Add [ f ar y; f x br ]
      | Some x, None -> f x br
      | None, Some y -> f ar y
      | None, None -> failwith "complex_bilinear: no complex operand"
    in
    Core.bind1 Complex [ real; cross ]
