type item =
  | Int of int
  | Slice of { start : int option; stop : int option; step : int option }
  | Newaxis
  | Ellipsis

type expr = Item of item | Tuple of item list

let slice ?start ?stop ?step () = Slice { start; stop; step }
let s_ = function [ single ] -> Item single | many -> Tuple many
let index_exp items = items
let items = function Item single -> [ single ] | Tuple many -> many
let consumes = function Int _ | Slice _ -> true | Newaxis | Ellipsis -> false

let expand_ellipsis rank exprs =
  let used = List.length (List.filter consumes exprs) in
  if used > rank then
    invalid_arg
      (Printf.sprintf
         "Too many indices: array has ndim of %d, but was indexed with %d \
          non-None/Ellipsis indices."
         rank used);
  let full = List.init (rank - used) (fun _ -> slice ()) in
  match List.length (List.filter (( = ) Ellipsis) exprs) with
  | 0 -> exprs @ full
  | 1 -> List.concat_map (fun e -> if e = Ellipsis then full else [ e ]) exprs
  | _ -> invalid_arg "an index can only have a single ellipsis ('...')"

let slice_indices start stop step n =
  let step = Option.value step ~default:1 in
  if step = 0 then invalid_arg "slice step cannot be zero";
  let lower = if step < 0 then -1 else 0 in
  let upper = if step < 0 then n - 1 else n in
  let clamp_bound v =
    let v = if v < 0 then v + n else v in
    Stdlib.min upper (Stdlib.max lower v)
  in
  let first =
    match start with
    | Some v -> clamp_bound v
    | None -> if step < 0 then upper else lower
  in
  let last =
    match stop with
    | Some v -> clamp_bound v
    | None -> if step < 0 then lower else upper
  in
  let count =
    if step > 0 then if last > first then ((last - first - 1) / step) + 1 else 0
    else if first > last then ((first - last - 1) / -step) + 1
    else 0
  in
  (first, step, count)

let apply x exprs =
  let module L = Lax_api in
  let sh = (Core.get_aval x).Types.shape in
  let rank = Array.length sh in
  let exprs = expand_ellipsis rank exprs in
  let flips = ref [] in
  let starts = Array.make rank 0 in
  let limits = Array.copy sh in
  let strides = Array.make rank 1 in
  let out_shape = ref [] in
  let axis = ref 0 in
  List.iter
    (fun e ->
      match e with
      | Newaxis -> out_shape := 1 :: !out_shape
      | Ellipsis -> ()
      | Int i ->
          let d = !axis in
          let n = sh.(d) in
          let i = if i < 0 then i + n else i in
          let i = Stdlib.max 0 (Stdlib.min (n - 1) i) in
          starts.(d) <- i;
          limits.(d) <- i + 1;
          incr axis
      | Slice { start; stop; step } ->
          let d = !axis in
          let n = sh.(d) in
          let first, step, count = slice_indices start stop step n in
          out_shape := count :: !out_shape;
          if count = 0 then begin
            starts.(d) <- 0;
            limits.(d) <- 0
          end
          else if step > 0 then begin
            starts.(d) <- first;
            limits.(d) <- first + ((count - 1) * step) + 1;
            strides.(d) <- step
          end
          else begin
            flips := d :: !flips;
            let flipped = n - 1 - first in
            starts.(d) <- flipped;
            limits.(d) <- flipped + ((count - 1) * -step) + 1;
            strides.(d) <- -step
          end;
          incr axis)
    exprs;
  let flipped =
    if !flips = [] then x else L.rev x (Array.of_list (List.rev !flips))
  in
  let sliced =
    if rank = 0 then flipped else L.slice ~strides flipped starts limits
  in
  L.reshape sliced (Array.of_list (List.rev !out_shape))

let get x expr = apply x (items expr)
