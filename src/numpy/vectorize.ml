module T = Types
module C = Core
module LN = Lax_numpy

let get_aval = C.get_aval
let shape_of v = (get_aval v).T.shape

let parse_side (s : string) : string list list =
  let n = String.length s in
  let groups = ref [] in
  let i = ref 0 in
  while !i < n do
    if s.[!i] = '(' then (
      let j = String.index_from s !i ')' in
      let inner = String.sub s (!i + 1) (j - !i - 1) in
      let dims =
        if String.trim inner = "" then []
        else List.map String.trim (String.split_on_char ',' inner)
      in
      groups := dims :: !groups;
      i := j + 1)
    else incr i
  done;
  List.rev !groups

let parse_signature (s : string) : int list * int list =
  let arrow =
    match String.index_opt s '-' with
    | Some k when k + 1 < String.length s && s.[k + 1] = '>' -> k
    | _ -> invalid_arg ("not a valid gufunc signature: " ^ s)
  in
  let lhs = String.sub s 0 arrow in
  let rhs = String.sub s (arrow + 2) (String.length s - arrow - 2) in
  (List.map List.length (parse_side lhs), List.map List.length (parse_side rhs))

let vectorize ?(excluded = []) ?signature pyfunc raw_args =
  let n = List.length raw_args in
  let args_arr = Array.of_list raw_args in
  let is_excl i = List.mem i excluded in
  let dynamic = List.filteri (fun i _ -> not (is_excl i)) raw_args in
  let call_full dyn_slices =
    let dyn = ref dyn_slices in
    let full =
      List.init n (fun i ->
          if is_excl i then args_arr.(i)
          else
            match !dyn with
            | x :: r ->
                dyn := r;
                x
            | [] -> assert false)
    in
    pyfunc full
  in
  let input_ncs, output_ncs =
    match signature with
    | Some s -> parse_signature s
    | None -> (List.map (fun _ -> 0) dynamic, [ 0 ])
  in
  let dyn_arr = Array.of_list dynamic in
  let ncs = Array.of_list input_ncs in
  if Array.length dyn_arr <> Array.length ncs then
    invalid_arg "vectorize: wrong number of positional arguments";
  if List.length output_ncs <> 1 then
    failwith "vectorize: only single-output gufuncs supported";
  let batch_shapes =
    Array.to_list
      (Array.mapi
         (fun idx v ->
           let sh = shape_of v in
           let ndim = Array.length sh in
           let nc = ncs.(idx) in
           if ndim < nc then
             invalid_arg
               "vectorize: input has too few dimensions for its core dimensions";
           Array.sub sh 0 (ndim - nc))
         dyn_arr)
  in
  let broadcast_shape = LN.broadcast_shapes_n batch_shapes in
  let rank = Array.length broadcast_shape in
  let squeezed =
    List.map2
      (fun v noncore ->
        let dims =
          List.filter
            (fun i -> noncore.(i) = 1)
            (List.init (Array.length noncore) Fun.id)
        in
        if dims = [] then v else LN.squeeze ~axis:(Array.of_list dims) v)
      (Array.to_list dyn_arr) batch_shapes
  in
  let filled =
    List.map
      (fun noncore ->
        Array.append (Array.make (rank - Array.length noncore) 1) noncore)
      batch_shapes
  in
  let vectorized = ref (fun slices -> [ call_full slices ]) in
  let to_expand = ref [] in
  for negdim = 0 to rank - 1 do
    let axis = rank - 1 - negdim in
    let in_axes =
      List.map (fun sh -> if sh.(axis) = 1 then None else Some 0) filled
    in
    if List.for_all Option.is_none in_axes then
      to_expand := !to_expand @ [ axis ]
    else
      let inner = !vectorized in
      vectorized := fun args -> Interpreters.Batching.vmap inner in_axes args
  done;
  let result = List.hd (!vectorized squeezed) in
  match !to_expand with
  | [] -> result
  | dims -> LN.expand_dims result (Array.of_list dims)
