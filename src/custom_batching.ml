open Types

type vmap_rule =
  axis_size:int -> in_batched:bool list -> value list -> value list * bool list

let custom_vmap ~primal ~vmap_rule args =
  let top = Core.find_top_trace args in
  let tracers = List.map (Core.full_raise top) args in
  let outs = Core.process_custom_vmap top primal ~vmap_rule tracers in
  List.map Core.full_lower outs

let split_by mask values =
  let rec loop mapped unmapped mask values =
    match (mask, values) with
    | [], [] -> (List.rev mapped, List.rev unmapped)
    | true :: ms, x :: xs -> loop (x :: mapped) unmapped ms xs
    | false :: ms, x :: xs -> loop mapped (x :: unmapped) ms xs
    | _ -> invalid_arg "sequential_vmap: batching specification arity mismatch"
  in
  loop [] [] mask values

let merge_by mask mapped unmapped =
  let rec loop acc mask mapped unmapped =
    match (mask, mapped, unmapped) with
    | [], [], [] -> List.rev acc
    | true :: ms, x :: xs, ys -> loop (x :: acc) ms xs ys
    | false :: ms, xs, y :: ys -> loop (y :: acc) ms xs ys
    | _ -> invalid_arg "sequential_vmap: batching specification arity mismatch"
  in
  loop [] mask mapped unmapped

let sequential_vmap primal =
  let vmap_rule ~axis_size:_ ~in_batched args =
    let mapped, unmapped = split_by in_batched args in
    let body mapped_slice =
      primal (merge_by in_batched mapped_slice unmapped)
    in
    let outs = Lax.scan body [] mapped in
    (outs, List.map (fun _ -> true) outs)
  in
  custom_vmap ~primal ~vmap_rule
