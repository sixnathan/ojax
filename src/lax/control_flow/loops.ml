open Types

let split_at lst n =
  match Util.split_list lst [ n ] with
  | [ a; b ] -> (a, b)
  | _ -> failwith "loops: split_at"

let mapped_leading (a : aval) : aval =
  let n = Array.length a.shape in
  { a with shape = Array.sub a.shape 1 (n - 1) }

let unmapped_leading (length : int) (a : aval) : aval =
  { a with shape = Array.append [| length |] a.shape }

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }
  | DropVar a -> a

let scan_out_avals ~length ~num_carry (jaxpr : closed_jaxpr) : aval list =
  let outs = List.map aval_of_atom jaxpr.jaxpr.outs in
  let carry_out, y_slices = split_at outs num_carry in
  carry_out @ List.map (unmapped_leading length) y_slices

let index_leading (i : int) (v : value) : value =
  let a = Core.get_aval v in
  let sh = a.shape in
  let n = Array.length sh in
  let start = Array.init n (fun k -> if k = 0 then i else 0) in
  let limit = Array.init n (fun k -> if k = 0 then i + 1 else sh.(k)) in
  let sliced =
    Core.bind1
      (Slice { start_indices = start; limit_indices = limit; strides = None })
      [ v ]
  in
  Core.bind1 (Squeeze [| 0 |]) [ sliced ]

let scan_impl ~length ~reverse ~num_carry (jaxpr : closed_jaxpr)
    (inputs : value list) : value list =
  let carry0, xs = split_at inputs num_carry in
  let num_ys = List.length jaxpr.jaxpr.outs - num_carry in
  let ys_by_iter = Array.make length [] in
  let carry = ref carry0 in
  let order =
    if reverse then List.init length (fun i -> length - 1 - i)
    else List.init length (fun i -> i)
  in
  List.iter
    (fun i ->
      let x_slice = List.map (index_leading i) xs in
      let outs = Jaxpr.eval_closed_jaxpr jaxpr (!carry @ x_slice) in
      let c', y = split_at outs num_carry in
      carry := c';
      ys_by_iter.(i) <- y)
    order;
  let stacked =
    List.init num_ys (fun j ->
        let column = List.init length (fun i -> List.nth ys_by_iter.(i) j) in
        Core.bind1 (Stack 0) column)
  in
  !carry @ stacked

let scan ?(reverse = false) (body : value list -> value list)
    (init : value list) (xs : value list) : value list =
  let num_carry = List.length init in
  let length =
    match xs with
    | x :: _ -> (Core.get_aval x).shape.(0)
    | [] -> failwith "loops: scan requires at least one xs input (M2)"
  in
  let carry_avals = List.map Core.get_aval init in
  let x_slice_avals = List.map (fun x -> mapped_leading (Core.get_aval x)) xs in
  let jaxpr = Jaxpr.make_jaxpr (carry_avals @ x_slice_avals) body in
  Core.bind (Scan { length; reverse; num_carry; jaxpr }) (init @ xs)
