module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)
let prod sh = Array.fold_left ( * ) 1 sh
let astype v dt = if dtype v = dt then v else NL.astype v dt
let chars s = List.init (String.length s) (String.get s)
let string_of_chars cs = String.init (List.length cs) (List.nth cs)

let distinct s =
  let seen = Hashtbl.create 16 in
  List.filter
    (fun c ->
      if Hashtbl.mem seen c then false
      else begin
        Hashtbl.add seen c ();
        true
      end)
    (chars s)

let counts_of s =
  let tbl = Hashtbl.create 16 in
  String.iter
    (fun c ->
      let n = try Hashtbl.find tbl c with Not_found -> 0 in
      Hashtbl.replace tbl c (n + 1))
    s;
  tbl

let count tbl c = try Hashtbl.find tbl c with Not_found -> 0

let removechars s to_remove =
  string_of_chars
    (List.filter (fun c -> not (String.contains to_remove c)) (chars s))

let indices_where s c =
  List.filter (fun i -> s.[i] = c) (List.init (String.length s) Fun.id)

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (prod sh) x))

let zeros_like v = const_full (dtype v) (shape v) 0.0
let ones_bool sh = const_full D.Bool sh 1.0

let sum_axes operand axes =
  if Array.length axes = 0 then operand
  else if dtype operand = D.Bool then C.bind1 (T.Reduce_or axes) [ operand ]
  else C.bind1 (T.Reduce_sum axes) [ operand ]

let sum_uniques operand names uniques =
  if uniques = [] then (operand, names)
  else begin
    let axes =
      Array.of_list (List.map (fun c -> String.index names c) uniques)
    in
    (sum_axes operand axes, removechars names (string_of_chars uniques))
  end

let delta sh axes =
  match axes with
  | [] | [ _ ] -> ones_bool sh
  | a0 :: rest ->
      List.fold_left
        (fun acc ak ->
          let ia =
            C.bind1 (T.Iota { dtype = D.I32; shape = sh; dimension = a0 }) []
          in
          let ik =
            C.bind1 (T.Iota { dtype = D.I32; shape = sh; dimension = ak }) []
          in
          let eqk = C.bind1 T.Eq [ ia; ik ] in
          C.bind1 T.And [ acc; eqk ])
        (ones_bool sh) rest

let remove_n s c n =
  let removed = ref 0 in
  string_of_chars
    (List.filter
       (fun ch ->
         if ch = c && !removed < n then begin
           incr removed;
           false
         end
         else true)
       (chars s))

let sum_repeats operand names keep =
  let orig = counts_of names in
  let order = distinct names in
  List.fold_left
    (fun (operand, names) name ->
      if count orig name > 1 then begin
        let axes = indices_where names name in
        let eye = delta (shape operand) axes in
        let operand = NL.where_ eye operand (zeros_like operand) in
        if not (String.contains keep name) then
          ( sum_axes operand (Array.of_list axes),
            remove_n names name (String.length names) )
        else begin
          let all_but_last =
            match List.rev axes with
            | [] -> [||]
            | _ :: t -> Array.of_list (List.rev t)
          in
          ( sum_axes operand all_but_last,
            remove_n names name (count orig name - 1) )
        end
      end
      else (operand, names))
    (operand, names) order

let final_transpose operand names result =
  if names = result then operand
  else begin
    let perm =
      Array.of_list (List.map (fun c -> String.index names c) (chars result))
    in
    NL.transpose ~axes:perm operand
  end

let einsum_single operand names result =
  let cnts = counts_of names in
  let contracted =
    List.sort Char.compare
      (List.filter (fun c -> not (String.contains result c)) (distinct names))
  in
  let uniques = List.filter (fun c -> count cnts c = 1) contracted in
  let operand, names = sum_uniques operand names uniques in
  let operand, names = sum_repeats operand names result in
  final_transpose operand names result

let filter_singleton operand names other_shape other_names =
  let sh = shape operand in
  let n = Array.length sh in
  let keep =
    Array.init n (fun i ->
        let c = names.[i] in
        match String.index_opt other_names c with
        | None -> true
        | Some j -> (not (sh.(i) = 1)) || other_shape.(j) = 1)
  in
  let sqez = ref [] and kept = ref [] in
  Array.iteri
    (fun i k -> if k then kept := i :: !kept else sqez := i :: !sqez)
    keep;
  let sqez_axes = Array.of_list (List.rev !sqez) in
  let kept_axes = List.rev !kept in
  let operand =
    if Array.length sqez_axes = 0 then operand
    else NL.squeeze ~axis:sqez_axes operand
  in
  (operand, string_of_chars (List.map (fun i -> names.[i]) kept_axes))

let einsum_pair lhs rhs lhs_names rhs_names result =
  let lhs, lhs_names = filter_singleton lhs lhs_names (shape rhs) rhs_names in
  let rhs, rhs_names = filter_singleton rhs rhs_names (shape lhs) lhs_names in
  let lhs_counts = counts_of lhs_names and rhs_counts = counts_of rhs_names in
  let all_names = distinct (lhs_names ^ rhs_names) in
  let contracted0 =
    List.sort Char.compare
      (List.filter (fun c -> not (String.contains result c)) all_names)
  in
  let lhs_uniques =
    List.filter
      (fun c -> count lhs_counts c = 1 && count rhs_counts c = 0)
      contracted0
  in
  let lhs, lhs_names = sum_uniques lhs lhs_names lhs_uniques in
  let rhs_uniques =
    List.filter
      (fun c -> count rhs_counts c = 1 && count lhs_counts c = 0)
      contracted0
  in
  let rhs, rhs_names = sum_uniques rhs rhs_names rhs_uniques in
  let lhs, lhs_names = sum_repeats lhs lhs_names (result ^ rhs_names) in
  let rhs, rhs_names = sum_repeats rhs rhs_names (result ^ lhs_names) in
  let contracted =
    List.filter
      (fun c -> String.contains lhs_names c || String.contains rhs_names c)
      contracted0
  in
  let batch_names =
    List.filter
      (fun c -> String.contains lhs_names c && String.contains rhs_names c)
      (chars result)
  in
  let idx names c = String.index names c in
  let lhs_batch = Array.of_list (List.map (idx lhs_names) batch_names) in
  let rhs_batch = Array.of_list (List.map (idx rhs_names) batch_names) in
  let lhs_cont = Array.of_list (List.map (idx lhs_names) contracted) in
  let rhs_cont = Array.of_list (List.map (idx rhs_names) contracted) in
  let batch_str = string_of_chars batch_names in
  let deleted = batch_str ^ string_of_chars contracted in
  let rem_lhs = removechars lhs_names deleted in
  let rem_rhs = removechars rhs_names deleted in
  let names_rl = batch_str ^ rem_rhs ^ rem_lhs in
  let dg lc rc lb rb x y =
    C.bind1
      (T.Dot_general
         {
           lhs_contract = lc;
           rhs_contract = rc;
           lhs_batch = lb;
           rhs_batch = rb;
         })
      [ x; y ]
  in
  let operand, names =
    if names_rl = result then
      (dg rhs_cont lhs_cont rhs_batch lhs_batch rhs lhs, names_rl)
    else
      let names_lr = batch_str ^ rem_lhs ^ rem_rhs in
      (dg lhs_cont rhs_cont lhs_batch rhs_batch lhs rhs, names_lr)
  in
  final_transpose operand names result

let split_arrow s =
  let n = String.length s in
  let rec find i =
    if i + 1 >= n then None
    else if s.[i] = '-' && s.[i + 1] = '>' then Some i
    else find (i + 1)
  in
  match find 0 with
  | Some i -> (String.sub s 0 i, Some (String.sub s (i + 2) (n - i - 2)))
  | None -> (s, None)

let implicit_output inputs =
  let all = String.concat "" inputs in
  let cnts = counts_of all in
  let once = List.filter (fun c -> count cnts c = 1) (distinct all) in
  string_of_chars (List.sort Char.compare once)

let einsum subscripts operands =
  let s = String.concat "" (String.split_on_char ' ' subscripts) in
  if String.contains s '.' then
    invalid_arg "einsum: ellipsis subscripts not supported";
  let input_part, output_opt = split_arrow s in
  let inputs = String.split_on_char ',' input_part in
  let result =
    match output_opt with Some o -> o | None -> implicit_output inputs
  in
  let pdt = NL.result_type operands in
  let operands = List.map (fun v -> astype v pdt) operands in
  match (operands, inputs) with
  | [ op ], [ n0 ] -> einsum_single op n0 result
  | [ a; b ], [ na; nb ] -> einsum_pair a b na nb result
  | _ ->
      invalid_arg
        "einsum: only one or two operands supported (n>2 needs the opt_einsum \
         contraction path)"
