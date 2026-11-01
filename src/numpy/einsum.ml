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

let const_full dt sh x = Core.splat dt sh x
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

let einsum_pair ?precision ?preferred lhs rhs lhs_names rhs_names result =
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
    Lax_api.dot_general ?precision
      ~preferred_element_type:
        (Option.value preferred
           ~default:(Dtypes.promote_types (dtype x) (dtype y)))
      x y ~lhs_contract:lc ~rhs_contract:rc ~lhs_batch:lb ~rhs_batch:rb
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

let remove_ellipsis term = String.concat "" (String.split_on_char '.' term)
let has_ellipsis term = String.contains term '.'

let spare_letters used n =
  let is_used c = List.exists (fun t -> String.contains t c) used in
  let all =
    List.filter
      (fun c -> not (is_used c))
      (List.init 26 (fun i -> Char.chr (Char.code 'A' + i))
      @ List.init 26 (fun i -> Char.chr (Char.code 'a' + i)))
  in
  string_of_chars (List.filteri (fun i _ -> i < n) all)

let expand_ellipsis inputs output operands =
  if not (List.exists has_ellipsis (Option.to_list output @ inputs)) then
    (inputs, output, operands)
  else
    let explicit = List.map remove_ellipsis (Option.to_list output @ inputs) in
    let widths =
      List.map2
        (fun term op ->
          if has_ellipsis term then
            ndim op - String.length (remove_ellipsis term)
          else 0)
        inputs operands
    in
    let widest = List.fold_left Stdlib.max 0 widths in
    let letters = spare_letters explicit widest in
    let replace term width =
      if has_ellipsis term then
        let i = String.index term '.' in
        String.sub term 0 i
        ^ String.sub letters (widest - width) width
        ^ String.sub term (i + 3) (String.length term - i - 3)
      else term
    in
    let new_inputs = List.map2 replace inputs widths in
    let new_output =
      match output with
      | Some o -> Some (replace o widest)
      | None ->
          Some
            (letters
            ^ remove_ellipsis
                (implicit_output (List.map remove_ellipsis inputs)))
    in
    (new_inputs, new_output, operands)

let contract ?precision ?preferred optimize inputs operands result =
  if List.length inputs <> List.length operands then
    invalid_arg "einsum: operand and subscript counts differ";
  let spec = String.concat "," inputs ^ "->" ^ result in
  let _, info =
    Einsum_paths.contract_path ~optimize spec (List.map shape operands)
  in
  let take positions items =
    ( List.map (List.nth items) positions,
      List.filteri (fun i _ -> not (List.mem i positions)) items )
  in
  let final =
    List.fold_left
      (fun ops (step : Einsum_paths.contraction) ->
        let picked, rest = take step.positions ops in
        let input_str, result_names = split_arrow step.einsum_str in
        let names = String.split_on_char ',' input_str in
        let out = Option.value result_names ~default:"" in
        let merged =
          match (picked, names) with
          | [ op ], [ n0 ] -> einsum_single op n0 out
          | [ l; r ], [ nl; nr ] ->
              einsum_pair ?precision ?preferred l r nl nr out
          | _ ->
              failwith
                "jax.numpy.einsum does not support simultaneous contraction of \
                 3 or more operands. Typically this means you've passed an \
                 unsupported path to the einsum optimize parameter."
        in
        rest @ [ merged ])
      operands info.contraction_list
  in
  match final with
  | [ out ] -> out
  | _ -> invalid_arg "einsum: path did not contract every operand"

let einsum ?(optimize = Einsum_paths.Auto) ?precision ?preferred_element_type
    subscripts operands =
  let s = String.concat "" (String.split_on_char ' ' subscripts) in
  let input_part, output_opt = split_arrow s in
  let inputs = String.split_on_char ',' input_part in
  let inputs, output_opt, operands =
    expand_ellipsis inputs output_opt operands
  in
  let result =
    match output_opt with Some o -> o | None -> implicit_output inputs
  in
  Eager_jit.one "numpy.einsum._einsum"
    (optimize, precision, preferred_element_type, inputs, result) [] operands
    (fun _ operands ->
      let pdt = NL.result_type operands in
      let operands = List.map (fun v -> astype v pdt) operands in
      let out =
        contract ?precision ?preferred:preferred_element_type optimize inputs
          operands result
      in
      astype out (Option.value preferred_element_type ~default:pdt))

let einsum_path ?optimize subscripts operands =
  Einsum_paths.contract_path ?optimize subscripts (List.map shape operands)
