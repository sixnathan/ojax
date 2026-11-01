open Types

let lift_consts ((c : closed_jaxpr), hoisted, _) =
  let n = List.length c.consts in
  let binders = c.jaxpr.in_binders in
  let total = List.length binders in
  let nh = List.length hoisted in
  let const_part = List.filteri (fun i _ -> i < n) binders in
  let hoist_part = List.filteri (fun i _ -> i >= total - nh) binders in
  let arg_part = List.filteri (fun i _ -> i >= n && i < total - nh) binders in
  ( {
      c with
      consts = [];
      jaxpr = { c.jaxpr with in_binders = const_part @ hoist_part @ arg_part };
    },
    List.map (fun nd -> Concrete nd) c.consts @ hoisted )

let build_branches all_avals lifted hoisted_each =
  let n_hoisted = List.length (List.concat hoisted_each) in
  let rec offsets acc = function
    | [] -> []
    | h :: tl -> acc :: offsets (acc + List.length h) tl
  in
  List.map2
    (fun ((c : closed_jaxpr), own) offset ->
      let n_own = List.length own in
      Jaxpr.make_jaxpr all_avals (fun args ->
          let ops = List.filteri (fun i _ -> i >= n_hoisted) args in
          let mine =
            List.filteri (fun i _ -> i >= offset && i < offset + n_own) args
          in
          Jaxpr.eval_closed_jaxpr c (mine @ ops)))
    lifted (offsets 0 hoisted_each)

let rec branch_call ?platforms (index : value)
    (branches : (value list -> value list) list) (operands : value list) :
    value list =
  let avals = List.map Core.get_aval operands in
  let traced = List.map (Jaxpr.make_jaxpr_stateful avals) branches in
  let cells =
    List.fold_left
      (fun acc (_, _, branch_cells) ->
        acc
        @ List.filter
            (fun (c : Ref_state.cell) -> not (List.memq c acc))
            branch_cells)
      [] traced
  in
  if cells <> [] then begin
    let n = List.length operands in
    let stateful branch args =
      match Util.split_list args [ n ] with
      | [ ops; state ] ->
          Ref_state.pin cells state;
          let outs = branch ops in
          outs @ Ref_state.contents cells
      | _ -> failwith "switch: arity"
    in
    let outs =
      branch_call ?platforms index
        (List.map stateful branches)
        (operands @ List.map Ref_state.read cells)
    in
    let n_out = List.length outs - List.length cells in
    match Util.split_list outs [ n_out ] with
    | [ results; finals ] ->
        List.iter2 Ref_state.write cells finals;
        results
    | _ -> failwith "switch: arity"
  end
  else
    let lifted = List.map lift_consts traced in
    let hoisted_each = List.map snd lifted in
    let hoisted = List.concat hoisted_each in
    let all_avals = List.map Core.get_aval hoisted @ avals in
    let branches' =
      if hoisted = [] then List.map fst lifted
      else build_branches all_avals lifted hoisted_each
    in
    Core.bind
      (Cond { branches = branches'; platforms })
      ((index :: hoisted) @ operands)

let rec cond (pred : value) (true_fun : value list -> value list)
    (false_fun : value list -> value list) (operands : value list) : value list
    =
  let avals = List.map Core.get_aval operands in
  let traced_t = Jaxpr.make_jaxpr_stateful avals true_fun in
  let traced_f = Jaxpr.make_jaxpr_stateful avals false_fun in
  let t, hoisted_t, cells_t = traced_t in
  let f, hoisted_f, cells_f = traced_f in
  let cells =
    cells_t
    @ List.filter
        (fun (c : Ref_state.cell) -> not (List.memq c cells_t))
        cells_f
  in
  if cells <> [] then begin
    let n = List.length operands in
    let stateful branch args =
      match Util.split_list args [ n ] with
      | [ ops; state ] ->
          Ref_state.pin cells state;
          let outs = branch ops in
          outs @ Ref_state.contents cells
      | _ -> failwith "cond: arity"
    in
    let outs =
      cond pred (stateful true_fun) (stateful false_fun)
        (operands @ List.map Ref_state.read cells)
    in
    let n_out = List.length outs - List.length cells in
    match Util.split_list outs [ n_out ] with
    | [ results; finals ] ->
        List.iter2 Ref_state.write cells finals;
        results
    | _ -> failwith "cond: arity"
  end
  else
    let index = Core.bind1 (Convert_element_type Dtype.I32) [ pred ] in
    ignore (t, hoisted_t, f, hoisted_f);
    let lifted = List.map lift_consts [ traced_t; traced_f ] in
    let hoisted_each = List.map snd lifted in
    let hoisted = List.concat hoisted_each in
    let all_avals = List.map Core.get_aval hoisted @ avals in
    let branches' =
      if hoisted = [] then List.map fst lifted
      else build_branches all_avals lifted hoisted_each
    in
    let t', f' =
      match branches' with [ a; b ] -> (a, b) | _ -> assert false
    in
    Core.bind
      (Cond { branches = [ f'; t' ]; platforms = None })
      ((index :: hoisted) @ operands)

let index_scalar i =
  Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int i |])

let switch (index : value) (branches : (value list -> value list) list)
    (operands : value list) : value list =
  match branches with
  | [] -> invalid_arg "Empty branch sequence"
  | [ only ] -> only operands
  | _ ->
      let last = List.length branches - 1 in
      let i = Core.convert_like ~weak:false index Dtype.I32 in
      let i = Core.bind1 Clamp [ index_scalar 0; i; index_scalar last ] in
      branch_call i branches operands

let platform_index ~(platforms : string array option array) : value =
  Core.bind1 (Platform_index platforms) []
