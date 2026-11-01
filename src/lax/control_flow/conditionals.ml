open Types

let branch_call ?platforms (index : value)
    (branches : (value list -> value list) list) (operands : value list) :
    value list =
  let avals = List.map Core.get_aval operands in
  let traced = List.map (Jaxpr.make_jaxpr_stateful avals) branches in
  let hoisted_each = List.map (fun (_, h, _) -> h) traced in
  let hoisted = List.concat hoisted_each in
  let all_avals = List.map Core.get_aval hoisted @ avals in
  let n_hoisted = List.length hoisted in
  let offsets =
    let rec go acc = function
      | [] -> []
      | h :: tl -> acc :: go (acc + List.length h) tl
    in
    go 0 hoisted_each
  in
  let branches' =
    if n_hoisted = 0 then List.map (fun (c, _, _) -> c) traced
    else
      List.map2
        (fun ((c : closed_jaxpr), own, _) offset ->
          let n_own = List.length own in
          Jaxpr.make_jaxpr all_avals (fun args ->
              let ops = List.filteri (fun i _ -> i >= n_hoisted) args in
              let mine =
                List.filteri
                  (fun i _ -> i >= offset && i < offset + n_own)
                  args
              in
              Jaxpr.eval_closed_jaxpr c (ops @ mine)))
        traced offsets
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
    let hoisted = hoisted_t @ hoisted_f in
    let n_t = List.length hoisted_t in
    let n_f = List.length hoisted_f in
    let all_avals = List.map Core.get_aval hoisted @ avals in
    let split args =
      match Util.split_list args [ n_t; n_f ] with
      | [ h_t; h_f; ops ] -> (ops, h_t, h_f)
      | _ -> failwith "cond: arity"
    in
    let t', f' =
      if hoisted = [] then (t, f)
      else
        ( Jaxpr.make_jaxpr all_avals (fun args ->
              let ops, h_t, _ = split args in
              Jaxpr.eval_closed_jaxpr t (ops @ h_t)),
          Jaxpr.make_jaxpr all_avals (fun args ->
              let ops, _, h_f = split args in
              Jaxpr.eval_closed_jaxpr f (ops @ h_f)) )
    in
    Core.bind
      (Cond { branches = [ f'; t' ]; platforms = None })
      ((index :: hoisted) @ operands)

let index_scalar i = Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int i |])

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
