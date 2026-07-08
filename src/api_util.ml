let ensure_inbounds (num_args : int) (argnums : int list) : int list =
  List.map
    (fun i ->
      if i >= -num_args && i < num_args then (i + num_args) mod num_args
      else
        invalid_arg
          (Printf.sprintf
             "Positional argument indices, e.g. for `static_argnums`, must \
              have value greater than or equal to -len(args) and less than \
              len(args), but got value %d for len(args) == %d."
             i num_args))
    argnums

let flatten_fun (f : 'a Tree_util.t list -> 'a Tree_util.t)
    (in_tree : Tree_util.treedef) :
    ('a list, 'a list) Linear_util.t * (unit -> Tree_util.treedef) =
  Linear_util.transformation_with_aux2
    (fun inner store (args_flat : 'a list) ->
      let arg_trees =
        match Tree_util.tree_unflatten in_tree args_flat with
        | Tree_util.Tuple xs | Tree_util.List xs -> xs
        | other -> [ other ]
      in
      let out = inner arg_trees in
      let out_flat, out_tree = Tree_util.tree_flatten out in
      Linear_util.store store out_tree;
      out_flat)
    (Linear_util.wrap_init f)

let argnums_partial (f : 'a Tree_util.t list -> 'a Tree_util.t)
    (dyn_argnums : int list) (args : 'a Tree_util.t list) :
    ('a Tree_util.t list -> 'a Tree_util.t) * 'a Tree_util.t list =
  let n = List.length args in
  let dyn = ensure_inbounds n dyn_argnums in
  let arg_arr = Array.of_list args in
  let dyn_args = List.map (fun i -> arg_arr.(i)) dyn in
  let f_wrapped dyn_args_ =
    let a = Array.copy arg_arr in
    List.iter2 (fun i x -> a.(i) <- x) dyn dyn_args_;
    f (Array.to_list a)
  in
  (f_wrapped, dyn_args)
