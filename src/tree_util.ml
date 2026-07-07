type 'a t =
  | Leaf of 'a
  | List of 'a t list
  | Tuple of 'a t list
  | Dict of (string * 'a t) list
  | Null

type treedef =
  | Leaf_def
  | List_def of treedef list
  | Tuple_def of treedef list
  | Dict_def of (string * treedef) list
  | Null_def

let sort_dict kvs = List.sort (fun (a, _) (b, _) -> String.compare a b) kvs

let rec tree_flatten (tree : 'a t) : 'a list * treedef =
  match tree with
  | Leaf x -> ([ x ], Leaf_def)
  | Null -> ([], Null_def)
  | List children ->
      let leaves, defs = flatten_children children in
      (leaves, List_def defs)
  | Tuple children ->
      let leaves, defs = flatten_children children in
      (leaves, Tuple_def defs)
  | Dict kvs ->
      let sorted = sort_dict kvs in
      let leaves, defs = flatten_children (List.map snd sorted) in
      (leaves, Dict_def (List.combine (List.map fst sorted) defs))

and flatten_children children =
  let leaves, defs =
    List.fold_left
      (fun (leaves, defs) child ->
        let ls, d = tree_flatten child in
        (List.rev_append ls leaves, d :: defs))
      ([], []) children
  in
  (List.rev leaves, List.rev defs)

let tree_leaves (tree : 'a t) : 'a list = fst (tree_flatten tree)
let tree_structure (tree : 'a t) : treedef = snd (tree_flatten tree)

let tree_unflatten (def : treedef) (leaves : 'b list) : 'b t =
  let remaining = ref leaves in
  let take () =
    match !remaining with
    | x :: rest ->
        remaining := rest;
        x
    | [] -> invalid_arg "tree_unflatten: too few leaves for treedef"
  in
  let rec build = function
    | Leaf_def -> Leaf (take ())
    | Null_def -> Null
    | List_def defs -> List (List.map build defs)
    | Tuple_def defs -> Tuple (List.map build defs)
    | Dict_def kvs -> Dict (List.map (fun (k, d) -> (k, build d)) kvs)
  in
  let result = build def in
  match !remaining with
  | [] -> result
  | _ -> invalid_arg "tree_unflatten: too many leaves for treedef"

let rec tree_map (f : 'a -> 'b) (tree : 'a t) : 'b t =
  match tree with
  | Leaf x -> Leaf (f x)
  | Null -> Null
  | List children -> List (List.map (tree_map f) children)
  | Tuple children -> Tuple (List.map (tree_map f) children)
  | Dict kvs -> Dict (List.map (fun (k, v) -> (k, tree_map f v)) kvs)
