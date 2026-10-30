type 'a t =
  | Leaf of 'a
  | List of 'a t list
  | Tuple of 'a t list
  | Dict of (string * 'a t) list
  | Custom of string * string * 'a t list
  | Null

type treedef =
  | Leaf_def
  | List_def of treedef list
  | Tuple_def of treedef list
  | Dict_def of (string * treedef) list
  | Custom_def of string * string * treedef list
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
  | Custom (name, aux, children) ->
      let leaves, defs = flatten_children children in
      (leaves, Custom_def (name, aux, defs))

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
    | Custom_def (name, aux, defs) -> Custom (name, aux, List.map build defs)
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
  | Custom (name, aux, children) ->
      Custom (name, aux, List.map (tree_map f) children)

type key_entry =
  | Dict_key of string
  | Sequence_key of int
  | Flattened_index_key of int

type key_path = key_entry list

let rec tree_flatten_with_path_from prefix tree =
  match tree with
  | Leaf x -> [ (List.rev prefix, x) ]
  | Null -> []
  | List children | Tuple children ->
      List.concat
        (List.mapi
           (fun i c -> tree_flatten_with_path_from (Sequence_key i :: prefix) c)
           children)
  | Custom (_, _, children) ->
      List.concat
        (List.mapi
           (fun i c ->
             tree_flatten_with_path_from (Flattened_index_key i :: prefix) c)
           children)
  | Dict kvs ->
      List.concat_map
        (fun (k, c) -> tree_flatten_with_path_from (Dict_key k :: prefix) c)
        (sort_dict kvs)

let tree_leaves_with_path tree = tree_flatten_with_path_from [] tree

let tree_flatten_with_path tree =
  (tree_leaves_with_path tree, tree_structure tree)

let tree_map_with_path f tree =
  let paths = tree_leaves_with_path tree in
  tree_unflatten (tree_structure tree) (List.map (fun (p, x) -> f p x) paths)

let keystr path =
  String.concat ""
    (List.map
       (function
         | Dict_key k -> "['" ^ k ^ "']"
         | Sequence_key i -> "[" ^ string_of_int i ^ "]"
         | Flattened_index_key i -> "[<flat index " ^ string_of_int i ^ ">]")
       path)

let tree_reduce f init tree = List.fold_left f init (tree_leaves tree)

let tree_reduce_associative f ?identity tree =
  match (tree_leaves tree, identity) with
  | [], Some id -> id
  | [], None ->
      invalid_arg "Must specify identity for parallel reduction of empty tree"
  | first :: rest, _ -> List.fold_left f first rest

let tree_all tree = List.for_all Fun.id (tree_leaves tree)

let tree_map2 f a b =
  let la, da = tree_flatten a in
  let lb, db = tree_flatten b in
  if da <> db then invalid_arg "tree_map: mismatched tree structures";
  tree_unflatten da (List.map2 f la lb)

let treedef_children = function
  | Leaf_def | Null_def -> []
  | List_def ds | Tuple_def ds | Custom_def (_, _, ds) -> ds
  | Dict_def kvs -> List.map snd kvs

let treedef_is_leaf def = def = Leaf_def
let treedef_tuple defs = Tuple_def defs

let flatten_one_level = function
  | Leaf _ -> invalid_arg "flatten_one_level: not a tree node"
  | Null -> []
  | List c | Tuple c | Custom (_, _, c) -> c
  | Dict kvs -> List.map snd (sort_dict kvs)

let is_tree_node = function Leaf _ -> false | _ -> true

let all_leaves trees =
  List.for_all (function Leaf _ -> true | _ -> false) trees

let tree_broadcast prefix full =
  let rec go p f =
    match (p, f) with
    | Leaf x, _ -> tree_map (fun _ -> x) f
    | Null, Null -> Null
    | List a, List b -> List (List.map2 go a b)
    | Tuple a, Tuple b -> Tuple (List.map2 go a b)
    | Custom (n, x, a), Custom (_, _, b) -> Custom (n, x, List.map2 go a b)
    | Dict a, Dict b ->
        Dict
          (List.map2
             (fun (k, x) (_, y) -> (k, go x y))
             (sort_dict a) (sort_dict b))
    | _ -> invalid_arg "tree_broadcast: prefix is not a tree prefix"
  in
  go prefix full

let rec count_leaves = function
  | Leaf_def -> 1
  | Null_def -> 0
  | List_def ds | Tuple_def ds | Custom_def (_, _, ds) ->
      List.fold_left (fun acc d -> acc + count_leaves d) 0 ds
  | Dict_def kvs ->
      List.fold_left (fun acc (_, d) -> acc + count_leaves d) 0 kvs

let tree_transpose outer inner tree =
  let inner_size = count_leaves inner and outer_size = count_leaves outer in
  let leaves = Array.of_list (tree_leaves tree) in
  if Array.length leaves <> inner_size * outer_size then
    invalid_arg "tree_transpose: mismatched tree structures";
  let column i =
    tree_unflatten outer
      (List.init outer_size (fun o -> leaves.((o * inner_size) + i)))
  in
  let remaining = ref (List.init inner_size column) in
  let take () =
    match !remaining with
    | x :: rest ->
        remaining := rest;
        x
    | [] -> invalid_arg "tree_transpose: too few leaves"
  in
  let rec build = function
    | Leaf_def -> take ()
    | Null_def -> Null
    | List_def ds -> List (List.map build ds)
    | Tuple_def ds -> Tuple (List.map build ds)
    | Dict_def kvs -> Dict (List.map (fun (k, d) -> (k, build d)) kvs)
    | Custom_def (n, x, ds) -> Custom (n, x, List.map build ds)
  in
  build inner
