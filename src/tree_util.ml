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

let rec tree_map_is_leaf ~is_leaf (f : 'a t -> 'b t) (tree : 'a t) : 'b t =
  if is_leaf tree then f tree
  else
    let go = tree_map_is_leaf ~is_leaf f in
    match tree with
    | Leaf _ -> f tree
    | Null -> Null
    | List children -> List (List.map go children)
    | Tuple children -> Tuple (List.map go children)
    | Dict kvs -> Dict (List.map (fun (k, v) -> (k, go v)) kvs)
    | Custom (name, aux, children) -> Custom (name, aux, List.map go children)

let rec tree_leaves_is_leaf ~is_leaf (tree : 'a t) : 'a t list =
  if is_leaf tree then [ tree ]
  else
    let go = tree_leaves_is_leaf ~is_leaf in
    match tree with
    | Leaf _ -> [ tree ]
    | Null -> []
    | List children | Tuple children | Custom (_, _, children) ->
        List.concat_map go children
    | Dict kvs -> List.concat_map (fun (_, v) -> go v) (sort_dict kvs)

let rec tree_structure_is_leaf ~is_leaf (tree : 'a t) : treedef =
  if is_leaf tree then Leaf_def
  else
    let go = tree_structure_is_leaf ~is_leaf in
    match tree with
    | Leaf _ -> Leaf_def
    | Null -> Null_def
    | List children -> List_def (List.map go children)
    | Tuple children -> Tuple_def (List.map go children)
    | Dict kvs -> Dict_def (List.map (fun (k, v) -> (k, go v)) (sort_dict kvs))
    | Custom (name, aux, children) ->
        Custom_def (name, aux, List.map go children)

type key_entry =
  | Dict_key of string
  | Sequence_key of int
  | Flattened_index_key of int
  | Get_attr_key of string

type key_path = key_entry list

let attribute_keys : (string, string list) Hashtbl.t = Hashtbl.create 8

let register_attribute_keys name fields =
  Hashtbl.replace attribute_keys name fields

let instance_keys : (string * string, key_entry list) Hashtbl.t =
  Hashtbl.create 8

let custom_keys_for name aux i =
  match Hashtbl.find_opt instance_keys (name, aux) with
  | Some keys when i < List.length keys -> Some (List.nth keys i)
  | _ -> None

let custom_child_key name i =
  match Hashtbl.find_opt attribute_keys name with
  | Some fields when i < List.length fields -> Get_attr_key (List.nth fields i)
  | _ -> Flattened_index_key i

let rec tree_flatten_with_path_from prefix tree =
  match tree with
  | Leaf x -> [ (List.rev prefix, x) ]
  | Null -> []
  | List children | Tuple children ->
      List.concat
        (List.mapi
           (fun i c -> tree_flatten_with_path_from (Sequence_key i :: prefix) c)
           children)
  | Custom (name, aux, children) ->
      List.concat
        (List.mapi
           (fun i c ->
             let key =
               match custom_keys_for name aux i with
               | Some key -> key
               | None -> custom_child_key name i
             in
             tree_flatten_with_path_from (key :: prefix) c)
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

let keystr ?(simple = false) ?(separator = "") path =
  String.concat separator
    (List.map
       (function
         | Dict_key k -> if simple then k else "['" ^ k ^ "']"
         | Sequence_key i ->
             if simple then string_of_int i else "[" ^ string_of_int i ^ "]"
         | Flattened_index_key i ->
             if simple then string_of_int i
             else "[<flat index " ^ string_of_int i ^ ">]"
         | Get_attr_key name -> if simple then name else "." ^ name)
       path)

let tree_reduce f init tree = List.fold_left f init (tree_leaves tree)

let rec parallel_reduce f identity = function
  | [] -> (
      match identity with
      | Some i -> i
      | None ->
          invalid_arg
            "Must specify identity for parallel reduction of empty sequence.")
  | [ x ] -> x
  | xs ->
      let n = List.length xs / 2 in
      let left = List.filteri (fun i _ -> i < n) xs in
      let right = List.filteri (fun i _ -> i >= n) xs in
      f (parallel_reduce f identity left) (parallel_reduce f identity right)

let tree_reduce_associative f ?identity tree =
  parallel_reduce f identity (tree_leaves tree)

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

let flatten_one_level_with_keys = function
  | Leaf _ -> invalid_arg "can't tree-flatten type: leaf"
  | Null -> []
  | List c | Tuple c -> List.mapi (fun i x -> (Sequence_key i, x)) c
  | Custom (name, _, c) -> List.mapi (fun i x -> (custom_child_key name i, x)) c
  | Dict kvs -> List.map (fun (k, x) -> (Dict_key k, x)) (sort_dict kvs)

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

let tree_broadcast_is_leaf ~is_leaf prefix full =
  let stamp p f = tree_map_is_leaf ~is_leaf:(fun _ -> false) (fun _ -> p) f in
  let rec go p f =
    if is_leaf p then stamp p f
    else
      match (p, f) with
      | Leaf _, _ -> stamp p f
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

let tree_flatten_is_leaf ~is_leaf tree =
  (tree_leaves_is_leaf ~is_leaf tree, tree_structure_is_leaf ~is_leaf tree)

let rec with_path_is_leaf ~is_leaf_takes_path prefix tree =
  let path = List.rev prefix in
  if is_leaf_takes_path path tree then [ (path, tree) ]
  else
    let go key = with_path_is_leaf ~is_leaf_takes_path (key :: prefix) in
    match tree with
    | Leaf _ -> [ (path, tree) ]
    | Null -> []
    | List children | Tuple children ->
        List.concat (List.mapi (fun i c -> go (Sequence_key i) c) children)
    | Custom (name, aux, children) ->
        List.concat
          (List.mapi
             (fun i c ->
               let key =
                 match custom_keys_for name aux i with
                 | Some key -> key
                 | None -> custom_child_key name i
               in
               go key c)
             children)
    | Dict kvs ->
        List.concat_map (fun (k, c) -> go (Dict_key k) c) (sort_dict kvs)

let rec structure_is_leaf_path ~is_leaf_takes_path prefix tree =
  if is_leaf_takes_path (List.rev prefix) tree then Leaf_def
  else
    let go key = structure_is_leaf_path ~is_leaf_takes_path (key :: prefix) in
    match tree with
    | Leaf _ -> Leaf_def
    | Null -> Null_def
    | List children ->
        List_def (List.mapi (fun i c -> go (Sequence_key i) c) children)
    | Tuple children ->
        Tuple_def (List.mapi (fun i c -> go (Sequence_key i) c) children)
    | Dict kvs ->
        Dict_def
          (List.map (fun (k, c) -> (k, go (Dict_key k) c)) (sort_dict kvs))
    | Custom (name, aux, children) ->
        Custom_def
          ( name,
            aux,
            List.mapi
              (fun i c ->
                let key =
                  match custom_keys_for name aux i with
                  | Some key -> key
                  | None -> custom_child_key name i
                in
                go key c)
              children )

let path_predicate ?is_leaf ?is_leaf_takes_path () =
  match (is_leaf, is_leaf_takes_path) with
  | Some _, Some _ -> invalid_arg "is_leaf and is_leaf_takes_path are exclusive"
  | Some f, None -> fun _ x -> f x
  | None, Some f -> f
  | None, None -> fun _ _ -> false

let tree_leaves_with_path_is_leaf ?is_leaf ?is_leaf_takes_path tree =
  with_path_is_leaf
    ~is_leaf_takes_path:(path_predicate ?is_leaf ?is_leaf_takes_path ())
    [] tree

let tree_flatten_with_path_is_leaf ?is_leaf ?is_leaf_takes_path tree =
  let pred = path_predicate ?is_leaf ?is_leaf_takes_path () in
  ( with_path_is_leaf ~is_leaf_takes_path:pred [] tree,
    structure_is_leaf_path ~is_leaf_takes_path:pred [] tree )

let tree_map_with_path_is_leaf ?is_leaf ?is_leaf_takes_path f tree =
  let leaves, def =
    tree_flatten_with_path_is_leaf ?is_leaf ?is_leaf_takes_path tree
  in
  let rebuilt = List.map (fun (p, x) -> f p x) leaves in
  let remaining = ref rebuilt in
  let take () =
    match !remaining with
    | x :: rest ->
        remaining := rest;
        x
    | [] -> invalid_arg "tree_map_with_path: too few leaves"
  in
  let rec build = function
    | Leaf_def -> take ()
    | Null_def -> Null
    | List_def defs -> List (List.map build defs)
    | Tuple_def defs -> Tuple (List.map build defs)
    | Dict_def kvs -> Dict (List.map (fun (k, d) -> (k, build d)) kvs)
    | Custom_def (name, aux, defs) -> Custom (name, aux, List.map build defs)
  in
  build def

let tree_reduce_is_leaf ~is_leaf f init tree =
  List.fold_left f init (tree_leaves_is_leaf ~is_leaf tree)

let tree_all_is_leaf ~is_leaf pred tree =
  List.for_all pred (tree_leaves_is_leaf ~is_leaf tree)

let tree_reduce_no_init f tree =
  match tree_leaves tree with
  | [] ->
      raise
        (Invalid_argument "reduce() of empty iterable with no initial value")
  | x :: rest -> List.fold_left f x rest

let tree_reduce_associative_is_leaf ~is_leaf f ?identity tree =
  parallel_reduce f identity (tree_leaves_is_leaf ~is_leaf tree)

let all_leaves_is_leaf ~is_leaf trees =
  List.for_all
    (fun t -> is_leaf t || match t with Leaf _ -> true | _ -> false)
    trees

type py_tree_def = treedef

let rec num_nodes = function
  | Leaf_def | Null_def -> 1
  | List_def defs | Tuple_def defs | Custom_def (_, _, defs) ->
      List.fold_left (fun n d -> n + num_nodes d) 1 defs
  | Dict_def kvs -> List.fold_left (fun n (_, d) -> n + num_nodes d) 1 kvs

let rec treedef_body = function
  | Leaf_def -> "*"
  | Null_def -> "None"
  | List_def defs -> "[" ^ String.concat ", " (List.map treedef_body defs) ^ "]"
  | Tuple_def [ one ] -> "(" ^ treedef_body one ^ ",)"
  | Tuple_def defs ->
      "(" ^ String.concat ", " (List.map treedef_body defs) ^ ")"
  | Dict_def kvs ->
      "{"
      ^ String.concat ", "
          (List.map (fun (k, d) -> "'" ^ k ^ "': " ^ treedef_body d) kvs)
      ^ "}"
  | Custom_def (name, aux, defs) ->
      "CustomNode(" ^ name ^ "[" ^ aux ^ "], ["
      ^ String.concat ", " (List.map treedef_body defs)
      ^ "])"

let treedef_to_string def = "PyTreeDef(" ^ treedef_body def ^ ")"

type registry = (string, unit) Hashtbl.t

let default_registry : registry = Hashtbl.create 16
let registry_mem (r : registry) name = Hashtbl.mem r name

type ('n, 'a) node_class = {
  node_name : string;
  node_flatten : 'n -> 'a t list * string;
  node_unflatten : string -> 'a t list -> 'n;
  node_keys : ('n -> key_entry list) option;
}

let claim name =
  if Hashtbl.mem default_registry name then
    invalid_arg
      ("Duplicate custom PyTreeDef type registration for <class '" ^ name
     ^ "'>.");
  Hashtbl.replace default_registry name ()

let register_pytree_node ?flatten_with_keys name ~flatten ~unflatten =
  claim name;
  {
    node_name = name;
    node_flatten = flatten;
    node_unflatten = unflatten;
    node_keys =
      Option.map (fun f node -> List.map fst (fst (f node))) flatten_with_keys;
  }

let register_pytree_node_class name ~tree_flatten ~tree_unflatten =
  register_pytree_node name ~flatten:tree_flatten ~unflatten:tree_unflatten

let register_pytree_with_keys ?flatten_func name ~flatten_with_keys ~unflatten =
  let flatten =
    match flatten_func with
    | Some f -> f
    | None ->
        fun node ->
          let pairs, aux = flatten_with_keys node in
          (List.map snd pairs, aux)
  in
  register_pytree_node ~flatten_with_keys name ~flatten ~unflatten

let register_pytree_with_keys_class name ~tree_flatten_with_keys ~tree_unflatten
    =
  register_pytree_with_keys name ~flatten_with_keys:tree_flatten_with_keys
    ~unflatten:tree_unflatten

let register_static name ~to_aux ~of_aux =
  register_pytree_node name
    ~flatten:(fun node -> ([], to_aux node))
    ~unflatten:(fun aux _ -> of_aux aux)

type field_kind = Data_field | Static_field

let static () = Static_field

let register_dataclass ?(meta_fields = []) ?(drop_fields = []) name ~data_fields
    ~get ~meta ~make =
  let overlap =
    List.filter
      (fun f -> List.mem f meta_fields || List.mem f drop_fields)
      data_fields
  in
  if overlap <> [] then
    invalid_arg
      ("data_fields and meta_fields must not overlap: "
     ^ String.concat ", " overlap);
  register_attribute_keys name data_fields;
  register_pytree_node name
    ~flatten:(fun node -> (List.map (get node) data_fields, meta node))
    ~unflatten:make

let node_to_tree cls node =
  let children, aux = cls.node_flatten node in
  (match cls.node_keys with
  | Some keys -> Hashtbl.replace instance_keys (cls.node_name, aux) (keys node)
  | None -> ());
  Custom (cls.node_name, aux, children)

let node_of_tree cls = function
  | Custom (name, aux, children) when name = cls.node_name ->
      cls.node_unflatten aux children
  | _ -> invalid_arg ("expected a " ^ cls.node_name ^ " node")

type ('a, 'b) partial = {
  func_name : string;
  func : 'a t list -> 'b;
  args : 'a t list;
}

let partial ?(name = "Partial") func args = { func_name = name; func; args }
let partial_call p extra = p.func (p.args @ extra)
let partial_to_tree p = Custom ("Partial", p.func_name, p.args)

let partial_of_tree template = function
  | Custom ("Partial", name, args) when name = template.func_name ->
      { template with args }
  | _ -> invalid_arg "expected a Partial node"
