module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module PC = Parity_cases
module TR = Ojax.Tree_util
module UF = Ojax.Numpy.Ufuncs
module JU = Ojax.Numpy.Jnp_util
module X = Ojax.Numpy.Extras
module NL = Ojax.Numpy.Lax_numpy
module I = Ojax.Numpy.Indexing
module J = Ojax.Jax_top
module SP = Ojax.Ops.Special

let pstr name p = U.to_string (U.member name p)
let pbool name p = U.to_bool (U.member name p)

let text s =
  let n = String.length s in
  T.Concrete
    (Nd.of_floats D.Uint8 [| n |]
       (Array.init n (fun i -> float_of_int (Char.code s.[i]))))

let int32s values =
  let a = Array.of_list values in
  T.Concrete
    (Nd.of_floats D.I32 [| Array.length a |] (Array.map float_of_int a))

let flag b = T.Concrete (Nd.of_floats D.Bool [||] [| (if b then 1.0 else 0.0) |])
let times v f = UF.multiply v (JU.py_float f)
let plus v f = UF.add v (JU.py_float f)

let dtype_named name = X.dtype name

let cls = function
  | "flexible" -> NL.Flexible
  | "character" -> NL.Character
  | "object" -> NL.Object_
  | "generic" -> NL.Generic
  | "number" -> NL.Number
  | other -> failwith ("apicov: class " ^ other)

let tree_of a =
  TR.Dict
    [
      ("q", TR.List [ TR.Leaf (times a 2.0); TR.Null ]);
      ("p", TR.Tuple [ TR.Leaf a; TR.Leaf (times a 3.0) ]);
    ]

let is_tuple = function TR.Tuple _ -> true | _ -> false

let leaf_sum = function
  | TR.Tuple [ TR.Leaf l; TR.Leaf r ] -> UF.add l r
  | TR.Leaf v -> v
  | _ -> failwith "apicov: leaf_sum"

type node = { x : T.value TR.t; y : T.value TR.t; label : string }

let node_class =
  TR.register_pytree_node "Node"
    ~flatten:(fun n -> ([ n.x; n.y ], n.label))
    ~unflatten:(fun label children ->
      match children with
      | [ x; y ] -> { x; y; label }
      | _ -> failwith "apicov: node")

let keyed_class =
  TR.register_pytree_with_keys "KeyedNode"
    ~flatten_with_keys:(fun n ->
      ([ (TR.Get_attr_key "x", n.x); (TR.Dict_key "why", n.y) ], "None"))
    ~unflatten:(fun _ children ->
      match children with
      | [ x; y ] -> { x; y; label = "" }
      | _ -> failwith "apicov: keyed")

let class_node =
  TR.register_pytree_node_class "ClassNode"
    ~tree_flatten:(fun v -> ([ v ], "cn"))
    ~tree_unflatten:(fun _ children -> List.hd children)

let class_keyed =
  TR.register_pytree_with_keys_class "ClassKeyedNode"
    ~tree_flatten_with_keys:(fun v -> ([ (TR.Get_attr_key "inner", v) ], "None"))
    ~tree_unflatten:(fun _ children -> List.hd children)

let data_class =
  TR.register_dataclass "Data" ~data_fields:[ "x"; "y" ] ~meta_fields:[ "op" ]
    ~get:(fun n field -> if field = "x" then n.x else n.y)
    ~meta:(fun n -> n.label)
    ~make:(fun label children ->
      match children with
      | [ x; y ] -> { x; y; label }
      | _ -> failwith "apicov: dataclass")

let static_class : (string, int) TR.node_class =
  TR.register_static "Stat" ~to_aux:(fun s -> s) ~of_aux:(fun s -> s)

let leaf1 = function TR.Leaf v -> v | _ -> failwith "apicov: not a leaf"

let treedef_case = function
  | "dict" ->
      TR.Dict
        [
          ("b", TR.Leaf 1);
          ("a", TR.List [ TR.Leaf 2; TR.Tuple [ TR.Leaf 3; TR.Null ] ]);
        ]
  | "one" -> TR.Tuple [ TR.Leaf 1 ]
  | "leaf" -> TR.Leaf 1
  | "none" -> TR.Null
  | "empty" -> TR.Tuple [ TR.Tuple []; TR.List []; TR.Dict [] ]
  | other -> failwith ("apicov: tree " ^ other)

let axis_of p =
  match U.member "axis" p with
  | `Null -> None
  | _ -> Some (Array.of_list (PC.ints "axis" p |> Array.to_list))

let logsumexp_results ?b p a =
  let axis = axis_of p in
  let keepdims = PC.opt_bool "keepdims" p in
  let where =
    match U.member "where" p with
    | `Null -> None
    | v -> Some (UF.greater a (JU.py_float (U.to_number v)))
  in
  if PC.opt_bool "return_sign" p = Some true then
    let out, sign = SP.logsumexp_sign ?axis ?b ?keepdims ?where a in
    [ out; sign ]
  else [ SP.logsumexp ?axis ?b ?keepdims ?where a ]

let partition_items p =
  List.map
    (function
      | `String s -> Ojax.Spmd.Named s
      | `Null -> Ojax.Spmd.Unconstrained
      | `List names -> Ojax.Spmd.Axes (List.map U.to_string names)
      | _ -> failwith "apicov: partition item")
    (U.to_list (U.member "items" p))

let strings name p =
  match U.member name p with
  | `Null -> None
  | v -> Some (List.map U.to_string (U.to_list v))

let run op p get =
  let a () = get "a" in
  let b () = get "b" in
  match op with
  | "ap_logsumexp" -> logsumexp_results p (a ())
  | "ap_logsumexp_b" -> logsumexp_results ~b:(b ()) p (a ())
  | "ap_special_logsumexp" -> (
      let axis = axis_of p in
      match PC.opt_bool "return_sign" p with
      | Some true ->
          let out, sign =
            Ojax.Scipy.Special.logsumexp_sign ?axis (a ())
          in
          [ out; sign ]
      | _ -> [ Ojax.Nn.Functions.logsumexp ?axis (a ()) ])
  | "ap_shape" -> [ int32s (Array.to_list (X.shape (a ()))) ]
  | "ap_broadcast_shapes" ->
      let shapes =
        List.map
          (fun s -> Array.of_list (List.map U.to_int (U.to_list s)))
          (U.to_list (U.member "shapes" p))
      in
      [ int32s (Array.to_list (X.broadcast_shapes shapes)) ]
  | "ap_iterable" -> [ flag (X.iterable (a ())) ]
  | "ap_complex_" -> [ X.complex_ (a ()) ]
  | "ap_np_dtype" ->
      [ text (Ojax.Ref_state.dtype_name (dtype_named (pstr "name" p))) ]
  | "ap_issubdtype" ->
      [ flag (NL.issubdtype (cls (pstr "left" p)) (cls (pstr "right" p))) ]
  | "ap_issubdtype_named" ->
      [
        flag
          (NL.issubdtype
             (NL.Cdtype (dtype_named (pstr "left" p)))
             (cls (pstr "right" p)));
      ]
  | "ap_lax_dtype" ->
      [ text (Ojax.Ref_state.dtype_name (Ojax.Lax_extras.dtype (a ()))) ]
  | "ap_ds" ->
      let slice =
        if PC.opt_bool "full" p = Some true then J.ds None
        else
          let start =
            T.Concrete
              (Nd.of_floats D.I32 [||] [| float_of_int (PC.int "start" p) |])
          in
          J.ds ?size:(PC.opt_int "size" p) ?stride:(PC.opt_int "stride" p)
            (Some start)
      in
      [ I.dslice_index (a ()) [ slice ] ]
  | "ap_dce_sink" ->
      let x = a () in
      Ojax.Lax_extras.dce_sink ~prevent_mlir_dce:(pbool "prevent" p) x;
      [ x ]
  | "ap_io_callback" ->
      let x = a () in
      let aval = Ojax.Core.get_aval x in
      J.io_callback
        ?ordered:(PC.opt_bool "ordered" p)
        (fun vs -> List.map (fun v -> times v 2.0) vs)
        ~result_avals:[ { aval with T.weak_type = false } ]
        [ x ]
  | "ap_tree_flatten_is_leaf" ->
      let leaves, td = TR.tree_flatten_is_leaf ~is_leaf:is_tuple (tree_of (a ())) in
      List.map leaf_sum leaves
      @ [
          text (TR.treedef_to_string td);
          int32s [ TR.count_leaves td; TR.num_nodes td ];
        ]
  | "ap_tree_paths" ->
      let tree = tree_of (a ()) in
      let paths =
        match pstr "mode" p with
        | "plain" ->
            List.map
              (fun (k, _) -> k)
              (TR.tree_leaves_with_path tree)
        | "is_leaf" ->
            List.map fst
              (TR.tree_leaves_with_path_is_leaf ~is_leaf:is_tuple tree)
        | _ ->
            List.map fst
              (fst
                 (TR.tree_flatten_with_path_is_leaf
                    ~is_leaf_takes_path:(fun path _ -> List.length path >= 2)
                    tree))
      in
      List.map (fun k -> text (TR.keystr k)) paths
  | "ap_tree_map_path" ->
      TR.tree_leaves
        (TR.tree_map_with_path_is_leaf ~is_leaf:is_tuple
           (fun path x ->
             TR.Leaf (times (leaf_sum x) (float_of_int (List.length path))))
           (tree_of (a ())))
  | "ap_tree_reduce" ->
      [
        TR.tree_reduce_is_leaf ~is_leaf:is_tuple
          (fun acc x -> UF.add acc (leaf_sum x))
          (JU.py_float 1.0)
          (tree_of (a ()));
      ]
  | "ap_tree_reduce_noinit" ->
      let x = a () in
      [
        TR.tree_reduce_no_init UF.multiply
          (TR.Tuple [ TR.Leaf x; TR.Leaf (plus x 1.0); TR.Leaf (plus x 2.0) ]);
      ]
  | "ap_tree_assoc" ->
      let x = a () in
      [
        TR.tree_reduce_associative UF.subtract
          (TR.Tuple
             [
               TR.Leaf x;
               TR.Leaf (times x 2.0);
               TR.Leaf (times x 3.0);
               TR.Leaf (times x 5.0);
               TR.Leaf (times x 7.0);
             ]);
      ]
  | "ap_tree_all" ->
      [
        flag
          (TR.tree_all_is_leaf
             ~is_leaf:(function TR.List _ -> true | _ -> false)
             (function TR.List [] -> false | _ -> true)
             (TR.Tuple
                [ TR.Leaf true; TR.List [ TR.Leaf true; TR.Leaf false ] ]));
      ]
  | "ap_tree_broadcast" ->
      let x = a () in
      TR.tree_leaves
        (TR.tree_broadcast
           (TR.Tuple [ TR.Leaf x; TR.Leaf (times x 2.0) ])
           (TR.Tuple
              [
                TR.List [ TR.Leaf 1; TR.Leaf 2 ];
                TR.Dict
                  [
                    ("k", TR.Leaf 3);
                    ("j", TR.Tuple [ TR.Leaf 4; TR.Leaf 5 ]);
                  ];
              ]))
  | "ap_all_leaves" ->
      let of_bool b = if b then 1 else 0 in
      let items = [ TR.Tuple [ TR.Leaf 1 ]; TR.Leaf 2 ] in
      [
        int32s
          [
            of_bool (TR.all_leaves [ TR.Leaf 1; TR.Leaf 2 ]);
            of_bool (TR.all_leaves items);
            of_bool (TR.all_leaves_is_leaf ~is_leaf:is_tuple items);
          ];
      ]
  | "ap_treedef_str" ->
      let td = TR.tree_structure (treedef_case (pstr "tree" p)) in
      [
        text (TR.treedef_to_string td);
        int32s [ TR.count_leaves td; TR.num_nodes td ];
      ]
  | "ap_node" ->
      let x = a () in
      let n = { x = TR.Leaf x; y = TR.Leaf (times x 2.0); label = "tag" } in
      let tree = TR.node_to_tree node_class n in
      let leaves, td = TR.tree_flatten tree in
      let back =
        TR.node_of_tree node_class
          (TR.tree_unflatten td (List.map (fun v -> times v 3.0) leaves))
      in
      let mapped =
        TR.node_of_tree node_class (TR.tree_map (fun v -> plus v 1.0) tree)
      in
      [
        leaf1 back.x;
        leaf1 back.y;
        leaf1 mapped.x;
        leaf1 mapped.y;
        text back.label;
        text (TR.treedef_to_string td);
      ]
      @ List.map
          (fun (k, _) -> text (TR.keystr k))
          (TR.tree_leaves_with_path tree)
  | "ap_keyed_node" ->
      let x = a () in
      let n = { x = TR.Leaf x; y = TR.Leaf (times x 2.0); label = "" } in
      let tree = TR.node_to_tree keyed_class n in
      let found = TR.tree_leaves_with_path tree in
      List.map (fun (k, _) -> text (TR.keystr k)) found
      @ List.map snd found
      @ [
          leaf1
            (TR.node_of_tree keyed_class
               (TR.tree_map (fun v -> times v 2.0) tree))
              .y;
        ]
  | "ap_class_node" ->
      let x = a () in
      let c = TR.node_to_tree class_node (TR.Leaf x) in
      let k = TR.node_to_tree class_keyed (TR.Leaf (times x 2.0)) in
      [
        leaf1
          (TR.node_of_tree class_node (TR.tree_map (fun v -> times v 2.0) c));
        text (TR.treedef_to_string (TR.tree_structure c));
        List.hd (TR.tree_leaves k);
        text (TR.keystr (fst (List.hd (TR.tree_leaves_with_path k))));
      ]
  | "ap_dataclass" ->
      let x = a () in
      let d = { x = TR.Leaf x; y = TR.Leaf (times x 2.0); label = "add" } in
      let tree = TR.node_to_tree data_class d in
      let back =
        TR.node_of_tree data_class (TR.tree_map (fun v -> plus v 1.0) tree)
      in
      [
        leaf1 back.x;
        leaf1 back.y;
        text back.label;
        text (TR.keystr (fst (List.nth (TR.tree_leaves_with_path tree) 1)));
      ]
  | "ap_static" ->
      let tree = TR.node_to_tree static_class "st" in
      let td = TR.tree_structure tree in
      [
        int32s [ TR.count_leaves td ];
        text (TR.node_of_tree static_class (TR.tree_unflatten td []));
      ]
  | "ap_partial" ->
      let x = a () in
      let f = function
        | [ TR.Leaf u; TR.Leaf v ] -> UF.multiply u v
        | _ -> failwith "apicov: partial"
      in
      let pf = TR.partial f [ TR.Leaf x ] in
      let tree = TR.partial_to_tree pf in
      let mapped =
        TR.partial_of_tree pf (TR.tree_map (fun v -> plus v 1.0) tree)
      in
      let first = TR.partial_call pf [ TR.Leaf (times x 2.0) ] in
      let second = TR.partial_call mapped [ TR.Leaf (times x 2.0) ] in
      [ first; List.hd (TR.tree_leaves tree); second ]
  | "ap_primal_tangent" ->
      let d =
        J.primal_tangent_dtype ?name:(PC.opt_string "name" p)
          (dtype_named (pstr "primal" p))
          (dtype_named (pstr "tangent" p))
      in
      [ text d.J.name ]
  | "ap_partition" ->
      [
        text
          (J.p_to_string
             (J.partition ?unreduced:(strings "unreduced" p)
                ?reduced:(strings "reduced" p) (partition_items p)));
      ]
  | "ap_config_flags" ->
      let of_bool b = if b then 1 else 0 in
      let cfg name =
        match J.config_read name with
        | Ojax.Config.Config_bool b -> of_bool b
        | _ -> failwith "apicov: config"
      in
      [
        int32s
          [
            of_bool (Ojax.Config.value J.auto_pcast);
            of_bool (Ojax.Config.value J.remove_size_one_mesh_axis_from_type);
            cfg "jax_auto_pcast";
            cfg "jax_threefry_partitionable";
            1;
          ];
      ]
  | "ap_memory" ->
      List.map
        (fun s -> text (J.Memory.to_string s))
        [ J.Memory.Device; J.Memory.Host; J.Memory.Any ]
  | other -> failwith ("apicov: unknown op " ^ other)
