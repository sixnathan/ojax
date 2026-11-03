module T = Tree_util
module F = Flatbuf

type aval = { shape : string list; dtype : int; memory_space : int }
type disabled_check = Platform | Custom_call of string

type t = {
  fun_name : string;
  in_tree : T.treedef;
  out_tree : T.treedef;
  in_avals : aval list;
  out_avals : aval list;
  nr_devices : int;
  in_shardings : int;
  out_shardings : int;
  platforms : string list;
  ordered_effects : string list;
  unordered_effects : string list;
  disabled_safety_checks : disabled_check list;
  mlir_module_serialized : string;
  calling_convention_version : int;
  module_kept_var_idx : int list;
  uses_global_constants : bool;
  vjp : t option;
}

let dtype_kinds =
  Dtype.
    [
      (Bool, 0);
      (I8, 1);
      (I16, 2);
      (I32, 3);
      (I64, 4);
      (Uint8, 5);
      (Uint16, 6);
      (Uint32, 7);
      (Uint64, 8);
      (F16, 9);
      (F32, 10);
      (F64, 11);
      (Complex64, 12);
      (Complex128, 13);
      (Bf16, 14);
      (I4, 15);
      (Uint4, 16);
      (F8_e4m3b11fnuz, 17);
      (F8_e4m3fn, 18);
      (F8_e4m3fnuz, 19);
      (F8_e5m2, 20);
      (F8_e5m2fnuz, 21);
      (Float0, 22);
      (F8_e4m3, 23);
      (F8_e3m4, 24);
      (F8_e8m0fnu, 25);
      (F4_e2m1fn, 26);
    ]

let dtype_of_kind k =
  match List.find_opt (fun (_, c) -> c = k) dtype_kinds with
  | Some (d, _) -> d
  | None -> Py_exn.not_implemented_error (Printf.sprintf "dtype kind %d" k)

let kind_of_dtype d = List.assoc d dtype_kinds
let aval_dtype a = dtype_of_kind a.dtype

let aval_of_shape dtype shape =
  {
    shape = List.map string_of_int (Array.to_list shape);
    dtype = kind_of_dtype dtype;
    memory_space = 1;
  }

let serialize_array b f elements =
  let offsets = List.map (f b) elements in
  F.offsets_vector b offsets

let rec serialize_tree b def =
  let children = T.treedef_children def in
  let children_vector =
    if children = [] then 0 else serialize_array b serialize_tree children
  in
  let kind, names, custom =
    match def with
    | T.Leaf_def -> (0, 0, None)
    | T.Null_def -> (1, 0, None)
    | T.Tuple_def _ -> (2, 0, None)
    | T.List_def _ -> (3, 0, None)
    | T.Dict_def kvs ->
        (4, serialize_array b F.create_string (List.map fst kvs), None)
    | T.Custom_def (node_type, aux, _) -> (
        match Export.custom_serialization node_type with
        | None ->
            Py_exn.value_error
              (Printf.sprintf
                 "Cannot serialize PyTreeDef containing an unregistered type \
                  `<class '%s'>`. Use \
                  `export.register_pytree_node_serialization` or \
                  `export.register_namedtuple_serialization`."
                 node_type)
        | Some (serialized_name, serialize_auxdata) ->
            let name = F.create_string b serialized_name in
            let auxdata = F.create_byte_vector b (serialize_auxdata aux) in
            (5, 0, Some (name, auxdata)))
  in
  F.start_object b 5;
  F.add_int_slot b 1 0 kind;
  F.add_offset_slot b 1 children_vector;
  F.add_offset_slot b 2 names;
  (match custom with
  | Some (name, auxdata) ->
      F.add_offset_slot b 3 name;
      F.add_offset_slot b 4 auxdata
  | None -> ());
  F.end_object b

let serialize_aval b a =
  let dims = List.map (F.create_string b) a.shape in
  let shape = F.offsets_vector b dims in
  F.start_object b 4;
  F.add_offset_slot b 1 shape;
  F.add_int_slot b 1 2 a.dtype;
  F.add_int_slot b 1 3 a.memory_space;
  F.end_object b

let serialize_none_sharding b () =
  F.start_object b 3;
  F.end_object b

let serialize_effect b name =
  let off = F.create_string b name in
  F.start_object b 1;
  F.add_offset_slot b 0 off;
  F.end_object b

let serialize_disabled b = function
  | Custom_call target ->
      let off = F.create_string b target in
      F.start_object b 2;
      F.add_int_slot b 1 0 1;
      F.add_offset_slot b 1 off;
      F.end_object b
  | Platform ->
      F.start_object b 2;
      F.end_object b

let uniques avals =
  List.fold_left
    (fun acc a -> if List.mem a acc then acc else acc @ [ a ])
    [] avals

let index_of x l =
  let rec go i = function
    | [] -> raise Not_found
    | y :: r -> if y = x then i else go (i + 1) r
  in
  go 0 l

let rec serialize_exported b e vjp_order =
  let unique_avals = uniques (e.in_avals @ e.out_avals) in
  let fun_name = F.create_string b e.fun_name in
  let in_tree = serialize_tree b e.in_tree in
  let in_avals = serialize_array b serialize_aval e.in_avals in
  let out_tree = serialize_tree b e.out_tree in
  let out_avals = serialize_array b serialize_aval e.out_avals in
  let in_shardings =
    serialize_array b serialize_none_sharding (List.init e.in_shardings ignore)
  in
  let out_shardings =
    serialize_array b serialize_none_sharding (List.init e.out_shardings ignore)
  in
  let ordered = serialize_array b serialize_effect e.ordered_effects in
  let unordered = serialize_array b serialize_effect e.unordered_effects in
  let disabled =
    serialize_array b serialize_disabled e.disabled_safety_checks
  in
  let platforms = serialize_array b F.create_string e.platforms in
  let mlir = F.create_byte_vector b e.mlir_module_serialized in
  let kept = F.create_uint_vector b 2 e.module_kept_var_idx in
  let vjp =
    if vjp_order > 0 then
      match e.vjp with
      | None ->
          Py_exn.value_error
            "serialization of an Exported that does not have vjps of \
             high-enough order"
      | Some v -> serialize_exported b v (vjp_order - 1)
    else 0
  in
  let unique_avals_off = serialize_array b serialize_aval unique_avals in
  let meshes = serialize_array b (fun _ () -> 0) [] in
  let named = serialize_array b (fun _ () -> 0) [] in
  let idxs avals = List.map (fun a -> index_of a unique_avals) avals in
  let in_idx = F.create_uint_vector b 4 (idxs e.in_avals) in
  let out_idx = F.create_uint_vector b 4 (idxs e.out_avals) in
  let in_sh_idx =
    F.create_uint_vector b 4 (List.init e.in_shardings (fun _ -> 0))
  in
  let out_sh_idx =
    F.create_uint_vector b 4 (List.init e.out_shardings (fun _ -> 0))
  in
  F.start_object b 26;
  F.add_int_slot b 2 0 3;
  F.add_offset_slot b 1 fun_name;
  F.add_offset_slot b 2 in_tree;
  F.add_offset_slot b 3 in_avals;
  F.add_offset_slot b 4 out_tree;
  F.add_offset_slot b 5 out_avals;
  F.add_int_slot b 4 18 e.nr_devices;
  F.add_offset_slot b 7 in_shardings;
  F.add_offset_slot b 8 out_shardings;
  F.add_offset_slot b 9 platforms;
  F.add_offset_slot b 10 ordered;
  F.add_offset_slot b 11 unordered;
  F.add_offset_slot b 12 disabled;
  F.add_offset_slot b 13 mlir;
  F.add_int_slot b 2 14 e.calling_convention_version;
  F.add_offset_slot b 15 kept;
  F.add_int_slot b 1 16 (if e.uses_global_constants then 1 else 0);
  F.add_offset_slot b 17 vjp;
  F.add_offset_slot b 19 unique_avals_off;
  F.add_offset_slot b 20 meshes;
  F.add_offset_slot b 21 named;
  F.add_offset_slot b 22 in_idx;
  F.add_offset_slot b 23 out_idx;
  F.add_offset_slot b 24 in_sh_idx;
  F.add_offset_slot b 25 out_sh_idx;
  F.end_object b

let serialize ?(vjp_order = 0) e =
  let b = F.create 65536 in
  let root = serialize_exported b e vjp_order in
  F.finish b root;
  F.output b

let deserialize_aval t =
  let memory_space = F.get_int8 t 3 in
  {
    shape = F.strings t 1;
    dtype = F.get_int8 t 2;
    memory_space = (if memory_space = 0 then 1 else memory_space);
  }

let rec deserialize_tree t =
  let children = List.map deserialize_tree (F.tables t 1) in
  match F.get_int8 t 0 with
  | 0 -> T.Leaf_def
  | 1 -> T.Null_def
  | 2 -> T.Tuple_def children
  | 3 -> T.List_def children
  | 4 -> T.Dict_def (List.combine (F.strings t 2) children)
  | 5 -> (
      let name = Option.value ~default:"" (F.get_string t 3) in
      match Export.custom_deserialization name with
      | None ->
          Py_exn.value_error
            (Printf.sprintf
               "Cannot deserialize a PyTreeDef containing an unregistered type \
                `%s`. Use `export.register_pytree_node_serialization` or \
                `export.register_namedtuple_serialization`."
               name)
      | Some rebuild ->
          let rec unit_tree = function
            | T.Leaf_def -> T.Leaf ()
            | T.Null_def -> T.Null
            | T.Tuple_def ds -> T.Tuple (List.map unit_tree ds)
            | T.List_def ds -> T.List (List.map unit_tree ds)
            | T.Dict_def kvs ->
                T.Dict (List.map (fun (k, d) -> (k, unit_tree d)) kvs)
            | T.Custom_def (n, a, ds) -> T.Custom (n, a, List.map unit_tree ds)
          in
          T.tree_structure
            (rebuild (F.get_bytes t 4) (List.map unit_tree children)))
  | k ->
      Py_exn.value_error
        (Printf.sprintf "Cannot deserialize PyTreeDef with unknown kind: %d" k)

let unsupported_sharding () =
  Py_exn.not_implemented_error
    "deserializing an Exported with NamedSharding or HloSharding"

let shardings_count t idx_slot old_slot =
  match F.uints t idx_slot 4 with
  | [] ->
      List.iter
        (fun s -> if F.has s 1 || F.has s 2 then unsupported_sharding ())
        (F.tables t old_slot);
      F.vector_length t old_slot
  | l ->
      if List.exists (fun i -> i <> 0) l then unsupported_sharding ();
      List.length l

let deserialize_disabled t =
  match F.get_int8 t 0 with
  | 1 -> Custom_call (Option.value ~default:"" (F.get_string t 1))
  | 0 -> Platform
  | k ->
      Py_exn.value_error
        (Printf.sprintf
           "Cannot deserialize DisabledSafetyCheck with unknown kind: %d" k)

let rec deserialize_table t =
  let unique = List.map deserialize_aval (F.tables t 19) in
  let avals idx_slot old_slot =
    match F.uints t idx_slot 4 with
    | [] -> List.map deserialize_aval (F.tables t old_slot)
    | l -> List.map (List.nth unique) l
  in
  let effect_name e = Option.value ~default:"" (F.get_string e 0) in
  let nr_devices =
    match F.get_int t 18 4 ~default:0 with
    | 0 -> F.get_int t 6 2 ~default:0
    | n -> n
  in
  let in_tree =
    match F.get_table t 2 with
    | Some p -> deserialize_tree p
    | None -> T.Leaf_def
  in
  let out_tree =
    match F.get_table t 4 with
    | Some p -> deserialize_tree p
    | None -> T.Leaf_def
  in
  {
    fun_name = Option.value ~default:"" (F.get_string t 1);
    in_tree;
    out_tree;
    in_avals = avals 22 3;
    out_avals = avals 23 5;
    nr_devices;
    in_shardings = shardings_count t 24 7;
    out_shardings = shardings_count t 25 8;
    platforms = F.strings t 9;
    ordered_effects = List.map effect_name (F.tables t 10);
    unordered_effects = List.map effect_name (F.tables t 11);
    disabled_safety_checks = List.map deserialize_disabled (F.tables t 12);
    mlir_module_serialized = F.get_bytes t 13;
    calling_convention_version = F.get_int t 14 2 ~default:0;
    module_kept_var_idx = F.uints t 15 2;
    uses_global_constants = F.get_int t 16 1 ~default:0 <> 0;
    vjp = Option.map deserialize_table (F.get_table t 17);
  }

let deserialize s = deserialize_table (F.root s)
let has_vjp e = e.vjp <> None

let disabled_to_string = function
  | Platform -> "platform"
  | Custom_call target -> "custom_call:" ^ target

let aval_str a =
  Dtype.numpy_name (aval_dtype a) ^ "[" ^ String.concat "," a.shape ^ "]"

let describe e =
  String.concat "\n"
    [
      "fun_name=" ^ e.fun_name;
      "in_tree=" ^ T.treedef_to_string e.in_tree;
      "out_tree=" ^ T.treedef_to_string e.out_tree;
      "in_avals=" ^ String.concat ";" (List.map aval_str e.in_avals);
      "out_avals=" ^ String.concat ";" (List.map aval_str e.out_avals);
      "platforms=" ^ String.concat "," e.platforms;
      "nr_devices=" ^ string_of_int e.nr_devices;
      "calling_convention_version=" ^ string_of_int e.calling_convention_version;
      "module_kept_var_idx="
      ^ String.concat "," (List.map string_of_int e.module_kept_var_idx);
      ("uses_global_constants="
      ^ if e.uses_global_constants then "True" else "False");
      "disabled_safety_checks="
      ^ String.concat "," (List.map disabled_to_string e.disabled_safety_checks);
      "ordered_effects=" ^ string_of_int (List.length e.ordered_effects);
      "unordered_effects=" ^ string_of_int (List.length e.unordered_effects);
      "in_shardings="
      ^ String.concat "," (List.init e.in_shardings (fun _ -> "None"));
      "out_shardings="
      ^ String.concat "," (List.init e.out_shardings (fun _ -> "None"));
      ("has_vjp=" ^ if has_vjp e then "True" else "False");
    ]

let python_type_name = function
  | T.Leaf_def -> "int"
  | T.Null_def -> "NoneType"
  | T.Tuple_def _ -> "tuple"
  | T.List_def _ -> "list"
  | T.Dict_def _ -> "dict"
  | T.Custom_def (n, _, _) -> n

let path_to_string = function
  | T.Sequence_key 0 :: rest -> "args" ^ T.keystr rest
  | T.Sequence_key 1 :: rest -> "kwargs" ^ T.keystr rest
  | path -> T.keystr path

let rec equality_errors path d1 d2 =
  let strict_leaf = function T.Leaf_def -> true | _ -> false in
  if strict_leaf d1 && strict_leaf d2 then []
  else
    let n1 = python_type_name d1 and n2 = python_type_name d2 in
    if n1 <> n2 then
      [
        ( path,
          "<class '" ^ n1 ^ "'>",
          "<class '" ^ n2 ^ "'>",
          "their Python types differ" );
      ]
    else
      let c1 = T.treedef_children d1 and c2 = T.treedef_children d2 in
      let len_mismatch =
        match d1 with
        | (T.Tuple_def _ | T.List_def _) when List.length c1 <> List.length c2
          ->
            Some
              [
                ( path,
                  Printf.sprintf "%s of length %d" n1 (List.length c1),
                  Printf.sprintf "%s of length %d" n2 (List.length c2),
                  "the lengths do not match" );
              ]
        | _ -> None
      in
      match len_mismatch with
      | Some e -> e
      | None ->
          let keys = function
            | T.Dict_def kvs -> List.map (fun (k, _) -> T.Dict_key k) kvs
            | d ->
                List.mapi (fun i _ -> T.Sequence_key i) (T.treedef_children d)
          in
          let k1 = keys d1 and k2 = keys d2 in
          let children n = if n > 1 then "ren" else "" in
          if List.length c1 <> List.length c2 then
            let diff =
              List.filter_map
                (fun k ->
                  match k with
                  | T.Dict_key s when not (List.mem k k2 && List.mem k k1) ->
                      Some ("'" ^ s ^ "'")
                  | _ -> None)
                (k1 @ k2)
            in
            [
              ( path,
                Printf.sprintf "<class '%s'> with %d child%s" n1
                  (List.length c1)
                  (children (List.length c1)),
                Printf.sprintf "<class '%s'> with %d child%s" n2
                  (List.length c2)
                  (children (List.length c2)),
                "the numbers of children do not match"
                ^
                if diff = [] then ""
                else
                  ", with the symmetric difference of key sets: {"
                  ^ String.concat " " diff ^ "}" );
            ]
          else if k1 <> k2 then
            [
              ( path,
                Printf.sprintf "<class '%s'> with pytree metadata" n1,
                Printf.sprintf "<class '%s'> with pytree metadata" n2,
                "the pytree node metadata does not match" );
            ]
          else
            List.concat
              (List.map2
                 (fun (k, a) b -> equality_errors (path @ [ k ]) a b)
                 (List.combine k1 c1) c2)

let check_tree e def =
  if def <> e.in_tree then
    Py_exn.value_error
      (Printf.sprintf
         "The invocation args and kwargs must have the same pytree structure \
          as when the function '%s' was exported, but they have the following \
          structural differences:\n"
         e.fun_name
      ^ String.concat "\n"
          (List.map
             (fun (path, a, b, why) ->
               Printf.sprintf
                 "   - %s is a %s in the invocation and a %s when exported, so \
                  %s.\n"
                 (path_to_string path) a b why)
             (equality_errors [] def e.in_tree)))

let python_tuple l =
  match l with [ x ] -> "(" ^ x ^ ",)" | l -> "(" ^ String.concat ", " l ^ ")"

let arg_names e =
  List.map
    (fun (path, ()) -> path_to_string path)
    (T.tree_leaves_with_path
       (T.tree_unflatten e.in_tree
          (List.init (T.count_leaves e.in_tree) (fun _ -> ()))))

let is_constant s = int_of_string_opt s <> None

let check_avals e (actual : Types.aval list) =
  let names = arg_names e in
  List.iteri
    (fun i (exp, (act : Types.aval)) ->
      let name = List.nth names i in
      let exp_s = python_tuple exp.shape in
      let act_s = Shape_poly.int_shape_to_string act.shape in
      if List.length exp.shape <> Array.length act.shape then
        Py_exn.value_error
          (Printf.sprintf "Rank mismatch for %s: expected %s and called with %s"
             name exp_s act_s);
      if aval_dtype exp <> act.dtype then
        Py_exn.value_error
          (Printf.sprintf
             "Dtype mismatch for %s: expected %s and called with %s" name
             (Dtype.numpy_name (aval_dtype exp))
             (Dtype.numpy_name act.dtype));
      List.iteri
        (fun d s ->
          match int_of_string_opt s with
          | Some k when k <> act.shape.(d) ->
              Py_exn.value_error
                (Printf.sprintf
                   "Shape mismatch for %s.shape[%d] (expected same constant): \
                    expected %s and called with %s"
                   name d exp_s act_s)
          | _ -> ())
        exp.shape)
    (List.combine e.in_avals actual)

let symbolic_specs e =
  let scope = Shape_poly.symbolic_scope () in
  List.map
    (fun a ->
      {
        Shape_poly.shape =
          (if a.shape = [] then [||]
           else
             Shape_poly.symbolic_shape ~scope (Some (String.concat "," a.shape)));
        dtype = aval_dtype a;
      })
    e.in_avals

let check_platform e =
  let current = Config.current_platform () in
  if
    (not (List.mem Platform e.disabled_safety_checks))
    && not (List.mem current e.platforms)
  then
    Py_exn.value_error
      (Printf.sprintf
         "Function '%s' was exported for platforms '%s' but it is used on '%s'."
         e.fun_name
         (python_tuple (List.map (fun p -> "'" ^ p ^ "'") e.platforms))
         (python_tuple [ "'" ^ current ^ "'" ]))

let run_module e args =
  let kept = List.map (List.nth args) e.module_kept_var_idx in
  let kept =
    match e.platforms with
    | [] | [ _ ] -> kept
    | platforms ->
        let current = Config.current_platform () in
        let rec index i = function
          | [] -> 0
          | p :: r -> if p = current then i else index (i + 1) r
        in
        Types.Concrete
          (Ndarray.of_floats Dtype.I32 [||]
             [| float_of_int (index 0 platforms) |])
        :: kept
  in
  Backend.run_text e.mlir_module_serialized kept

let canonical_leaf = function
  | Types.Concrete nd ->
      let dt = Dtypes.canonicalize_dtype (Ndarray.dtype nd) in
      if dt = Ndarray.dtype nd then Types.Concrete nd
      else Types.Concrete (Ndarray.canonicalize dt nd)
  | v -> v

let call_checked e args =
  let leaves, def = T.tree_flatten args in
  let leaves = List.map canonical_leaf leaves in
  check_tree e (T.Tuple_def [ def; T.Dict_def [] ]);
  check_platform e;
  let actual = List.map Core.get_aval leaves in
  check_avals e actual;
  let specs = symbolic_specs e in
  if Shape_poly.all_dim_vars specs <> [] then begin
    ignore
      (Shape_poly.solve_dim_vars ~arg_names:(arg_names e) specs
         (List.map (fun (a : Types.aval) -> a.shape) actual));
    Py_exn.not_implemented_error
      "calling a deserialized shape-polymorphic Exported needs StableHLO shape \
       refinement, which ojax does not implement"
  end;
  T.tree_unflatten e.out_tree
    (List.map Backend.to_host_value (run_module e leaves))

let call e args =
  try call_checked e args with Invalid_argument m -> Py_exn.value_error m

let of_disabled c =
  match Export.is_custom_call c with
  | Some target -> Custom_call target
  | None -> Platform

let of_exported (e : Export.exported) =
  let aval (a : Types.aval) = aval_of_shape a.dtype a.shape in
  {
    fun_name = e.fun_name;
    in_tree = e.in_tree;
    out_tree = e.out_tree;
    in_avals = List.map aval e.in_avals;
    out_avals = List.map aval e.out_avals;
    nr_devices = e.nr_devices;
    in_shardings = List.length e.in_avals;
    out_shardings = List.length e.out_avals;
    platforms = e.platforms;
    ordered_effects = [];
    unordered_effects = [];
    disabled_safety_checks = List.map of_disabled e.disabled_safety_checks;
    mlir_module_serialized = e.mlir_module;
    calling_convention_version = e.calling_convention_version;
    module_kept_var_idx = e.module_kept_var_idx;
    uses_global_constants = e.uses_global_constants;
    vjp = None;
  }
