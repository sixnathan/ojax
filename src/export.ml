open Types
module T = Tree_util

type disabled_safety_check = Platform_check | Custom_call_check of string

let platform_check = Platform_check
let custom_call_check target = Custom_call_check target

let is_custom_call = function
  | Custom_call_check t -> Some t
  | Platform_check -> None

let disabled_check_to_string = function
  | Platform_check -> "platform"
  | Custom_call_check t -> "custom_call:" ^ t

let disabled_check_of_string s =
  let prefix = "custom_call:" in
  if s = "platform" then Platform_check
  else if String.starts_with ~prefix s then
    let n = String.length prefix in
    Custom_call_check (String.sub s n (String.length s - n))
  else invalid_arg ("deserialize: unknown safety check " ^ s)

let minimum_supported_calling_convention_version = 9
let maximum_supported_calling_convention_version = 10
let default_export_platform () = Config.current_platform ()

type exported = {
  fun_name : string;
  in_tree : T.treedef;
  out_tree : T.treedef;
  mlir_module : string;
  in_avals : aval list;
  out_avals : aval list;
  platforms : string list;
  disabled_safety_checks : disabled_safety_check list;
  nr_devices : int;
  calling_convention_version : int;
  module_kept_var_idx : int list;
  uses_global_constants : bool;
  get_vjp : (exported -> exported) option;
}

let cpu_ffi_kernels =
  [
    "lapack_sgeqrf_ffi"; "lapack_dgeqrf_ffi"; "lapack_cgeqrf_ffi";
    "lapack_zgeqrf_ffi"; "lapack_sorgqr_ffi"; "lapack_dorgqr_ffi";
    "lapack_cungqr_ffi"; "lapack_zungqr_ffi"; "lapack_spotrf_ffi";
    "lapack_dpotrf_ffi"; "lapack_cpotrf_ffi"; "lapack_zpotrf_ffi";
    "lapack_sgesdd_ffi"; "lapack_dgesdd_ffi"; "lapack_cgesdd_ffi";
    "lapack_zgesdd_ffi"; "lapack_ssyevd_ffi"; "lapack_dsyevd_ffi";
    "lapack_cheevd_ffi"; "lapack_zheevd_ffi"; "lapack_sgeev_ffi";
    "lapack_dgeev_ffi"; "lapack_cgeev_ffi"; "lapack_zgeev_ffi";
    "lapack_sgees_ffi"; "lapack_dgees_ffi"; "lapack_cgees_ffi";
    "lapack_zgees_ffi"; "lapack_sgehrd_ffi"; "lapack_dgehrd_ffi";
    "lapack_cgehrd_ffi"; "lapack_zgehrd_ffi"; "lapack_ssytrd_ffi";
    "lapack_dsytrd_ffi"; "lapack_chetrd_ffi"; "lapack_zhetrd_ffi";
    "lapack_sgetrf_ffi"; "lapack_dgetrf_ffi"; "lapack_cgetrf_ffi";
    "lapack_zgetrf_ffi"; "lapack_strsm_ffi"; "lapack_dtrsm_ffi";
    "lapack_ctrsm_ffi"; "lapack_ztrsm_ffi"; "lapack_sgtsv_ffi";
    "lapack_dgtsv_ffi"; "lapack_cgtsv_ffi"; "lapack_zgtsv_ffi";
  ]
[@@ocamlformat "disable"]

let gpu_ffi_kernels =
  [ "cu_lu_pivots_to_permutation"; "cusolver_getrf_ffi"; "hip_lu_pivots_to_permutation"; "hipsolver_getrf_ffi"; "cusolver_geqrf_ffi"; "cusolver_orgqr_ffi"; "hipsolver_geqrf_ffi"; "hipsolver_orgqr_ffi"; "cusolver_potrf_ffi"; "hipsolver_potrf_ffi"; "cusolver_syevd_ffi"; "hipsolver_syevd_ffi"; "cusolver_gesvd_ffi"; "cusolver_gesvdj_ffi"; "hipsolver_gesvd_ffi"; "hipsolver_gesvdj_ffi"; "hipsolver_gesdd_ffi"; "cusolver_sytrd_ffi"; "hipsolver_sytrd_ffi"; "cusparse_gtsv2_ffi" ]
[@@ocamlformat "disable"]

let stable_targets =
  cpu_ffi_kernels
  @ gpu_ffi_kernels
  @ [
      "Sharding"; "SPMDFullToShardShape"; "SPMDShardToFullShape";
      "annotate_device_placement"; "cu_threefry2x32_ffi";
      "hip_threefry2x32_ffi"; "Eigh"; "Qr";
      "ProductOfElementaryHouseholderReflectors"; "LuDecomposition";
      "ApproxTopK"; "stablehlo.dynamic_approx_top_k"; "tf.call_tf_function";
      "tpu_custom_call"; "AllocateBuffer"; "mosaic_gpu_v2";
      "stablehlo.dynamic_reduce_window"; "stablehlo.dynamic_rng_bit_generator";
      "stablehlo.dynamic_top_k"; "shape_assertion";
    ]
[@@ocamlformat "disable"]

let custom_call_marker = "stablehlo.custom_call @"

let custom_call_ops text =
  List.filter_map
    (fun line ->
      let m = String.length custom_call_marker in
      let n = String.length line in
      let rec find i =
        if i + m > n then None
        else if String.sub line i m = custom_call_marker then Some (i + m)
        else find (i + 1)
      in
      match find 0 with
      | None -> None
      | Some start ->
          let stop = String.index_from line start '(' in
          Some (String.sub line start (stop - start), String.trim line))
    (String.split_on_char '\n' text)

let check_module text disabled =
  let allowed = stable_targets @ List.filter_map is_custom_call disabled in
  if not (List.mem "ALL" allowed) then
    let bad =
      List.filter_map
        (fun (target, op) -> if List.mem target allowed then None else Some op)
        (custom_call_ops text)
    in
    if bad <> [] then
      invalid_arg
        ("Cannot serialize code with custom calls whose targets have no \
          compatibility guarantees. See \
          https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls. \
          Examples are:\n" ^ String.concat "\n" bad)

let check_version () =
  let version = Config.value Config.export_calling_convention_version in
  if
    version < minimum_supported_calling_convention_version
    || version > maximum_supported_calling_convention_version
  then
    invalid_arg
      (Printf.sprintf
         "The requested export calling convention version %d is outside the \
          range of supported versions [%d..%d]"
         version minimum_supported_calling_convention_version
         maximum_supported_calling_convention_version);
  version

let atom_uses = function A_var v -> [ v.vid ] | A_lit _ | DropVar _ -> []

let kept_indices (cj : closed_jaxpr) =
  let jx = cj.jaxpr in
  let n_consts = List.length cj.consts in
  let used = Hashtbl.create 16 in
  let mark a = List.iter (fun id -> Hashtbl.replace used id ()) (atom_uses a) in
  List.iter (fun (e : eqn) -> List.iter mark e.inputs) jx.eqns;
  List.iter mark jx.outs;
  let binders = List.filteri (fun i _ -> i >= n_consts) jx.in_binders in
  List.filter_map
    (fun (i, (v : var)) -> if Hashtbl.mem used v.vid then Some i else None)
    (List.mapi (fun i v -> (i, v)) binders)

let prune_unused (cj : closed_jaxpr) kept =
  let jx = cj.jaxpr in
  let n_consts = List.length cj.consts in
  let in_binders =
    List.filteri
      (fun i _ -> i < n_consts || List.mem (i - n_consts) kept)
      jx.in_binders
  in
  { cj with jaxpr = { jx with in_binders } }

let args_tree_of_avals avals =
  T.Tuple_def
    [ T.Tuple_def (List.map (fun _ -> T.Leaf_def) avals); T.Dict_def [] ]

let flat_vjp f in_count =
 fun args_and_cts ->
  let args = List.filteri (fun i _ -> i < in_count) args_and_cts in
  let cts = List.filteri (fun i _ -> i >= in_count) args_and_cts in
  let _, pullback = Interpreters.Ad.vjp f args in
  pullback cts

let rec export_internal ~fun_name ~platforms ~disabled_checks ~in_tree ~out_tree
    f avals =
  let version = check_version () in
  let platforms =
    match platforms with Some p -> p | None -> [ default_export_platform () ]
  in
  let lowered = Stages.lower f avals in
  let result_names, out_tree =
    match out_tree with
    | Some t ->
        let d = t () in
        (Some (Stages.result_names_of_tree d), d)
    | None ->
        ( None,
          T.Tuple_def (List.map (fun _ -> T.Leaf_def) lowered.Stages.out_avals)
        )
  in
  let { Stablehlo.Emit.text; kept } =
    Stablehlo.Emit.emit_export_text ~name:fun_name ~platforms ?result_names
      ~uses_shape_polymorphism:(List.length platforms > 1)
      lowered.Stages.closed
  in
  if Stablehlo.Mlir_names.mentions_host_callback text then
    failwith "serialization of host_callbacks is not yet implemented";
  check_module text disabled_checks;
  let get_vjp (primal : exported) =
    let tangent_avals =
      List.map (fun (a : aval) -> { a with weak_type = false }) primal.out_avals
    in
    let vjp_in = primal.in_avals @ tangent_avals in
    export_internal
      ~fun_name:("jax.vjp(" ^ primal.fun_name ^ ")")
      ~platforms:(Some primal.platforms)
      ~disabled_checks:primal.disabled_safety_checks
      ~in_tree:(args_tree_of_avals vjp_in)
      ~out_tree:None
      (flat_vjp f (List.length primal.in_avals))
      vjp_in
  in
  {
    fun_name;
    in_tree;
    out_tree;
    mlir_module = text;
    in_avals = lowered.Stages.in_avals;
    out_avals = lowered.Stages.out_avals;
    platforms;
    disabled_safety_checks = disabled_checks;
    nr_devices = 1;
    calling_convention_version = version;
    module_kept_var_idx = kept;
    uses_global_constants = List.length platforms > 1;
    get_vjp = Some get_vjp;
  }

let export ?(fun_name = "exported") ?platforms ?(disabled_checks = []) f avals =
  export_internal ~fun_name ~platforms ~disabled_checks
    ~in_tree:(args_tree_of_avals avals) ~out_tree:None f avals

let export_tree ?(fun_name = "exported") ?platforms ?(disabled_checks = []) f
    args =
  let leaves, args_def = T.tree_flatten args in
  let result_def = ref None in
  let flat values =
    let result = f (T.tree_unflatten args_def values) in
    let out_leaves, out_def = T.tree_flatten result in
    result_def := Some out_def;
    out_leaves
  in
  export_internal ~fun_name ~platforms ~disabled_checks
    ~in_tree:(T.Tuple_def [ args_def; T.Dict_def [] ])
    ~out_tree:
      (Some
         (fun () ->
           match !result_def with
           | Some d -> d
           | None -> failwith "export_tree: function was not traced"))
    flat leaves

let serialization_registry : (string, string * (string -> string)) Hashtbl.t =
  Hashtbl.create 8

let deserialization_registry :
    ( string,
      string * (string -> string) * (string -> unit T.t list -> unit T.t) )
    Hashtbl.t =
  Hashtbl.create 8

let register_pytree_node_serialization ~serialized_name ~serialize_auxdata
    ~deserialize_auxdata ?from_children node_type =
  (match Hashtbl.find_opt serialization_registry node_type with
  | Some (previous, _) ->
      invalid_arg
        (Printf.sprintf
           "Duplicate serialization registration for type `%s`. Previous \
            registration was with serialized_name `%s`."
           node_type previous)
  | None -> ());
  (match Hashtbl.find_opt deserialization_registry serialized_name with
  | Some (previous, _, _) ->
      invalid_arg
        (Printf.sprintf
           "Duplicate serialization registration for serialized_name `%s`. \
            Previous registration was for type `%s`."
           serialized_name previous)
  | None -> ());
  let from_children =
    match from_children with
    | Some g -> g
    | None -> fun aux children -> T.Custom (node_type, aux, children)
  in
  Hashtbl.replace serialization_registry node_type
    (serialized_name, serialize_auxdata);
  Hashtbl.replace deserialization_registry serialized_name
    (node_type, deserialize_auxdata, from_children)

let custom_serialization node_type =
  Hashtbl.find_opt serialization_registry node_type

let custom_deserialization serialized_name =
  match Hashtbl.find_opt deserialization_registry serialized_name with
  | None -> None
  | Some (_, deserialize_auxdata, from_children) ->
      Some
        (fun aux children -> from_children (deserialize_auxdata aux) children)

let json_quote s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (function
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let json_of_strings l = "[" ^ String.concat ", " (List.map json_quote l) ^ "]"

let strings_of_json text =
  let n = String.length text in
  let rec go i acc =
    if i >= n then List.rev acc
    else
      match text.[i] with
      | '"' ->
          let b = Buffer.create 16 in
          let rec str j =
            match text.[j] with
            | '"' -> j + 1
            | '\\' ->
                Buffer.add_char b text.[j + 1];
                str (j + 2)
            | c ->
                Buffer.add_char b c;
                str (j + 1)
          in
          let next = str (i + 1) in
          go next (Buffer.contents b :: acc)
      | _ -> go (i + 1) acc
  in
  go 0 []

let register_namedtuple_serialization ~serialized_name ~fields node_type =
  let from_children aux children =
    let ser_keys = strings_of_json aux in
    if List.length ser_keys <> List.length children then
      invalid_arg "deserialize: namedtuple field count differs from children";
    let by_key = List.combine ser_keys children in
    let ordered =
      List.map
        (fun field ->
          match List.assoc_opt field by_key with
          | Some c -> c
          | None ->
              invalid_arg
                (Printf.sprintf "%s() missing required argument: '%s'" node_type
                   field))
        fields
    in
    List.iter
      (fun key ->
        if not (List.mem key fields) then
          invalid_arg
            (Printf.sprintf "%s() got an unexpected keyword argument '%s'"
               node_type key))
      ser_keys;
    T.Custom (node_type, "", ordered)
  in
  register_pytree_node_serialization ~serialized_name
    ~serialize_auxdata:(fun _ -> json_of_strings fields)
    ~deserialize_auxdata:Fun.id ~from_children node_type;
  T.register_namedtuple node_type

let add_field b s =
  Buffer.add_string b (string_of_int (String.length s));
  Buffer.add_char b ':';
  Buffer.add_string b s

let read_field text pos =
  let colon =
    try String.index_from text !pos ':'
    with Not_found -> invalid_arg "deserialize: truncated module"
  in
  let len =
    match int_of_string_opt (String.sub text !pos (colon - !pos)) with
    | Some l -> l
    | None -> invalid_arg "deserialize: malformed field"
  in
  if colon + 1 + len > String.length text then
    invalid_arg "deserialize: truncated module";
  pos := colon + 1 + len;
  String.sub text (colon + 1) len

let rec encode_treedef b = function
  | T.Leaf_def -> Buffer.add_char b '*'
  | T.Null_def -> Buffer.add_char b 'n'
  | T.Tuple_def ds ->
      Buffer.add_char b 't';
      add_field b (string_of_int (List.length ds));
      List.iter (encode_treedef b) ds
  | T.List_def ds ->
      Buffer.add_char b 'l';
      add_field b (string_of_int (List.length ds));
      List.iter (encode_treedef b) ds
  | T.Dict_def kvs ->
      Buffer.add_char b 'd';
      add_field b (string_of_int (List.length kvs));
      List.iter
        (fun (k, d) ->
          add_field b k;
          encode_treedef b d)
        kvs
  | T.Custom_def (name, aux, ds) -> (
      match Hashtbl.find_opt serialization_registry name with
      | None ->
          invalid_arg
            (Printf.sprintf
               "Cannot serialize PyTreeDef containing an unregistered type \
                `%s`. Use `export.register_pytree_node_serialization` or \
                `export.register_namedtuple_serialization`."
               name)
      | Some (serialized_name, serialize_auxdata) ->
          Buffer.add_char b 'c';
          add_field b serialized_name;
          add_field b (serialize_auxdata aux);
          add_field b (string_of_int (List.length ds));
          List.iter (encode_treedef b) ds)

let rec unit_tree_of_def = function
  | T.Leaf_def -> T.Leaf ()
  | T.Null_def -> T.Null
  | T.Tuple_def ds -> T.Tuple (List.map unit_tree_of_def ds)
  | T.List_def ds -> T.List (List.map unit_tree_of_def ds)
  | T.Dict_def kvs ->
      T.Dict (List.map (fun (k, d) -> (k, unit_tree_of_def d)) kvs)
  | T.Custom_def (n, a, ds) -> T.Custom (n, a, List.map unit_tree_of_def ds)

let rec decode_treedef text pos =
  if !pos >= String.length text then
    invalid_arg "deserialize: truncated treedef";
  let kind = text.[!pos] in
  incr pos;
  let count () = int_of_string (read_field text pos) in
  let children () =
    let n = count () in
    List.init n (fun _ -> ()) |> List.map (fun () -> decode_treedef text pos)
  in
  match kind with
  | '*' -> T.Leaf_def
  | 'n' -> T.Null_def
  | 't' -> T.Tuple_def (children ())
  | 'l' -> T.List_def (children ())
  | 'd' ->
      let n = count () in
      let items =
        List.init n (fun _ -> ())
        |> List.map (fun () ->
            let k = read_field text pos in
            (k, decode_treedef text pos))
      in
      T.Dict_def items
  | 'c' -> (
      let serialized_name = read_field text pos in
      let aux = read_field text pos in
      let ds = children () in
      match Hashtbl.find_opt deserialization_registry serialized_name with
      | None ->
          invalid_arg
            (Printf.sprintf
               "Cannot deserialize a PyTreeDef containing an unregistered type \
                `%s`. Use `export.register_pytree_node_serialization` or \
                `export.register_namedtuple_serialization`."
               serialized_name)
      | Some (_, deserialize_auxdata, from_children) ->
          let rebuilt =
            from_children (deserialize_auxdata aux)
              (List.map unit_tree_of_def ds)
          in
          T.tree_structure rebuilt)
  | c -> invalid_arg (Printf.sprintf "deserialize: unknown tree kind %c" c)

let treedef_to_string d =
  let b = Buffer.create 32 in
  encode_treedef b d;
  Buffer.contents b

let treedef_of_string s = decode_treedef s (ref 0)

let aval_to_string (a : aval) =
  Dtype.short_name a.dtype ^ ":"
  ^ (if a.weak_type then "w:" else "")
  ^ String.concat "," (Array.to_list (Array.map string_of_int a.shape))

let dtype_of_name name =
  match List.find_opt (fun d -> Dtype.short_name d = name) Dtype.all with
  | Some d -> d
  | None -> invalid_arg ("deserialize: unknown dtype " ^ name)

let aval_of_string s =
  let parse dt dims weak_type =
    let shape =
      if dims = "" then [||]
      else
        Array.of_list (List.map int_of_string (String.split_on_char ',' dims))
    in
    { shape; dtype = dtype_of_name dt; weak_type }
  in
  match String.split_on_char ':' s with
  | [ dt; dims ] -> parse dt dims false
  | [ dt; "w"; dims ] -> parse dt dims true
  | _ -> invalid_arg "deserialize: malformed type"

let avals_of_string s =
  if s = "" then [] else List.map aval_of_string (String.split_on_char ';' s)

let strings_of_list s = if s = "" then [] else String.split_on_char ';' s
let magic_v1 = "ojax-exported-v1"
let magic_v2 = "ojax-exported-v2"

let rec serialize ?(vjp_order = 0) e =
  let vjp_blob =
    if vjp_order <= 0 then ""
    else
      match e.get_vjp with
      | None ->
          invalid_arg
            "Cannot serialize a VJP: this Exported has no VJP available"
      | Some g -> serialize ~vjp_order:(vjp_order - 1) (g e)
  in
  let b = Buffer.create 256 in
  Buffer.add_string b magic_v2;
  Buffer.add_char b '\n';
  List.iter (add_field b)
    [
      e.fun_name;
      treedef_to_string e.in_tree;
      treedef_to_string e.out_tree;
      String.concat ";" (List.map aval_to_string e.in_avals);
      String.concat ";" (List.map aval_to_string e.out_avals);
      String.concat ";" e.platforms;
      String.concat ";"
        (List.map disabled_check_to_string e.disabled_safety_checks);
      string_of_int e.nr_devices;
      string_of_int e.calling_convention_version;
      String.concat ";" (List.map string_of_int e.module_kept_var_idx);
      string_of_bool e.uses_global_constants;
      e.mlir_module;
      vjp_blob;
    ];
  Buffer.contents b

let split_once s =
  match String.index_opt s '\n' with
  | Some i -> (String.sub s 0 i, String.sub s (i + 1) (String.length s - i - 1))
  | None -> invalid_arg "deserialize: truncated module"

let deserialize_v1 rest =
  let inputs, rest = split_once rest in
  let outputs, rest = split_once rest in
  let platforms, mlir_module = split_once rest in
  let in_avals = avals_of_string inputs in
  let out_avals = avals_of_string outputs in
  {
    fun_name = "exported";
    in_tree = args_tree_of_avals in_avals;
    out_tree = T.Tuple_def (List.map (fun _ -> T.Leaf_def) out_avals);
    mlir_module;
    in_avals;
    out_avals;
    platforms = String.split_on_char ';' platforms;
    disabled_safety_checks = [];
    nr_devices = 1;
    calling_convention_version = 9;
    module_kept_var_idx = List.mapi (fun i _ -> i) in_avals;
    uses_global_constants = false;
    get_vjp = None;
  }

let rec deserialize text =
  let header, rest = split_once text in
  if header = magic_v1 then deserialize_v1 rest
  else if header <> magic_v2 then invalid_arg "deserialize: unknown format"
  else
    let pos = ref 0 in
    let next () = read_field rest pos in
    let fun_name = next () in
    let in_tree = treedef_of_string (next ()) in
    let out_tree = treedef_of_string (next ()) in
    let in_avals = avals_of_string (next ()) in
    let out_avals = avals_of_string (next ()) in
    let platforms = strings_of_list (next ()) in
    let disabled_safety_checks =
      List.map disabled_check_of_string (strings_of_list (next ()))
    in
    let nr_devices = int_of_string (next ()) in
    let calling_convention_version = int_of_string (next ()) in
    let module_kept_var_idx =
      List.map int_of_string (strings_of_list (next ()))
    in
    let uses_global_constants = bool_of_string (next ()) in
    let mlir_module = next () in
    let vjp_blob = next () in
    {
      fun_name;
      in_tree;
      out_tree;
      mlir_module;
      in_avals;
      out_avals;
      platforms;
      disabled_safety_checks;
      nr_devices;
      calling_convention_version;
      module_kept_var_idx;
      uses_global_constants;
      get_vjp =
        (if vjp_blob = "" then None else Some (fun _ -> deserialize vjp_blob));
    }

let has_vjp e = e.get_vjp <> None

let vjp e =
  match e.get_vjp with
  | Some g -> g e
  | None -> invalid_arg "No VJP is available"

let python_tuple l =
  match l with
  | [ x ] -> "('" ^ x ^ "',)"
  | l -> "(" ^ String.concat ", " (List.map (fun x -> "'" ^ x ^ "'") l) ^ ")"

let current_platforms () = [ Config.current_platform () ]

let check_platform e =
  if
    (not (List.mem Platform_check e.disabled_safety_checks))
    && not
         (List.exists (fun p -> List.mem p e.platforms) (current_platforms ()))
  then
    invalid_arg
      (Printf.sprintf
         "Function '%s' was exported for platforms '%s' but it is used on '%s'."
         e.fun_name (python_tuple e.platforms)
         (python_tuple (current_platforms ())))

let arg_descriptors in_tree =
  let paths =
    List.map fst
      (T.tree_leaves_with_path
         (T.tree_unflatten in_tree
            (List.init (T.count_leaves in_tree) (fun _ -> ()))))
  in
  List.map
    (function
      | T.Sequence_key 0 :: rest -> "args" ^ T.keystr rest
      | T.Sequence_key 1 :: rest -> "kwargs" ^ T.keystr rest
      | path -> T.keystr path)
    paths

let check_call_avals fun_name in_tree expected actual =
  if List.length expected <> List.length actual then
    invalid_arg
      (Printf.sprintf
         "The invocation args and kwargs must have the same pytree structure \
          as when the function '%s' was exported, but they have the following \
          structural differences:\n"
         fun_name);
  let names = arg_descriptors in_tree in
  List.iteri
    (fun i ((exp : aval), (act : aval)) ->
      let name = List.nth names i in
      let exp_s = Shape_poly.int_shape_to_string exp.shape in
      let act_s = Shape_poly.int_shape_to_string act.shape in
      if Array.length exp.shape <> Array.length act.shape then
        invalid_arg
          (Printf.sprintf "Rank mismatch for %s: expected %s and called with %s"
             name exp_s act_s);
      if exp.dtype <> act.dtype then
        invalid_arg
          (Printf.sprintf
             "Dtype mismatch for %s: expected %s and called with %s" name
             (Dtype.numpy_name exp.dtype)
             (Dtype.numpy_name act.dtype));
      Array.iteri
        (fun d e ->
          if e <> act.shape.(d) then
            invalid_arg
              (Printf.sprintf
                 "Shape mismatch for %s.shape[%d] (expected same constant): \
                  expected %s and called with %s"
                 name d exp_s act_s))
        exp.shape)
    (List.combine expected actual)

let is_tracer = function Tracer _ -> true | Concrete _ | Device _ -> false

let is_batched = function
  | Tracer { payload = Batch _; _ } -> true
  | Tracer _ | Concrete _ | Device _ -> false

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
        Concrete
          (Ndarray.of_floats Dtype.I32 [||]
             [| float_of_int (index 0 platforms) |])
        :: kept
  in
  Backend.run_text e.mlir_module kept

let host_args args = List.map Backend.to_host_value args

let rec call e args =
  check_platform e;
  check_call_avals e.fun_name e.in_tree e.in_avals (List.map Core.get_aval args);
  if List.exists is_batched args then
    failwith "Batching rule for 'call_exported' not implemented";
  call_checked e args

and call_checked e args =
  if List.exists is_tracer args then call_traced e args else run_module e args

and call_traced e args =
  let primal xs =
    if List.exists is_tracer xs then
      Custom_derivatives.pure_callback
        (fun concrete -> run_module e (host_args concrete))
        ~result_avals:e.out_avals xs
    else run_module e xs
  in
  match e.get_vjp with
  | None -> primal args
  | Some _ ->
      Custom_derivatives.custom_vjp primal
        ~fwd:(fun xs ->
          (Core.with_eval_dynamic (fun () -> call_checked e xs), xs))
        ~bwd:(fun residuals cts -> call (vjp e) (residuals @ cts))
        args

let call_tree e args =
  let leaves, def = T.tree_flatten args in
  let wanted =
    match e.in_tree with T.Tuple_def [ d; T.Dict_def [] ] -> d | d -> d
  in
  if def <> wanted then
    invalid_arg
      (Printf.sprintf
         "The invocation args and kwargs must have the same pytree structure \
          as when the function '%s' was exported, but they have the following \
          structural differences:\n"
         e.fun_name);
  T.tree_unflatten e.out_tree (call e leaves)

let symbolic_scope = Shape_poly.symbolic_scope
let symbolic_shape = Shape_poly.symbolic_shape
let symbolic_args_specs = Shape_poly.symbolic_args_specs
let is_symbolic_dim = Shape_poly.is_symbolic_dim

type symbolic_exported = {
  sym_fun_name : string;
  sym_platforms : string list option;
  sym_disabled : disabled_safety_check list;
  sym_fun : value list -> value list;
  sym_in_specs : Shape_poly.spec list;
  sym_cache : (int array list, exported) Hashtbl.t;
}

let export_symbolic ?(fun_name = "exported") ?platforms ?(disabled_checks = [])
    f specs =
  Shape_poly.check_scopes ~when_:("when exporting " ^ fun_name) specs;
  {
    sym_fun_name = fun_name;
    sym_platforms = platforms;
    sym_disabled = disabled_checks;
    sym_fun = f;
    sym_in_specs = specs;
    sym_cache = Hashtbl.create 4;
  }

let symbolic_in_specs s = s.sym_in_specs

let check_symbolic_args s (actual : aval list) =
  if List.length actual <> List.length s.sym_in_specs then
    invalid_arg
      (Printf.sprintf
         "The invocation args and kwargs must have the same pytree structure \
          as when the function '%s' was exported, but they have the following \
          structural differences:\n"
         s.sym_fun_name);
  List.iteri
    (fun i ((spec : Shape_poly.spec), (a : aval)) ->
      let name = Printf.sprintf "args[%d]" i in
      let exp_s = Shape_poly.shape_to_string spec.shape in
      let act_s = Shape_poly.int_shape_to_string a.shape in
      if Array.length spec.shape <> Array.length a.shape then
        invalid_arg
          (Printf.sprintf "Rank mismatch for %s: expected %s and called with %s"
             name exp_s act_s);
      if spec.dtype <> a.dtype then
        invalid_arg
          (Printf.sprintf
             "Dtype mismatch for %s: expected %s and called with %s" name
             (Dtype.numpy_name spec.dtype)
             (Dtype.numpy_name a.dtype));
      Array.iteri
        (fun d e ->
          match Shape_poly.to_constant e with
          | Some k when (not (Shape_poly.is_symbolic_dim e)) && k <> a.shape.(d)
            ->
              invalid_arg
                (Printf.sprintf
                   "Shape mismatch for %s.shape[%d] (expected same constant): \
                    expected %s and called with %s"
                   name d exp_s act_s)
          | _ -> ())
        spec.shape)
    (List.combine s.sym_in_specs actual)

let specialize s args =
  let avals = List.map Core.get_aval args in
  check_symbolic_args s avals;
  let shapes = List.map (fun (a : aval) -> a.shape) avals in
  ignore (Shape_poly.solve_dim_vars s.sym_in_specs shapes);
  match Hashtbl.find_opt s.sym_cache shapes with
  | Some e -> e
  | None ->
      let concrete =
        List.map (fun (a : aval) -> { a with weak_type = false }) avals
      in
      let e =
        export_internal ~fun_name:s.sym_fun_name ~platforms:s.sym_platforms
          ~disabled_checks:s.sym_disabled
          ~in_tree:(args_tree_of_avals concrete)
          ~out_tree:None s.sym_fun concrete
      in
      Hashtbl.replace s.sym_cache shapes e;
      e

let call_symbolic s args = call (specialize s args) args
