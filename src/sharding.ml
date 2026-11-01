open Types
module Tree = Tree_util
module PB = Pjrt.Buffer
include Spmd

type manual_axis_type = {
  varying : string list;
  unreduced : string list;
  reduced : string list;
}

let manual_axis_type ?(varying = []) ?(unreduced = []) ?(reduced = []) () =
  let names l =
    "frozenset({"
    ^ String.concat ", " (List.map (fun a -> "'" ^ a ^ "'") l)
    ^ "})"
  in
  let common a b = List.filter (fun x -> List.mem x b) a in
  if common varying unreduced <> [] then
    invalid_arg
      (Printf.sprintf
         "varying and unreduced cannot have common mesh axes. Got varying=%s \
          and unreduced=%s"
         (names varying) (names unreduced));
  if common varying reduced <> [] then
    invalid_arg
      (Printf.sprintf
         "varying and reduced cannot have common mesh axes. Got varying=%s and \
          reduced=%s"
         (names varying) (names reduced));
  let sorted = List.sort_uniq compare in
  {
    varying = sorted varying;
    unreduced = sorted unreduced;
    reduced = sorted reduced;
  }

let manual_axis_to_ct t =
  { t with unreduced = t.reduced; reduced = t.unreduced }

let single_device = Single_device None
let make_single_device_sharding d = Single_device (Some d)

let abstract_device ?num_cores ~device_kind ~platform () =
  { device_kind; num_cores; platform }

let abstract_mesh ?axis_types ?abstract_device axis_sizes axis_names =
  if List.length axis_sizes <> List.length axis_names then
    invalid_arg "AbstractMesh: axis sizes and names must have the same length";
  {
    abstract_axis_names = axis_names;
    abstract_axis_sizes = axis_sizes;
    abstract_axis_types =
      (match axis_types with
      | Some t -> t
      | None -> List.map (fun _ -> Auto) axis_names);
    abstract_device;
  }

let outside_trace name =
  if (Core.find_top_trace []).kind <> KEval then
    invalid_arg
      (Printf.sprintf "`%s` can only be used outside of `jax.jit`." name)

let get_mesh () =
  outside_trace "get_mesh";
  Spmd.get_mesh ()

let set_mesh m =
  outside_trace "set_mesh";
  if List.mem Manual m.axis_types then
    invalid_arg
      ("mesh " ^ mesh_to_string m
     ^ " contains manual axes which is not allowed when using `jax.set_mesh`. \
        Please use `jax.shard_map` to enter into `Manual` mode instead.");
  concrete_context := Some m;
  abstract_context := Some (abstract_mesh_of m)

let with_mesh m thunk =
  let saved_concrete = !concrete_context in
  let saved_abstract = !abstract_context in
  Fun.protect
    ~finally:(fun () ->
      concrete_context := saved_concrete;
      abstract_context := saved_abstract)
    (fun () ->
      set_mesh m;
      thunk ())

let use_abstract_mesh a thunk =
  (match !abstract_context with
  | Some prev
    when prev.abstract_axis_names <> []
         && a.abstract_axis_names <> []
         && abstract_mesh_size prev <> abstract_mesh_size a ->
      invalid_arg
        (Printf.sprintf
           "use_abstract_mesh cannot change the size of the mesh. Got new \
            mesh: %s with size=%d and prev mesh: %s with size=%d"
           (abstract_mesh_to_string a)
           (abstract_mesh_size a)
           (abstract_mesh_to_string prev)
           (abstract_mesh_size prev))
  | _ -> ());
  with_abstract_context a thunk

let of_spec spec =
  match !concrete_context with
  | Some m when m.axis_names <> [] -> named_sharding m spec
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Using PartitionSpec when you are not under a mesh context is not \
            allowed. Please pass a NamedSharding instance or enter into a mesh \
            context via `jax.set_mesh`. Got %s"
           (spec_to_string spec))

let reshard = device_put
let with_sharding_constraint = device_put
let reshard_spec x spec = device_put x (of_spec spec)
let shard_map = Spmd.shard_map
let smap = Spmd.smap
let leaves_of_args args = fst (Tree.tree_flatten (Tree.Tuple args))

let broadcast_shardings what shardings n =
  match shardings with
  | [ one ] -> List.init n (fun _ -> one)
  | many when List.length many = n -> many
  | many ->
      invalid_arg
        (Printf.sprintf "%s has %d entries for %d leaves" what
           (List.length many) n)

let check_in_sharding sharding v =
  match v with
  | Device (PB.Placed _) when not (same_placement v sharding) ->
      invalid_arg
        (Printf.sprintf
           "Sharding passed to jit does not match the sharding on the \
            respective arg. Got jit sharding: %s,\n\
            arg sharding: %s"
           (sharding_to_string sharding)
           (sharding_to_string (array_sharding v)))
  | _ -> ()

let mesh_of_shardings shardings =
  let meshes =
    List.filter_map
      (function Named_sharding (m, _) -> Some m | _ -> None)
      shardings
  in
  match meshes with
  | [] -> None
  | m :: rest ->
      List.iter
        (fun other ->
          if
            mesh_pairs other <> mesh_pairs m || device_ids other <> device_ids m
          then invalid_arg "Shardings passed to jit should have the same mesh")
        rest;
      Some m

let jit ?in_shardings ?out_shardings f =
  let cache = Hashtbl.create 8 in
  let plain = Api.jit f in
  fun (args : value Tree.t list) ->
    let flat, in_tree = Tree.tree_flatten (Tree.Tuple args) in
    let n = List.length flat in
    let in_list =
      Option.map (fun s -> broadcast_shardings "in_shardings" s n) in_shardings
    in
    Option.iter (fun l -> List.iter2 check_in_sharding l flat) in_list;
    let has_tracer =
      List.exists (function Tracer _ -> true | _ -> false) flat
    in
    let shardings =
      Option.value ~default:[] in_list @ Option.value ~default:[] out_shardings
    in
    let mesh = mesh_of_shardings shardings in
    let finish out =
      match out_shardings with
      | None -> out
      | Some specs ->
          let leaves, tree = Tree.tree_flatten out in
          let specs =
            broadcast_shardings "out_shardings" specs (List.length leaves)
          in
          Tree.tree_unflatten tree
            (List.map2
               (fun v s ->
                 match Spmd.device_put (Tree.Leaf v) s with
                 | Tree.Leaf placed -> placed
                 | _ -> v)
               leaves specs)
    in
    let placed_args () =
      match in_list with
      | None -> args
      | Some list -> (
          let placed =
            List.map2
              (fun v s ->
                match Spmd.device_put (Tree.Leaf v) s with
                | Tree.Leaf placed -> placed
                | _ -> v)
              flat list
          in
          match Tree.tree_unflatten in_tree placed with
          | Tree.Tuple a -> a
          | _ -> args)
    in
    match mesh with
    | Some m
      when mesh_size m > 1
           && (not has_tracer)
           && Backend.multi_device_available () -> (
        let avals = List.map Core.get_aval flat in
        let key = avals in
        let cj, out_tree_get =
          match Hashtbl.find_opt cache key with
          | Some entry -> entry
          | None ->
              let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
              let cj =
                Jaxpr.make_jaxpr avals (Linear_util.call_wrapped flat_fun)
              in
              let tree = out_tree () in
              Hashtbl.replace cache key (cj, tree);
              (cj, tree)
        in
        let arg_specs =
          match in_list with
          | Some list ->
              List.map2
                (fun s v ->
                  let rank = Array.length (Core.get_aval v).shape in
                  match s with
                  | Named_sharding (_, spec) -> spec_array spec rank
                  | _ -> Array.make rank [])
                list flat
          | None ->
              List.map
                (function
                  | Device (PB.Placed { layout = PB.Named nm; _ })
                    when nm.mesh_axes = mesh_pairs m
                         && nm.mesh_devices = device_ids m ->
                      nm.spec
                  | v -> Array.make (Array.length (Core.get_aval v).shape) [])
                flat
        in
        let out_specs =
          match out_shardings with
          | None -> []
          | Some specs ->
              let outs = cj.jaxpr.outs in
              List.map2
                (fun s atom ->
                  let rank =
                    match atom with
                    | A_var v -> Array.length v.vaval.shape
                    | A_lit nd -> Array.length (Ndarray.shape nd)
                    | DropVar a -> Array.length a.shape
                  in
                  match s with
                  | Named_sharding (_, spec) -> Some (spec_array spec rank)
                  | _ -> Some (Array.make rank []))
                (broadcast_shardings "out_shardings" specs (List.length outs))
                outs
        in
        let plan =
          {
            Backend.mesh = mesh_pairs m;
            devices = device_ids m;
            arg_specs;
            out_specs;
          }
        in
        match Backend.run_plan cj plan flat with
        | Some outs -> Tree.tree_unflatten out_tree_get outs
        | None -> finish (plain (placed_args ())))
    | _ -> finish (plain (placed_args ()))

let abstract_of_context name shardings =
  let current = get_abstract_mesh () in
  let from_sharding =
    List.filter_map
      (function
        | Named_sharding (m, _) -> Some (abstract_mesh_of m) | _ -> None)
      shardings
  in
  match (from_sharding, current.abstract_axis_names) with
  | [], [] -> None
  | [], _ -> Some current
  | s :: _, [] -> Some s
  | s :: _, _ ->
      if s <> current then
        invalid_arg
          (Printf.sprintf "Context mesh %s must match the mesh passed to %s"
             (abstract_mesh_to_string current)
             name);
      Some current

let spec_without_axes abstract spec =
  let keep a =
    match
      List.assoc_opt a
        (List.combine abstract.abstract_axis_names abstract.abstract_axis_types)
    with
    | Some Explicit -> true
    | _ -> false
  in
  List.map
    (fun entry ->
      match
        List.filter keep
          (match entry with
          | Named a -> [ a ]
          | Axes l -> l
          | Unconstrained -> [])
      with
      | [] -> Unconstrained
      | [ a ] -> Named a
      | l -> Axes l)
    spec

let reshard_modified abstract x =
  match x with
  | Tracer _ -> x
  | Device (PB.Placed { layout = PB.Named n; _ }) -> (
      match
        Spmd.device_put (Tree.Leaf x)
          (Named_sharding
             ( Spmd.make_mesh
                 ~devices:
                   (Array.of_list
                      (List.map
                         (fun id ->
                           List.find
                             (fun (d : device) -> d.id = id)
                             (Array.to_list (local_devices ())))
                         (Array.to_list n.mesh_devices)))
                 (List.map snd n.mesh_axes) (List.map fst n.mesh_axes),
               spec_without_axes abstract (spec_of_array n.spec) ))
      with
      | Tree.Leaf placed -> placed
      | _ -> x)
  | _ -> x

let lower_text ?out_shardings ~name f args =
  let flat, in_tree = Tree.tree_flatten (Tree.Tuple args) in
  let flat_fun, _ = Api_util.flatten_fun f in_tree in
  let cj =
    Jaxpr.make_jaxpr
      (List.map Core.get_aval flat)
      (Linear_util.call_wrapped flat_fun)
  in
  let out_specs =
    Option.map
      (List.map2
         (fun atom s ->
           let rank =
             match atom with
             | A_var v -> Array.length v.vaval.shape
             | A_lit nd -> Array.length (Ndarray.shape nd)
             | DropVar a -> Array.length a.shape
           in
           match s with
           | Named_sharding (_, spec) -> Some (spec_array spec rank)
           | _ -> None)
         cj.jaxpr.outs)
      out_shardings
  in
  Backend.placed_text ~name ?out_specs cj flat

let auto_axes ?axes ~out_sharding f args =
  match abstract_of_context "auto_axes" out_sharding with
  | None -> f args
  | Some prev ->
      let axes = Option.value ~default:prev.abstract_axis_names axes in
      let types =
        List.combine prev.abstract_axis_names prev.abstract_axis_types
      in
      List.iter
        (fun a ->
          if List.assoc_opt a types = Some Manual then
            invalid_arg
              "Going from `Manual` AxisType to `Auto` or `Explicit` AxisType \
               is not allowed.")
        axes;
      let current_auto =
        List.filter_map (fun (a, t) -> if t = Auto then Some a else None) types
      in
      if List.sort compare current_auto = List.sort compare axes then f args
      else
        let next =
          update_axis_types prev (List.map (fun a -> (a, Auto)) axes)
        in
        let out =
          with_abstract_context next (fun () ->
              let args =
                List.map (Tree.tree_map (reshard_modified next)) args
              in
              f args)
        in
        let leaves, tree = Tree.tree_flatten out in
        let targets =
          broadcast_shardings "out_sharding" out_sharding (List.length leaves)
        in
        Tree.tree_unflatten tree
          (List.map2
             (fun v s ->
               match Spmd.device_put (Tree.Leaf v) s with
               | Tree.Leaf p -> p
               | _ -> v)
             leaves targets)

let explicit_axes ?axes ~in_sharding f args =
  match abstract_of_context "explicit_axes" [] with
  | None ->
      invalid_arg
        "Context mesh cannot be empty. Please use `jax.set_mesh` API to enter \
         into a mesh context when using `explicit_axes` API."
  | Some prev ->
      let axes = Option.value ~default:prev.abstract_axis_names axes in
      let types =
        List.combine prev.abstract_axis_names prev.abstract_axis_types
      in
      List.iter
        (fun a ->
          if List.assoc_opt a types = Some Manual then
            invalid_arg
              "Going from `Manual` AxisType to `Auto` or `Explicit` AxisType \
               is not allowed.")
        axes;
      let next =
        update_axis_types prev (List.map (fun a -> (a, Explicit)) axes)
      in
      let out =
        with_abstract_context next (fun () ->
            let flat, tree = Tree.tree_flatten (Tree.Tuple args) in
            let targets =
              broadcast_shardings "in_sharding" in_sharding (List.length flat)
            in
            let placed =
              List.map2
                (fun v s ->
                  match Spmd.device_put (Tree.Leaf v) s with
                  | Tree.Leaf p -> p
                  | _ -> v)
                flat targets
            in
            match Tree.tree_unflatten tree placed with
            | Tree.Tuple a -> f a
            | _ -> invalid_arg "explicit_axes: argument tree")
      in
      Tree.tree_map (reshard_modified next) out

let make_array_from_single_device_arrays shape sharding arrays =
  let expected = devices_indices_map sharding shape in
  let count = List.length expected in
  if List.length arrays <> count then
    invalid_arg
      (Printf.sprintf
         "Expected %d per-device arrays (this is how many devices are \
          addressable by the sharding), but got %d"
         count (List.length arrays));
  let want_shape = shard_shape sharding shape in
  let describe s =
    "("
    ^ String.concat ", " (Array.to_list (Array.map string_of_int s))
    ^ if Array.length s = 1 then ",)" else ")"
  in
  let raw_of v =
    match v with
    | Device (PB.Placed { shards = [| s |]; _ }) -> s.data
    | Device (PB.Single raw) -> raw
    | _ ->
        invalid_arg
          "make_array_from_single_device_arrays needs arrays that are each on \
           a single device"
  in
  let raws = List.map raw_of arrays in
  List.iter
    (fun raw ->
      let got = PB.raw_dimensions raw in
      if got <> want_shape then
        invalid_arg
          (Printf.sprintf
             "Expected shard shape %s doesn't match the single device array \
              shape %s."
             (describe want_shape) (describe got)))
    raws;
  let dtype = PB.element_type (PB.Single (List.hd raws)) in
  let devices = Array.of_list (List.map fst expected) in
  let indices = Array.of_list (List.map snd expected) in
  let raws =
    Array.map
      (fun (d : device) ->
        match List.find_opt (fun r -> PB.raw_device r = d.id) raws with
        | Some r -> r
        | None ->
            invalid_arg
              (Printf.sprintf
                 "Addressable devices and per-device arrays devices do not \
                  match. Device %d has no array."
                 d.id))
      devices
  in
  let ids = Array.map (fun (d : device) -> d.id) devices in
  let layout =
    match sharding with
    | Named_sharding (m, spec) ->
        PB.Named
          {
            mesh_axes = mesh_pairs m;
            mesh_devices = device_ids m;
            spec = spec_array spec (Array.length shape);
          }
    | Single_device _ -> PB.On_device ids.(0)
    | Tiled_sharding _ -> PB.Tiled
  in
  Backend.placed_of ~shape ~dtype ~devices:ids ~indices ~layout raws

let make_array_from_process_local_data ?global_shape sharding local =
  let nd = host_array local in
  let local_shape = Ndarray.shape nd in
  (match global_shape with
  | Some g when g <> local_shape ->
      let describe s =
        "("
        ^ String.concat ", " (Array.to_list (Array.map string_of_int s))
        ^ if Array.length s = 1 then ",)" else ")"
      in
      invalid_arg
        (Printf.sprintf
           "When calling `make_array_from_process_local_data` on a single \
            process, global_shape should be None or equal to \
            local_data.shape.Got global_shape=%s and local_data.shape=%s."
           (describe g) (describe local_shape))
  | _ -> ());
  match Spmd.device_put (Tree.Leaf (Concrete nd)) sharding with
  | Tree.Leaf placed -> placed
  | _ -> invalid_arg "make_array_from_process_local_data"

let make_array_from_callback shape sharding callback =
  if not (Backend.multi_device_available ()) then
    invalid_arg "make_array_from_callback needs the PJRT plugin";
  let entries = devices_indices_map sharding shape in
  let client = Backend.client () in
  let raws =
    List.map
      (fun ((d : device), index) ->
        let nd = host_array (callback index) in
        let whole = Array.map (fun e -> (0, e)) (Ndarray.shape nd) in
        (PB.shards_of_host client nd [| (whole, d.id) |]).(0))
      entries
  in
  make_array_from_single_device_arrays shape sharding
    (List.map single_shard_value raws)
