open Types
module Tree = Tree_util
module B = Interpreters.Batching
module Ad = Interpreters.Ad
module L = Lax_api
module PB = Pjrt.Buffer

type device = Pjrt.Client.device = {
  id : int;
  process_index : int;
  kind : string;
}

type axis_type = Auto | Explicit | Manual

type abstract_device = {
  device_kind : string;
  num_cores : int option;
  platform : string;
}

type mesh = {
  axis_names : string list;
  axis_sizes : int list;
  axis_types : axis_type list;
  devices : device array;
}

type abstract_mesh = {
  abstract_axis_names : string list;
  abstract_axis_sizes : int list;
  abstract_axis_types : axis_type list;
  abstract_device : abstract_device option;
}

type axis_spec = Named of string | Axes of string list | Unconstrained
type partition_spec = axis_spec list

type t =
  | Single_device of device option
  | Named_sharding of mesh * partition_spec
  | Tiled_sharding of {
      devices : device array;
      indices : (int * int) array array;
    }

type shard = {
  device : device;
  index : (int * int) array;
  replica_id : int;
  data : value;
}

let local_devices = Backend.local_devices
let device_count () = Array.length (local_devices ())
let product = List.fold_left ( * ) 1
let mesh_size m = product m.axis_sizes
let mesh_pairs m = List.combine m.axis_names m.axis_sizes
let device_ids m = Array.map (fun d -> d.id) m.devices

let known_axis_types :
    ((string * int) list * int array, axis_type list) Hashtbl.t =
  Hashtbl.create 8

let make_mesh ?axis_types ?devices axis_shapes axis_names =
  let available = match devices with Some d -> d | None -> local_devices () in
  if List.length axis_shapes <> List.length axis_names then
    invalid_arg
      (Printf.sprintf
         "Mesh requires the ndim of its first argument (`devices`) to equal \
          the length of its second argument (`axis_names`), but got \
          devices.ndim == %d and len(axis_names) == %d."
         (List.length axis_shapes) (List.length axis_names));
  let size = product axis_shapes in
  if size > Array.length available then
    invalid_arg
      (Printf.sprintf
         "Number of devices %d must be >= the product of mesh_shape (%s)"
         (Array.length available)
         (String.concat ", " (List.map string_of_int axis_shapes)));
  let m =
    {
      axis_names;
      axis_sizes = axis_shapes;
      axis_types =
        (match axis_types with
        | Some types -> types
        | None -> List.map (fun _ -> Explicit) axis_names);
      devices = Array.sub available 0 size;
    }
  in
  Hashtbl.replace known_axis_types (mesh_pairs m, device_ids m) m.axis_types;
  m

let axes_of = function Named a -> [ a ] | Axes l -> l | Unconstrained -> []
let spec_names spec = List.concat_map axes_of spec
let quote s = "'" ^ s ^ "'"

let spec_to_string spec =
  let entry = function
    | Unconstrained -> "None"
    | Named a -> quote a
    | Axes l -> "(" ^ String.concat ", " (List.map quote l) ^ ")"
  in
  match spec with
  | [] -> "P()"
  | [ one ] -> "P(" ^ entry one ^ ",)"
  | many -> "P(" ^ String.concat ", " (List.map entry many) ^ ")"

let named_sharding m spec =
  List.iter
    (fun name ->
      if not (List.mem name m.axis_names) then
        invalid_arg
          (Printf.sprintf "Resource axis: %s of %s is not found in mesh: (%s)."
             name (spec_to_string spec)
             (String.concat ", " (List.map quote m.axis_names)
             ^ if List.length m.axis_names = 1 then "," else "")))
    (spec_names spec);
  let rec duplicates = function
    | [] -> ()
    | a :: rest ->
        if List.mem a rest then
          invalid_arg
            (Printf.sprintf
               "A single NamedSharding spec specification can map every mesh \
                axis to at most one positional dimension, but %s has duplicate \
                entries for `%s`"
               (spec_to_string spec) a);
        duplicates rest
  in
  duplicates (spec_names spec);
  Named_sharding (m, spec)

let spec_array spec rank =
  if List.length spec > rank then
    invalid_arg
      (Printf.sprintf
         "Sharding %s is only valid for values of rank at least %d, but got a \
          value of rank %d"
         (spec_to_string spec) (List.length spec) rank);
  Array.init rank (fun d ->
      match List.nth_opt spec d with Some s -> axes_of s | None -> [])

let spec_of_array spec =
  let entries =
    Array.to_list
      (Array.map
         (function [] -> Unconstrained | [ a ] -> Named a | l -> Axes l)
         spec)
  in
  let rec strip = function
    | Unconstrained :: rest -> strip rest
    | kept -> kept
  in
  List.rev (strip (List.rev entries))

let default_device () = (local_devices ()).(0)

let device_of_id id =
  match
    List.find_opt (fun d -> d.id = id) (Array.to_list (local_devices ()))
  with
  | Some d -> d
  | None -> invalid_arg (Printf.sprintf "unknown device id %d" id)

let mesh_of_named (n : PB.named) =
  {
    axis_names = List.map fst n.mesh_axes;
    axis_sizes = List.map snd n.mesh_axes;
    axis_types =
      (match
         Hashtbl.find_opt known_axis_types (n.mesh_axes, n.mesh_devices)
       with
      | Some types -> types
      | None -> List.map (fun _ -> Explicit) n.mesh_axes);
    devices = Array.map device_of_id n.mesh_devices;
  }

let array_sharding v =
  match v with
  | Device (PB.Placed { layout = PB.Named n; _ }) ->
      Named_sharding (mesh_of_named n, spec_of_array n.spec)
  | Device (PB.Placed { layout = PB.On_device id; _ }) ->
      Single_device (Some (device_of_id id))
  | Device (PB.Placed { layout = PB.Tiled; shards; _ }) ->
      Tiled_sharding
        {
          devices =
            Array.map (fun (s : PB.shard) -> device_of_id s.device) shards;
          indices = Array.map (fun (s : PB.shard) -> s.index) shards;
        }
  | Device (PB.Single _) | Concrete _ ->
      Single_device (Some (default_device ()))
  | Tracer _ -> invalid_arg "array_sharding: expected a concrete array"

let sharding_devices = function
  | Single_device (Some d) -> [| d |]
  | Single_device None -> [| default_device () |]
  | Named_sharding (m, _) -> m.devices
  | Tiled_sharding { devices; _ } -> devices

let num_devices s = Array.length (sharding_devices s)
let is_fully_addressable _ = true

let is_fully_replicated = function
  | Single_device _ -> true
  | Named_sharding (m, spec) ->
      let pairs = mesh_pairs m in
      List.for_all (fun a -> List.assoc a pairs = 1) (spec_names spec)
  | Tiled_sharding { indices; _ } ->
      Array.for_all (fun i -> i = indices.(0)) indices

let devices_indices_map s shape =
  match s with
  | Single_device _ ->
      Array.to_list
        (Array.map
           (fun d -> (d, Array.map (fun e -> (0, e)) shape))
           (sharding_devices s))
  | Named_sharding (m, spec) ->
      let spec = spec_array spec (Array.length shape) in
      let mesh = mesh_pairs m in
      Backend.check_divisible ~shape ~mesh ~spec;
      Array.to_list
        (Array.mapi
           (fun pos d -> (d, Backend.named_index ~shape ~mesh ~spec pos))
           m.devices)
  | Tiled_sharding { devices; indices } ->
      Array.to_list (Array.mapi (fun i d -> (d, indices.(i))) devices)

let shard_shape s shape =
  match devices_indices_map s shape with
  | (_, index) :: _ -> Array.map (fun (lo, hi) -> hi - lo) index
  | [] -> shape

let single_shard_value raw =
  let dtype = PB.element_type (PB.Single raw) in
  let shape = PB.raw_dimensions raw in
  Backend.placed_of ~shape ~dtype
    ~devices:[| PB.raw_device raw |]
    ~indices:[| Array.map (fun e -> (0, e)) shape |]
    ~layout:(PB.On_device (PB.raw_device raw))
    [| raw |]

let addressable_shards v =
  match v with
  | Device (PB.Placed p) ->
      Array.to_list
        (Array.map
           (fun (s : PB.shard) ->
             {
               device = device_of_id s.device;
               index = s.index;
               replica_id = s.replica_id;
               data = single_shard_value s.data;
             })
           p.shards)
  | Device (PB.Single _) | Concrete _ ->
      let shape = (Core.get_aval v).shape in
      [
        {
          device = default_device ();
          index = Array.map (fun e -> (0, e)) shape;
          replica_id = 0;
          data = v;
        };
      ]
  | Tracer _ -> invalid_arg "addressable_shards: expected a concrete array"

let global_shards = addressable_shards
let addressable_data v i = (List.nth (addressable_shards v) i).data

let is_committed = function
  | Device (PB.Placed _) -> true
  | Device (PB.Single _) | Concrete _ | Tracer _ -> false

let host_array = function
  | Concrete nd -> nd
  | Device b -> PB.to_host b
  | Tracer _ -> invalid_arg "expected a concrete array"

let same_placement v sharding =
  match (v, sharding) with
  | Device (PB.Placed { layout = PB.Named n; _ }), Named_sharding (m, spec) ->
      n.mesh_axes = mesh_pairs m
      && n.mesh_devices = device_ids m
      && n.spec = spec_array spec (Array.length n.spec)
  | Device (PB.Placed { layout = PB.On_device id; _ }), Single_device (Some d)
    ->
      id = d.id
  | _ -> false

let identity_rule () =
  Core.register_custom_rule
    (Core.Jvp_rule
       {
         primal = (fun xs -> xs);
         jvp = (fun primals tangents -> (primals, tangents));
       })

let axis_type_to_string = function
  | Auto -> "Auto"
  | Explicit -> "Explicit"
  | Manual -> "Manual"

let axis_types_string types =
  match types with
  | [ one ] -> "(" ^ axis_type_to_string one ^ ",)"
  | many -> "(" ^ String.concat ", " (List.map axis_type_to_string many) ^ ")"

let mesh_to_string m =
  if m.axis_names = [] then "Mesh()"
  else
    Printf.sprintf "Mesh(%s, axis_types=%s)"
      (String.concat ", "
         (List.map2
            (fun n s -> Printf.sprintf "'%s': %d" n s)
            m.axis_names m.axis_sizes))
      (axis_types_string m.axis_types)

let py_specs specs =
  match specs with
  | [ one ] -> "(" ^ spec_to_string one ^ ",)"
  | many -> "(" ^ String.concat ", " (List.map spec_to_string many) ^ ")"

let py_frozenset names =
  match names with
  | [] -> "frozenset({})"
  | l -> "frozenset({" ^ String.concat ", " (List.map quote l) ^ "})"

let constrained v m spec =
  if mesh_size m < 1 then v
  else begin
    let spec_list = spec in
    let rank = Array.length (Core.get_aval v).shape in
    let spec = spec_array spec rank in
    Backend.check_divisible ~shape:(Core.get_aval v).shape ~mesh:(mesh_pairs m)
      ~spec;
    let id = identity_rule () in
    Core.register_constraint_rule id
      { cr_mesh = mesh_pairs m; cr_devices = device_ids m; cr_spec = spec };
    Pretty_printer.register_custom_view id "sharding_constraint"
      [
        ("context_mesh", Plain "Mesh()");
        ("layout", Plain "None");
        ( "sharding",
          Plain
            (Printf.sprintf
               "NamedSharding(mesh=%s, spec=%s, memory_kind=device)"
               (mesh_to_string m) (spec_to_string spec_list)) );
        ("unconstrained_dims", Plain "frozenset({})");
      ];
    Core.bind1 (Custom_jvp_call id) [ v ]
  end

let put_leaf sharding v =
  match (v, sharding) with
  | Tracer _, Named_sharding (m, spec) -> constrained v m spec
  | Tracer _, _ -> v
  | _ when same_placement v sharding -> v
  | _, Single_device None -> Backend.of_host_value v
  | _, _ when not (Backend.multi_device_available ()) -> (
      match sharding with
      | Named_sharding (m, _) when mesh_size m > device_count () ->
          invalid_arg
            (Printf.sprintf
               "Mesh of %d devices needs more than the %d available"
               (mesh_size m) (device_count ()))
      | _ -> Backend.of_host_value v)
  | _, Single_device (Some d) -> Backend.place_on (host_array v) d.id
  | _, Named_sharding (m, spec)
    when mesh_size m = 1
         && List.length spec > Array.length (Core.get_aval v).shape ->
      Backend.of_host_value v
  | _, Named_sharding (m, spec) ->
      let nd = host_array v in
      Backend.place_named nd ~mesh:(mesh_pairs m) ~devices:(device_ids m)
        ~spec:(spec_array spec (Array.length (Ndarray.shape nd)))
  | _, Tiled_sharding _ ->
      invalid_arg "device_put: an XLA-chosen sharding cannot be a target"

let device_put x sharding = Tree.tree_map (put_leaf sharding) x
let concrete_context : mesh option ref = ref None
let abstract_context : abstract_mesh option ref = ref None

let abstract_mesh_of m =
  let kind = if Array.length m.devices > 0 then m.devices.(0).kind else "cpu" in
  {
    abstract_axis_names = m.axis_names;
    abstract_axis_sizes = m.axis_sizes;
    abstract_axis_types = m.axis_types;
    abstract_device =
      Some
        { device_kind = kind; num_cores = None; platform = Backend.platform () };
  }

let empty_abstract_mesh =
  {
    abstract_axis_names = [];
    abstract_axis_sizes = [];
    abstract_axis_types = [];
    abstract_device = None;
  }

let empty_mesh =
  { axis_names = []; axis_sizes = []; axis_types = []; devices = [||] }

let abstract_mesh_size a = product a.abstract_axis_sizes
let get_mesh () = Option.value ~default:empty_mesh !concrete_context

let get_abstract_mesh () =
  Option.value ~default:empty_abstract_mesh !abstract_context

let with_abstract_context a thunk =
  let saved = !abstract_context in
  abstract_context := Some a;
  Fun.protect ~finally:(fun () -> abstract_context := saved) thunk

let update_axis_types a updates =
  {
    a with
    abstract_axis_types =
      List.map2
        (fun name ty ->
          match List.assoc_opt name updates with Some t -> t | None -> ty)
        a.abstract_axis_names a.abstract_axis_types;
  }

let context_mesh args =
  match (!concrete_context, !abstract_context) with
  | Some m, _ when m.axis_names <> [] -> m
  | _, Some a when a.abstract_axis_names <> [] -> (
      let pairs = List.combine a.abstract_axis_names a.abstract_axis_sizes in
      let from_arg =
        List.find_map
          (function
            | Device (PB.Placed { layout = PB.Named n; _ })
              when n.mesh_axes = pairs ->
                Some (mesh_of_named n)
            | _ -> None)
          args
      in
      match from_arg with
      | Some m -> m
      | None ->
          invalid_arg
            "Please pass `jax.Array`s with a `NamedSharding` as input to \
             `shard_map` when passing `AbstractMesh` to the mesh argument.")
  | _ ->
      invalid_arg
        "The context mesh cannot be empty. Use `jax.set_mesh(mesh)` to enter \
         into a mesh context"

let split_dim v d sizes =
  let shape = (Core.get_aval v).shape in
  let target =
    Array.concat
      [
        Array.sub shape 0 d;
        Array.of_list sizes;
        Array.sub shape (d + 1) (Array.length shape - d - 1);
      ]
  in
  L.reshape v target

let merge_dims v d count =
  let shape = (Core.get_aval v).shape in
  let middle = Array.fold_left ( * ) 1 (Array.sub shape d count) in
  let target =
    Array.concat
      [
        Array.sub shape 0 d;
        [| middle |];
        Array.sub shape (d + count) (Array.length shape - d - count);
      ]
  in
  L.reshape v target

let locate axis spec =
  let found = ref None in
  Array.iteri
    (fun d axes ->
      if List.mem axis axes then
        let rec before acc = function
          | a :: _ when a = axis -> List.rev acc
          | a :: rest -> before (a :: acc) rest
          | [] -> List.rev acc
        in
        found := Some (d, before [] axes))
    spec;
  !found

let without done_axes spec =
  Array.map (List.filter (fun a -> not (List.mem a done_axes))) spec

let move_first_to v d =
  let rank = Array.length (Core.get_aval v).shape in
  if d = 0 then v
  else
    let perm =
      Array.init rank (fun i -> if i < d then i + 1 else if i = d then 0 else i)
    in
    L.transpose v perm

let rec map_axes pairs done_axes remaining in_specs out_spec_of f args =
  match remaining with
  | [] -> f args
  | axis :: rest ->
      let size = List.assoc axis pairs in
      let sizes_of axes =
        product (List.map (fun a -> List.assoc a pairs) axes)
      in
      let plan =
        List.map2
          (fun x spec ->
            match locate axis (without done_axes spec) with
            | None -> (x, None)
            | Some (d, before) ->
                let pb = sizes_of before in
                let extent = (Core.get_aval x).shape.(d) in
                ( split_dim x d [ pb; size; extent / (pb * size) ],
                  Some (d, d + 1) ))
          args in_specs
      in
      let prepared = List.map fst plan in
      let in_axes = List.map (fun (_, a) -> Option.map snd a) plan in
      let dims = List.map (fun (_, a) -> Option.map fst a) plan in
      let placements = ref [] in
      let inner xs =
        let xs = List.filteri (fun i _ -> i < List.length dims) xs in
        let merged =
          List.map2
            (fun x d -> match d with Some d -> merge_dims x d 2 | None -> x)
            xs dims
        in
        let ys =
          map_axes pairs (axis :: done_axes) rest in_specs out_spec_of f merged
        in
        let located =
          List.mapi
            (fun k y ->
              let spec = out_spec_of k (Array.length (Core.get_aval y).shape) in
              match locate axis (without done_axes spec) with
              | None -> (y, None)
              | Some (d, before) ->
                  let pb = sizes_of before in
                  let extent = (Core.get_aval y).shape.(d) in
                  (split_dim y d [ pb; extent / pb ], Some d))
            ys
        in
        placements := List.map snd located;
        List.map fst located
      in
      let counter =
        Concrete (Ndarray.of_floats Dtype.F32 [| size |] (Array.make size 0.0))
      in
      let stacked =
        B.vmap_flat ~axis_name:axis ~partial_permute:true inner
          (in_axes @ [ Some 0 ]) (prepared @ [ counter ])
      in
      List.map2
        (fun y placement ->
          match placement with
          | None -> L.index_in_dim ~axis:0 ~keepdims:false y 0
          | Some d -> merge_dims (move_first_to y (d + 1)) d 3)
        stacked !placements

let broadcast_specs specs n what =
  match specs with
  | [ one ] -> List.init n (fun _ -> one)
  | many when List.length many = n -> many
  | many ->
      invalid_arg
        (Printf.sprintf "shard_map %s has %d entries for %d leaves" what
           (List.length many) n)

let local_shape pairs manual (shape : int array) spec =
  Array.mapi
    (fun d extent ->
      let tiles =
        product
          (List.filter_map
             (fun a ->
               if List.mem a manual then Some (List.assoc a pairs) else None)
             spec.(d))
      in
      extent / tiles)
    shape

let check_args pairs specs args =
  List.iter2
    (fun x spec ->
      let shape = (Core.get_aval x).shape in
      Backend.check_divisible ~shape ~mesh:pairs ~spec)
    args specs

let inserts_pvary = function
  | Add | Add_any | Sub | Mul | Div | Rem | Max | Min | Pow | Atan2 | Complex
  | Nextafter | And | Or | Xor | Shift_left | Shift_right_arithmetic
  | Shift_right_logical | Eq | Ne | Ge | Gt | Le | Lt | Mulhi | Clamp
  | Concatenate _ | Dot_general _ | Ragged_dot_general | Pad _ | Select_n
  | Reduce _ | Sort _ | Rng_uniform | Optimization_barrier | Igamma
  | Igamma_grad_a | Igammac | Polygamma | Zeta | Regularized_incomplete_beta ->
      true
  | _ -> false

let atom_aval = function
  | A_var (v : var) -> v.vaval
  | A_lit nd ->
      { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }
  | DropVar a -> a

let note_vma env (v : var) axes =
  Hashtbl.replace env v.vid axes;
  Pretty_printer.note_varying v.vid axes

let pvary_inserted = ref 0

let rec type_vma manual env (jx : jaxpr) wanted_outs =
  let set = note_vma env in
  let vma_of = function
    | A_var (v : var) -> Option.value ~default:[] (Hashtbl.find_opt env v.vid)
    | A_lit _ | DropVar _ -> []
  in
  let eqns = ref [] in
  let cast atom wanted =
    let src = vma_of atom in
    let rest =
      List.filter (fun a -> List.mem a wanted && not (List.mem a src)) manual
    in
    match atom with
    | (A_var _ | A_lit _) when rest <> [] ->
        incr pvary_inserted;
        let v = { vid = Core.fresh_id (); vaval = atom_aval atom } in
        eqns :=
          {
            prim = Pvary rest;
            inputs = [ atom ];
            outs = [ v ];
            multiple_results = false;
          }
          :: !eqns;
        set v (List.filter (fun a -> List.mem a src || List.mem a rest) manual);
        A_var v
    | _ -> atom
  in
  List.iter
    (fun (e : eqn) ->
      let e =
        if inserts_pvary e.prim then
          let wanted = List.concat_map vma_of e.inputs in
          { e with inputs = List.map (fun x -> cast x wanted) e.inputs }
        else e
      in
      match e.prim with
      | Xla_call c ->
          let binders = c.jaxpr.in_binders in
          let skip = List.length binders - List.length e.inputs in
          List.iteri
            (fun i b ->
              set b
                (if i < skip then [] else vma_of (List.nth e.inputs (i - skip))))
            binders;
          let before = !pvary_inserted in
          let inner =
            type_vma manual env c.jaxpr (List.map (fun _ -> []) c.jaxpr.outs)
          in
          List.iter2 (fun v a -> set v (vma_of a)) e.outs inner.outs;
          let c =
            if !pvary_inserted = before then c
            else begin
              let jid = Core.fresh_id () in
              Core.inherit_inline ~from:c.jid jid;
              (match Pretty_printer.registered_call_name c.jid with
              | Some name -> Pretty_printer.register_call_name jid name
              | None -> ());
              { c with jid; jaxpr = inner }
            end
          in
          eqns := { e with prim = Xla_call c } :: !eqns
      | _ ->
          let ins = List.concat_map vma_of e.inputs in
          let axes = List.filter (fun a -> List.mem a ins) manual in
          let axes =
            match e.prim with
            | Collective { kind = Psum | Pmax | Pmin; axis_name } ->
                List.filter (fun a -> a <> axis_name) axes
            | _ -> axes
          in
          List.iter (fun v -> set v axes) e.outs;
          eqns := e :: !eqns)
    jx.eqns;
  let outs = List.map2 cast jx.outs wanted_outs in
  { jx with eqns = List.rev !eqns; outs }

let shard_map_flat f ~mesh ~manual ~in_specs ~out_specs args =
  let pairs = mesh_pairs mesh in
  let in_arrays =
    List.map2
      (fun spec x -> spec_array spec (Array.length (Core.get_aval x).shape))
      in_specs args
  in
  check_args pairs in_arrays args;
  let manual_pairs = List.filter (fun (a, _) -> List.mem a manual) pairs in
  let out_spec_of k rank =
    let n = List.length out_specs in
    let spec = if n = 1 then List.hd out_specs else List.nth out_specs k in
    spec_array spec rank
  in
  let manual_ordered = List.map fst manual_pairs in
  let fallback xs =
    map_axes pairs [] manual_ordered in_arrays out_spec_of f xs
  in
  let manual_context thunk =
    let current = abstract_mesh_of mesh in
    with_abstract_context
      (update_axis_types current (List.map (fun a -> (a, Manual)) manual))
      thunk
  in
  let fallback xs = manual_context (fun () -> fallback xs) in
  if mesh_size mesh < 1 then fallback args
  else
    let avals = List.map Core.get_aval args in
    let local_avals =
      List.map2
        (fun (a : aval) spec ->
          { a with shape = local_shape pairs manual a.shape spec })
        avals in_arrays
    in
    let body =
      match
        manual_context (fun () ->
            Core.with_mesh_axes manual_pairs (fun () ->
                Jaxpr.make_jaxpr_hoisted local_avals f))
      with
      | body, [] -> Some body
      | _, _ :: _ -> None
      | exception (Failure _ | Invalid_argument _) -> None
    in
    let id =
      Core.register_custom_rule
        (Core.Jvp_rule
           {
             primal = fallback;
             jvp = (fun primals tangents -> Ad.jvp fallback primals tangents);
           })
    in
    (match body with
    | Some body ->
        let out_arrays =
          List.mapi
            (fun k atom ->
              let local =
                match atom with
                | A_var v -> v.vaval
                | A_lit nd ->
                    {
                      shape = Ndarray.shape nd;
                      dtype = Ndarray.dtype nd;
                      weak_type = false;
                    }
                | DropVar a -> a
              in
              out_spec_of k (Array.length local.shape))
            body.jaxpr.outs
        in
        let out_list =
          List.mapi
            (fun k _ ->
              if List.length out_specs = 1 then List.hd out_specs
              else List.nth out_specs k)
            body.jaxpr.outs
        in
        let env = Hashtbl.create 16 in
        let n_consts = List.length body.consts in
        List.iteri
          (fun i (v : var) ->
            if i >= n_consts then
              note_vma env v
                (List.filter
                   (fun a ->
                     List.mem a (spec_names (List.nth in_specs (i - n_consts))))
                   manual_ordered))
          body.jaxpr.in_binders;
        let jaxpr =
          type_vma manual_ordered env body.jaxpr
            (List.mapi
               (fun k _ -> spec_names (List.nth out_list k))
               body.jaxpr.outs)
        in
        let body = { body with jid = Core.fresh_id (); jaxpr } in
        Pretty_printer.register_custom_view id "shard_map"
          [
            ("check_vma", Plain "True");
            ("in_specs", Plain (py_specs in_specs));
            ("jaxpr", Sub body);
            ("mesh", Plain (mesh_to_string mesh));
            ( "newly_manual_axes",
              Plain (py_frozenset (List.sort compare manual_ordered)) );
            ("out_specs", Plain (py_specs out_list));
          ];
        Core.register_shard_rule id
          {
            sr_mesh = pairs;
            sr_devices = device_ids mesh;
            sr_manual = manual_ordered;
            sr_in = in_arrays;
            sr_out = out_arrays;
            sr_body = body;
          }
    | None -> ());
    let call xs = Core.bind (Custom_jvp_call id) xs in
    let traced =
      List.exists (function Tracer _ -> true | _ -> false) args
      || (Core.find_top_trace []).kind <> KEval
    in
    if Option.is_none body || not (Backend.multi_device_available ()) then
      fallback args
    else if traced then call args
    else
      let cj = Jaxpr.make_jaxpr avals call in
      let outs = Backend.executor cj args in
      List.mapi
        (fun k out ->
          match out with
          | Concrete nd ->
              let spec = out_spec_of k (Array.length (Ndarray.shape nd)) in
              Backend.place_named nd ~mesh:pairs ~devices:(device_ids mesh)
                ~spec
          | other -> other)
        outs

let flatten_tree_args (args : value Tree.t list) =
  let flat, tree = Tree.tree_flatten (Tree.Tuple args) in
  (flat, tree)

let unpack_args tree flat =
  match Tree.tree_unflatten tree flat with
  | Tree.Tuple args -> args
  | _ -> invalid_arg "shard_map: argument tree"

let infer_in_spec = function
  | Device (PB.Placed { layout = PB.Named n; _ }) -> spec_of_array n.spec
  | _ -> []

let check_manual_refs what manual specs =
  List.iter
    (fun spec ->
      List.iter
        (fun name ->
          if not (List.mem name manual) then
            invalid_arg
              (Printf.sprintf
                 "shard_map %s argument must refer to an axis marked as manual \
                  (%s), but: %s refers to '%s'. Check the %s values passed to \
                  shard_map."
                 what
                 (String.concat ", " manual)
                 what name what))
        (spec_names spec))
    specs

let check_explicit_input mesh manual specs flat =
  let all_axes = List.sort compare mesh.axis_names in
  let explicit =
    List.for_all (fun t -> t = Explicit) mesh.axis_types
    && List.sort compare manual = all_axes
    && mesh_size mesh > 1
  in
  if explicit then
    List.iter2
      (fun spec v ->
        match v with
        | Tracer _ -> ()
        | _ ->
            let rank = Array.length (Core.get_aval v).shape in
            let wanted = spec_array spec rank in
            let actual =
              match v with
              | Device (PB.Placed { layout = PB.Named n; _ })
                when n.mesh_axes = mesh_pairs mesh ->
                  n.spec
              | _ -> Array.make rank []
            in
            if wanted <> actual then
              invalid_arg
                (Printf.sprintf
                   "in_specs passed to shard_map: %s does not match the specs \
                    of the input: %s. `in_specs` is an optional argument so \
                    you can omit specifying it and shard_map will infer the \
                    in_specs from the arguments. If you want to reshard your \
                    inputs, you can use `jax.reshard` on the arguments and \
                    then pass those args to shard_map."
                   (spec_to_string spec)
                   (spec_to_string (spec_of_array actual))))
      specs flat

let shard_map ?mesh ?axis_names ?in_specs ?(check_vma : bool option) ~out_specs
    f args =
  ignore check_vma;
  (match (mesh, !abstract_context) with
  | Some m, Some ctx
    when ctx.abstract_axis_names <> []
         && (ctx.abstract_axis_names <> m.axis_names
            || ctx.abstract_axis_sizes <> m.axis_sizes) ->
      invalid_arg "The context mesh should match the mesh passed to shard_map"
  | _ -> ());
  let flat, tree = flatten_tree_args args in
  let mesh = match mesh with Some m -> m | None -> context_mesh flat in
  let manual =
    match axis_names with Some names -> names | None -> mesh.axis_names
  in
  List.iter
    (fun a ->
      if not (List.mem a mesh.axis_names) then
        invalid_arg
          (Printf.sprintf
             "jax.shard_map requires axis_names=%s to be a subset of \
              mesh.axis_names=%s"
             (String.concat ", " manual)
             (String.concat ", " mesh.axis_names)))
    manual;
  check_manual_refs "out_specs" manual out_specs;
  let in_specs =
    match in_specs with
    | Some specs ->
        check_manual_refs "in_specs" manual specs;
        let specs = broadcast_specs specs (List.length flat) "in_specs" in
        check_explicit_input mesh manual specs flat;
        specs
    | None ->
        if
          List.exists
            (fun a ->
              List.assoc a (List.combine mesh.axis_names mesh.axis_types)
              <> Explicit)
            manual
        then
          invalid_arg
            "shard_map in_specs argument must be a pytree of \
             `jax.sharding.PartitionSpec` instances, but it was `None` when \
             the manual axes are not Explicit";
        List.map infer_in_spec flat
  in
  let out_tree = ref None in
  let flat_f xs =
    let out = f (unpack_args tree xs) in
    let leaves, shape = Tree.tree_flatten out in
    out_tree := Some shape;
    leaves
  in
  let outs = shard_map_flat flat_f ~mesh ~manual ~in_specs ~out_specs flat in
  match !out_tree with
  | Some shape -> Tree.tree_unflatten shape outs
  | None -> invalid_arg "shard_map: function was never traced"

let axis_to_spec name = function
  | None -> []
  | Some axis -> List.init axis (fun _ -> Unconstrained) @ [ Named name ]

let smap ?in_axes ~out_axes ~axis_name f args =
  let in_specs = Option.map (List.map (axis_to_spec axis_name)) in_axes in
  shard_map ~axis_names:[ axis_name ] ?in_specs
    ~out_specs:[ axis_to_spec axis_name out_axes ]
    f args

let pmap_axis = "_internal_pmap_axis_name"

let leaf_axes args in_axes =
  let per_arg =
    match in_axes with
    | Some axes -> axes
    | None -> List.map (fun _ -> Some 0) args
  in
  List.concat
    (List.map2
       (fun axis arg -> List.map (fun _ -> axis) (Tree.tree_leaves arg))
       per_arg args)

let pmap ?axis_name ?in_axes ?(out_axes = 0) ?axis_size f args =
  let flat, tree = flatten_tree_args args in
  let axes = leaf_axes args in_axes in
  if flat = [] then invalid_arg "pmap requires at least one argument";
  let sizes =
    List.filter_map
      (fun (v, axis) ->
        match axis with
        | None -> None
        | Some d ->
            let shape = (Core.get_aval v).shape in
            if d >= Array.length shape then
              invalid_arg
                "pmap got more replicas than the number of available devices";
            Some shape.(d))
      (List.combine flat axes)
  in
  let n =
    match sizes with
    | n :: rest ->
        if List.exists (fun m -> m <> n) rest then
          invalid_arg "pmap got inconsistent sizes for array axes to be mapped";
        n
    | [] -> (
        match axis_size with
        | Some n -> n
        | None -> invalid_arg "pmap requires at least one mapped argument")
  in
  if n > device_count () then
    invalid_arg "pmap got more replicas than the number of available devices";
  let out_tree = ref None in
  let flatten_out out =
    let leaves, shape = Tree.tree_flatten out in
    out_tree := Some shape;
    leaves
  in
  let name = Option.value ~default:pmap_axis axis_name in
  let outs =
    if n <= 1 then
      let single xs = flatten_out (f (unpack_args tree xs)) in
      let raw = B.vmap_flat ?axis_name single axes flat in
      List.map (fun y -> move_first_to y out_axes) raw
    else
      let mesh =
        make_mesh ~axis_types:[ Auto ]
          ~devices:(Array.sub (local_devices ()) 0 n)
          [ n ] [ name ]
      in
      let local xs =
        let squeezed =
          List.map2
            (fun x axis ->
              match axis with
              | Some d -> L.index_in_dim ~axis:d ~keepdims:false x 0
              | None -> x)
            xs axes
        in
        let ys = flatten_out (f (unpack_args tree squeezed)) in
        List.map (fun y -> L.expand_dims y [| out_axes |]) ys
      in
      shard_map_flat local ~mesh ~manual:[ name ]
        ~in_specs:(List.map (axis_to_spec name) axes)
        ~out_specs:[ axis_to_spec name (Some out_axes) ]
        flat
  in
  match !out_tree with
  | Some shape -> Tree.tree_unflatten shape outs
  | None -> invalid_arg "pmap: function was never traced"

let device_to_string d =
  let label = String.capitalize_ascii (Backend.platform ()) in
  Printf.sprintf "%sDevice(id=%d)" label d.id

let abstract_device_fields d =
  Printf.sprintf "device_kind=%s, num_cores=%s, platform=%s" d.device_kind
    (match d.num_cores with Some n -> string_of_int n | None -> "None")
    d.platform

let abstract_device_to_string d =
  "AbstractDevice(" ^ abstract_device_fields d ^ ")"

let abstract_mesh_to_string a =
  if a.abstract_axis_names = [] then "AbstractMesh((), axis_types=())"
  else
    Printf.sprintf "AbstractMesh(%s, axis_types=%s%s)"
      (String.concat ", "
         (List.map2
            (fun n s -> Printf.sprintf "'%s': %d" n s)
            a.abstract_axis_names a.abstract_axis_sizes))
      (axis_types_string a.abstract_axis_types)
      (match a.abstract_device with
      | Some d -> ", " ^ abstract_device_fields d
      | None -> "")

let sharding_to_string = function
  | Single_device d ->
      let d = match d with Some d -> d | None -> default_device () in
      Printf.sprintf "SingleDeviceSharding(device=%s, memory_kind=device)"
        (device_to_string d)
  | Named_sharding (m, spec) ->
      Printf.sprintf "NamedSharding(mesh=%s, spec=%s, memory_kind=device)"
        (mesh_to_string m) (spec_to_string spec)
  | Tiled_sharding { devices; _ } ->
      Printf.sprintf "GSPMDSharding(devices=%d)" (Array.length devices)

let slice_to_chunk extent (lo, hi) =
  ignore extent;
  if hi = lo then 0 else lo / (hi - lo)

let visualize_sharding_string ?(scale = 1.0) ?(min_width = 9) ?(max_width = 80)
    shape sharding =
  let rank = Array.length shape in
  if rank < 1 || rank > 2 then
    invalid_arg
      "`visualize_sharding` only works for shapes with 1 and 2 dimensions.";
  let base_height = int_of_float (10.0 *. scale) in
  let aspect =
    (if rank = 2 then float_of_int shape.(1) else 1.0) /. float_of_int shape.(0)
  in
  let base_width = int_of_float (float_of_int base_height *. aspect) in
  let label = String.uppercase_ascii (Backend.platform ()) in
  let cells = Hashtbl.create 8 in
  let order = ref [] in
  List.iter
    (fun ((d : device), index) ->
      let size i =
        let lo, hi = index.(i) in
        float_of_int (hi - lo) /. float_of_int shape.(i)
      in
      let chunk i = slice_to_chunk shape.(i) index.(i) in
      let key, height, width =
        if rank = 2 then ((chunk 0, chunk 1), Some (size 0), size 1)
        else ((0, chunk 0), None, size 0)
      in
      if not (Hashtbl.mem cells key) then order := key :: !order;
      let ids, _, _ =
        Option.value ~default:([], height, width) (Hashtbl.find_opt cells key)
      in
      Hashtbl.replace cells key (d.id :: ids, height, width))
    (devices_indices_map sharding shape);
  let keys = List.rev !order in
  let num_rows = 1 + List.fold_left (fun acc (r, _) -> max acc r) 0 keys in
  let num_cols = 1 + List.fold_left (fun acc (_, c) -> max acc c) 0 keys in
  let floor_div a b =
    int_of_float (Float.floor (float_of_int a /. float_of_int b))
  in
  let cell r c =
    let ids, height, width = Hashtbl.find cells (r, c) in
    let entry =
      label ^ " "
      ^ String.concat "," (List.map string_of_int (List.sort compare ids))
    in
    let w = int_of_float (width *. float_of_int base_width *. 2.5) in
    let h =
      match height with
      | None -> 1
      | Some v -> int_of_float (v *. float_of_int base_height)
    in
    let w = min (max w min_width) max_width in
    let left = floor_div (w - String.length entry - 2) 2 in
    let right = left + (w - String.length entry - 2 - (2 * left)) in
    let top = floor_div (h - 2) 2 in
    let bottom = top + (h - 2 - (2 * top)) in
    (entry, max left 0, max right 0, max top 0, max bottom 0)
  in
  let rows =
    List.init num_rows (fun r -> List.init num_cols (fun c -> cell r c))
  in
  let widths =
    List.init num_cols (fun c ->
        List.fold_left
          (fun acc row ->
            let e, l, ri, _, _ = List.nth row c in
            max acc (String.length e + l + ri))
          0 rows)
  in
  let rule left mid right =
    left
    ^ String.concat mid
        (List.map
           (fun w -> String.concat "" (List.init w (fun _ -> "\u{2500}")))
           widths)
    ^ right
  in
  let bar = "\u{2502}" in
  let render_row row =
    let height =
      List.fold_left (fun acc (_, _, _, t, b) -> max acc (t + 1 + b)) 0 row
    in
    List.init height (fun line ->
        bar
        ^ String.concat bar
            (List.map2
               (fun (e, l, r, t, _) w ->
                 if line = t then
                   String.make l ' ' ^ e ^ String.make r ' '
                   ^ String.make (w - String.length e - l - r) ' '
                 else String.make w ' ')
               row widths)
        ^ bar)
  in
  let separator = rule "\u{251C}" "\u{253C}" "\u{2524}" in
  let body =
    List.concat
      (List.mapi
         (fun i row -> (if i > 0 then [ separator ] else []) @ render_row row)
         rows)
  in
  String.concat "\n"
    ((rule "\u{250C}" "\u{252C}" "\u{2510}" :: body)
    @ [ rule "\u{2514}" "\u{2534}" "\u{2518}" ])
  ^ "\n\n"
