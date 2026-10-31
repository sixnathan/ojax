open Types

module type S = sig
  type buffer
  type compiled

  val of_host : Ndarray.t -> buffer
  val to_host : buffer -> Ndarray.t
  val compile : closed_jaxpr -> compiled
  val execute : compiled -> buffer list -> buffer list
end

module Interpreter = struct
  type buffer = Ndarray.t
  type compiled = closed_jaxpr

  let of_host nd = nd
  let to_host nd = nd
  let compile cj = cj

  let concrete = function
    | Concrete nd -> nd
    | Device b -> Pjrt.Buffer.to_host b
    | Tracer _ -> failwith "Backend.Interpreter: unexpected tracer result"

  let execute cj bufs =
    List.map concrete
      (Jaxpr.eval_closed_jaxpr cj (List.map (fun nd -> Concrete nd) bufs))
end

module Xla = struct
  type buffer = Pjrt.Buffer.t
  type compiled = Pjrt.Executable.t

  let forced_cpu_devices () =
    let configured = Config.value Config.num_cpu_devices in
    if configured > 0 then Some configured
    else
      let prefix = "--xla_force_host_platform_device_count=" in
      let n = String.length prefix in
      match Sys.getenv_opt "XLA_FLAGS" with
      | None -> None
      | Some flags -> (
          String.split_on_char ' ' flags
          |> List.filter_map (fun flag ->
              if String.length flag > n && String.sub flag 0 n = prefix then
                int_of_string_opt (String.sub flag n (String.length flag - n))
              else None)
          |> List.rev
          |> function
          | latest :: _ -> Some latest
          | [] -> None)

  let plugin_root = ref None

  let client =
    lazy
      (let path = Pjrt.Discover.preflight () in
       let plugin = Pjrt.Abi.open_plugin path in
       plugin_root := Some plugin;
       let first = Pjrt.Client.create plugin in
       match forced_cpu_devices () with
       | Some wanted
         when Pjrt.Client.platform first = "cpu"
              && wanted <> Array.length (Pjrt.Client.devices first) ->
           Pjrt.Client.destroy first;
           Pjrt.Client.create ~cpu_devices:wanted plugin
       | _ -> first)

  let client_handle () = Lazy.force client
  let of_host nd = Pjrt.Buffer.of_host (Lazy.force client) nd
  let to_host buf = Pjrt.Buffer.to_host buf
  let compile_text text = Pjrt.Executable.compile (Lazy.force client) text
  let compile cj = compile_text (Stablehlo.Emit.emit_closed_jaxpr cj)

  let execute exec bufs =
    Array.to_list (Pjrt.Executable.execute exec (Array.of_list bufs))
end

let plugin_available () =
  try
    ignore (Pjrt.Discover.preflight ());
    true
  with Pjrt.Discover.Error _ -> false

let use_xla =
  lazy
    (match Sys.getenv_opt "OJAX_BACKEND" with
    | Some "interpreter" -> false
    | Some "xla" -> true
    | None -> plugin_available ()
    | Some other ->
        invalid_arg
          (Printf.sprintf
             "OJAX_BACKEND: unknown backend %S (expected \"interpreter\" or \
              \"xla\")"
             other))

let is_device = function Device _ -> true | Concrete _ | Tracer _ -> false
let is_hostable = function Concrete _ | Device _ -> true | Tracer _ -> false

let to_input = function
  | Device b -> (b, false)
  | Concrete nd -> (Xla.of_host nd, true)
  | Tracer _ -> failwith "Backend: xla execution requires concrete arguments"

let xla_run exec args resident =
  let inputs = List.map to_input args in
  Fun.protect
    ~finally:(fun () ->
      List.iter (fun (b, owned) -> if owned then Pjrt.Buffer.destroy b) inputs)
    (fun () ->
      let outs = Xla.execute exec (List.map fst inputs) in
      if resident then List.map (fun b -> Device b) outs
      else
        Fun.protect
          ~finally:(fun () -> List.iter Pjrt.Buffer.destroy outs)
          (fun () -> List.map (fun b -> Concrete (Xla.to_host b)) outs))

let compiled_on_xla = ref 0
let host_fallbacks : string list ref = ref []
let stats () = (!compiled_on_xla, List.rev !host_fallbacks)

module PB = Pjrt.Buffer
module PE = Pjrt.Executable

let local_devices =
  let cached =
    lazy
      (if Lazy.force use_xla then
         let client = Xla.client_handle () in
         let all = Pjrt.Client.devices client in
         if
           Pjrt.Client.platform client = "cpu"
           && Xla.forced_cpu_devices () = None
         then Array.sub all 0 1
         else all
       else [| { Pjrt.Client.id = 0; process_index = 0; kind = "cpu" } |])
  in
  fun () -> Lazy.force cached

let multi_device_available () = Lazy.force use_xla

let platform () =
  if Lazy.force use_xla then Pjrt.Client.platform (Xla.client_handle ())
  else "cpu"

let mesh_coordinates mesh pos =
  let rem = ref pos in
  List.rev_map
    (fun (axis, size) ->
      let c = !rem mod size in
      rem := !rem / size;
      (axis, c))
    (List.rev mesh)

let check_divisible ~shape ~mesh ~spec =
  Array.iteri
    (fun d axes ->
      let tiles =
        List.fold_left (fun acc a -> acc * List.assoc a mesh) 1 axes
      in
      if tiles > 1 && shape.(d) mod tiles <> 0 then
        invalid_arg
          (Printf.sprintf
             "Sharding implies that array axis %d is partitioned %d times, but \
              the dimension size is %d (full shape: (%s), per-dimension tiling \
              factors should evenly divide the shape)"
             d tiles shape.(d)
             (String.concat ", "
                (Array.to_list (Array.map string_of_int shape)))))
    spec

let named_index ~shape ~mesh ~spec pos =
  let coords = mesh_coordinates mesh pos in
  Array.mapi
    (fun d axes ->
      let tile, tiles =
        List.fold_left
          (fun (tile, tiles) a ->
            let size = List.assoc a mesh in
            ((tile * size) + List.assoc a coords, tiles * size))
          (0, 1) axes
      in
      let chunk = shape.(d) / tiles in
      (tile * chunk, (tile + 1) * chunk))
    spec

let replica_ids indices =
  let seen = Hashtbl.create 8 in
  Array.map
    (fun index ->
      let n = Option.value ~default:0 (Hashtbl.find_opt seen index) in
      Hashtbl.replace seen index (n + 1);
      n)
    indices

let placed_of ~shape ~dtype ~devices ~indices ~layout raws =
  let ids = replica_ids indices in
  Device
    (PB.Placed
       {
         global_shape = shape;
         dtype;
         layout;
         shards =
           Array.mapi
             (fun p raw ->
               {
                 PB.device = devices.(p);
                 index = indices.(p);
                 replica_id = ids.(p);
                 data = raw;
               })
             raws;
       })

let host_of = function
  | Concrete nd -> nd
  | Device b -> PB.to_host b
  | Tracer _ -> failwith "Backend: placement requires concrete arguments"

let place_named nd ~mesh ~devices ~spec =
  let shape = Ndarray.shape nd in
  check_divisible ~shape ~mesh ~spec;
  let client = Xla.client_handle () in
  let indices =
    Array.mapi (fun pos _ -> named_index ~shape ~mesh ~spec pos) devices
  in
  let raws =
    PB.shards_of_host client nd
      (Array.mapi (fun pos device -> (indices.(pos), device)) devices)
  in
  placed_of ~shape ~dtype:(Ndarray.dtype nd) ~devices ~indices
    ~layout:(PB.Named { mesh_axes = mesh; mesh_devices = devices; spec })
    raws

let place_on nd device =
  let shape = Ndarray.shape nd in
  let index = Array.map (fun d -> (0, d)) shape in
  let raws =
    PB.shards_of_host (Xla.client_handle ()) nd [| (index, device) |]
  in
  placed_of ~shape ~dtype:(Ndarray.dtype nd) ~devices:[| device |]
    ~indices:[| index |] ~layout:(PB.On_device device) raws

let replicated_spec rank = Array.make rank []

type plan = {
  mesh : (string * int) list;
  devices : int array;
  arg_specs : string list array list;
  out_specs : string list array option list;
}

type placed_program = {
  program : PE.t;
  tilings : PE.output_sharding list Lazy.t;
}

let placed_programs : (int * plan, placed_program option) Hashtbl.t =
  Hashtbl.create 16

let compile_plan (cj : closed_jaxpr) plan =
  match Hashtbl.find_opt placed_programs (cj.jid, plan) with
  | Some entry -> entry
  | None ->
      let entry =
        match
          if Array.length plan.devices = 1 then
            Stablehlo.Emit.emit_closed_jaxpr cj
          else
            Stablehlo.Emit.emit_closed_jaxpr
              ~partition:
                {
                  pt_mesh = plan.mesh;
                  pt_devices = plan.devices;
                  pt_in = plan.arg_specs;
                  pt_out = plan.out_specs;
                }
              cj
        with
        | text ->
            let propagate_outputs =
              Array.length plan.devices > 1
              && List.exists Option.is_none plan.out_specs
            in
            let program =
              PE.compile_placed (Xla.client_handle ()) text
                ~devices:plan.devices ~propagate_outputs
            in
            incr compiled_on_xla;
            Some { program; tilings = lazy (PE.output_shardings program) }
        | exception (Failure reason | Invalid_argument reason) ->
            host_fallbacks := reason :: !host_fallbacks;
            None
      in
      Hashtbl.replace placed_programs (cj.jid, plan) entry;
      entry

let raws_for_arg plan arg spec =
  match arg with
  | Device (PB.Placed { layout = PB.Named n; shards; _ })
    when n.mesh_axes = plan.mesh
         && n.mesh_devices = plan.devices
         && n.spec = spec ->
      Array.map
        (fun device ->
          let s =
            List.find
              (fun (s : PB.shard) -> s.device = device)
              (Array.to_list shards)
          in
          s.data)
        plan.devices
  | _ ->
      let nd = host_of arg in
      let shape = Ndarray.shape nd in
      check_divisible ~shape ~mesh:plan.mesh ~spec;
      let client = Xla.client_handle () in
      PB.shards_of_host client nd
        (Array.mapi
           (fun pos device ->
             (named_index ~shape ~mesh:plan.mesh ~spec pos, device))
           plan.devices)

let raw_on_device device = function
  | Device (PB.Placed { shards = [| s |]; _ }) when s.device = device -> s.data
  | Device (PB.Single raw) when PB.raw_device raw = device -> raw
  | arg ->
      let nd = host_of arg in
      (PB.shards_of_host (Xla.client_handle ()) nd
         [| (Array.map (fun d -> (0, d)) (Ndarray.shape nd), device) |]).(0)

let indices_of_tiling shape devices tiling =
  let full = Array.map (fun d -> (0, d)) shape in
  match tiling with
  | PE.Replicated -> Array.map (fun _ -> full) devices
  | PE.Tiled { dims; grid } ->
      let out = Array.map (fun _ -> full) devices in
      Array.iteri
        (fun flat partition ->
          let rem = ref flat in
          let coords = Array.make (Array.length dims) 0 in
          for i = Array.length dims - 1 downto 0 do
            coords.(i) <- !rem mod dims.(i);
            rem := !rem / dims.(i)
          done;
          out.(partition) <-
            Array.mapi
              (fun d extent ->
                let chunk = extent / dims.(d) in
                (coords.(d) * chunk, (coords.(d) + 1) * chunk))
              shape)
        grid;
      out

let tiles_of_tiling rank = function
  | PE.Replicated -> Array.make rank 1
  | PE.Tiled { dims; _ } -> Array.sub dims 0 rank

let derive_spec ~shape ~mesh indices =
  let rank = Array.length shape in
  let tiles_wanted =
    Array.mapi
      (fun d extent ->
        let lo, hi = indices.(0).(d) in
        if extent = 0 then 1 else extent / max 1 (hi - lo))
      shape
  in
  let rec choose_axes unused product =
    if product = 1 then [ [] ]
    else
      List.concat_map
        (fun (axis, size) ->
          if size > 1 && product mod size = 0 then
            List.map
              (fun rest -> axis :: rest)
              (choose_axes
                 (List.filter (fun (a, _) -> a <> axis) unused)
                 (product / size))
          else [])
        unused
  in
  let rec assign d unused =
    if d = rank then [ [] ]
    else
      List.concat_map
        (fun axes ->
          let remaining =
            List.filter (fun (a, _) -> not (List.mem a axes)) unused
          in
          List.map (fun rest -> axes :: rest) (assign (d + 1) remaining))
        (choose_axes unused tiles_wanted.(d))
  in
  List.find_map
    (fun candidate ->
      let spec = Array.of_list candidate in
      let matches =
        Array.for_all Fun.id
          (Array.mapi
             (fun pos index -> named_index ~shape ~mesh ~spec pos = index)
             indices)
      in
      if matches then Some spec else None)
    (assign 0 mesh)

let run_program plan placed args =
  let single = Array.length plan.devices = 1 in
  let lists =
    if single then
      [| Array.of_list (List.map (raw_on_device plan.devices.(0)) args) |]
    else
      let per_arg =
        Array.of_list (List.map2 (raws_for_arg plan) args plan.arg_specs)
      in
      Array.mapi
        (fun pos _ -> Array.map (fun raws -> raws.(pos)) per_arg)
        plan.devices
  in
  let outs = PE.execute_devices placed.program lists in
  let n_out = if Array.length outs = 0 then 0 else Array.length outs.(0) in
  List.init n_out (fun k ->
      let raws = Array.map (fun row -> row.(k)) outs in
      let probe = PB.Single raws.(0) in
      let dtype = PB.element_type probe in
      if single then
        let shape = PB.raw_dimensions raws.(0) in
        placed_of ~shape ~dtype ~devices:plan.devices
          ~indices:[| Array.map (fun d -> (0, d)) shape |]
          ~layout:(PB.On_device plan.devices.(0))
          raws
      else
        let tiling =
          match List.nth_opt plan.out_specs k with
          | Some (Some _) -> None
          | _ -> Some (List.nth (Lazy.force placed.tilings) k)
        in
        let local = PB.raw_dimensions raws.(0) in
        let shape =
          match (List.nth_opt plan.out_specs k, tiling) with
          | Some (Some spec), _ ->
              Array.mapi
                (fun d extent ->
                  List.fold_left
                    (fun acc a -> acc * List.assoc a plan.mesh)
                    extent spec.(d))
                local
          | _, Some t ->
              let tiles = tiles_of_tiling (Array.length local) t in
              Array.mapi (fun d extent -> extent * tiles.(d)) local
          | _ -> local
        in
        let indices, layout =
          match (List.nth_opt plan.out_specs k, tiling) with
          | Some (Some spec), _ ->
              ( Array.mapi
                  (fun pos _ -> named_index ~shape ~mesh:plan.mesh ~spec pos)
                  plan.devices,
                PB.Named
                  { mesh_axes = plan.mesh; mesh_devices = plan.devices; spec }
              )
          | _, Some t -> (
              let indices = indices_of_tiling shape plan.devices t in
              match derive_spec ~shape ~mesh:plan.mesh indices with
              | Some spec ->
                  ( indices,
                    PB.Named
                      {
                        mesh_axes = plan.mesh;
                        mesh_devices = plan.devices;
                        spec;
                      } )
              | None -> (indices, PB.Tiled))
          | _ -> failwith "Backend: output has no sharding"
        in
        placed_of ~shape ~dtype ~devices:plan.devices ~indices ~layout raws)

let rank_of v = Array.length (Core.get_aval v).shape

let incompatible () =
  invalid_arg "Received incompatible devices for jitted computation."

let plan_of_args args =
  let placed =
    List.filter_map (function Device (PB.Placed p) -> Some p | _ -> None) args
  in
  let device_list (p : PB.placed) =
    Array.map (fun (s : PB.shard) -> s.device) p.shards
  in
  let one_device =
    List.for_all (fun (p : PB.placed) -> Array.length p.shards = 1) placed
  in
  match placed with
  | [] -> None
  | first :: _ when one_device ->
      let device = (device_list first).(0) in
      if List.exists (fun p -> (device_list p).(0) <> device) placed then
        incompatible ();
      Some
        {
          mesh = [];
          devices = [| device |];
          arg_specs = List.map (fun v -> replicated_spec (rank_of v)) args;
          out_specs = [];
        }
  | _ ->
      let named =
        List.filter_map
          (fun (p : PB.placed) ->
            match p.layout with PB.Named n -> Some n | _ -> None)
          placed
      in
      let mesh, devices =
        match named with
        | n :: _ -> (n.mesh_axes, n.mesh_devices)
        | [] ->
            let d = device_list (List.hd placed) in
            ([ ("_tiled", Array.length d) ], d)
      in
      List.iter
        (fun (p : PB.placed) ->
          match p.layout with
          | PB.Named n when n.mesh_axes = mesh && n.mesh_devices = devices -> ()
          | PB.Tiled when device_list p = devices -> ()
          | _ -> incompatible ())
        placed;
      Some
        {
          mesh;
          devices;
          arg_specs =
            List.map
              (function
                | Device (PB.Placed { layout = PB.Named n; _ }) -> n.spec
                | v -> replicated_spec (rank_of v))
              args;
          out_specs = [];
        }

let plan_for_mesh mesh devices args =
  {
    mesh;
    devices;
    arg_specs =
      List.map
        (function
          | Device (PB.Placed { layout = PB.Named n; _ })
            when n.mesh_axes = mesh && n.mesh_devices = devices ->
              n.spec
          | v -> replicated_spec (rank_of v))
        args;
    out_specs = [];
  }

let run_plan cj plan args =
  match compile_plan cj plan with
  | Some placed -> Some (run_program plan placed args)
  | None -> None

let () =
  Core.placed_eval :=
    fun prim args ->
      match plan_of_args args with
      | None -> None
      | Some plan -> (
          let avals = List.map Core.get_aval args in
          match Jaxpr.make_jaxpr avals (fun vs -> Core.bind prim vs) with
          | exception (Failure _ | Invalid_argument _) -> None
          | cj -> run_plan cj plan args)

let executor (cj : closed_jaxpr) : value list -> value list =
  if Lazy.force use_xla then (
    let emitted =
      lazy
        (match Stablehlo.Emit.emit_closed_jaxpr cj with
        | text -> (Some text, Stablehlo.Emit.shard_mesh_seen ())
        | exception (Failure reason | Invalid_argument reason) ->
            host_fallbacks := reason :: !host_fallbacks;
            (None, None))
    in
    let exec =
      lazy
        (match fst (Lazy.force emitted) with
        | Some text ->
            incr compiled_on_xla;
            Some (Xla.compile_text text)
        | None -> None)
    in
    let plain args =
      match Lazy.force exec with
      | Some compiled when List.for_all is_hostable args ->
          xla_run compiled args (List.exists is_device args)
      | Some _ | None -> Jaxpr.eval_closed_jaxpr cj args
    in
    fun args ->
      if Config.value Config.no_execution then
        failwith "compiled execution is disabled by no_execution";
      if not (List.for_all is_hostable args) then plain args
      else
        let via plan =
          match run_plan cj plan args with
          | Some outs -> outs
          | None -> Jaxpr.eval_closed_jaxpr cj args
        in
        match snd (Lazy.force emitted) with
        | Some (mesh, devices) when Array.length devices > 1 ->
            via (plan_for_mesh mesh devices args)
        | _ -> (
            match plan_of_args args with
            | Some plan -> via plan
            | None -> plain args))
  else fun args -> Jaxpr.eval_closed_jaxpr cj args

let run_text text args =
  if Lazy.force use_xla then
    let exec = Xla.compile_text text in
    xla_run exec args (List.exists is_device args)
  else
    failwith
      "Running an exported module needs the PJRT plugin; set OJAX_PJRT_PLUGIN"

let of_host_value v =
  if Lazy.force use_xla then
    match v with
    | Concrete nd -> Device (Xla.of_host nd)
    | Device _ | Tracer _ -> v
  else v

let to_host_value = function
  | Device b -> Concrete (Pjrt.Buffer.to_host b)
  | (Concrete _ | Tracer _) as v -> v

let client () = Xla.client_handle ()
let clear_placed_programs () = Hashtbl.reset placed_programs
