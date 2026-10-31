type stat_value =
  | Stat_int of int64
  | Stat_uint of int64
  | Stat_float of float
  | Stat_text of string
  | Stat_bytes of string

type profile_event = {
  event_name : string;
  start_ns : float;
  end_ns : float;
  duration_ns : float;
  event_stats : (string * stat_value) list;
}

type profile_line = { line_name : string; events : profile_event list }

type profile_plane = {
  plane_name : string;
  lines : profile_line list;
  plane_stats : (string * stat_value) list;
}

type profile_data = { planes : profile_plane list }

type profile_options = {
  host_tracer_level : int;
  python_tracer_level : int;
  device_tracer_level : int;
  enable_hlo_proto : bool;
  include_dataset_ops : bool;
  raise_error_on_start_failure : bool;
  start_timestamp_ns : int;
  duration_ms : int;
  repository_path : string;
  session_id : string;
  advanced_configuration : (string * string) list;
}

let profile_options () =
  {
    host_tracer_level = 2;
    python_tracer_level = 1;
    device_tracer_level = 1;
    enable_hlo_proto = true;
    include_dataset_ops = true;
    raise_error_on_start_failure = false;
    start_timestamp_ns = 0;
    duration_ms = 0;
    repository_path = "";
    session_id = "";
    advanced_configuration = [];
  }

type wire = Varint of int64 | Fixed64 of int64 | Bytes of string | Fixed32

exception Malformed

let parse_fields data =
  let n = String.length data in
  let pos = ref 0 in
  let varint () =
    let result = ref 0L and shift = ref 0 and go = ref true in
    while !go do
      if !pos >= n || !shift > 63 then raise Malformed;
      let byte = Char.code data.[!pos] in
      incr pos;
      result :=
        Int64.logor !result
          (Int64.shift_left (Int64.of_int (byte land 0x7f)) !shift);
      shift := !shift + 7;
      if byte land 0x80 = 0 then go := false
    done;
    !result
  in
  let fields = ref [] in
  while !pos < n do
    let tag = Int64.to_int (varint ()) in
    let field = tag lsr 3 in
    if field = 0 then raise Malformed;
    let value =
      match tag land 7 with
      | 0 -> Varint (varint ())
      | 1 ->
          if !pos + 8 > n then raise Malformed;
          let v = String.get_int64_le data !pos in
          pos := !pos + 8;
          Fixed64 v
      | 2 ->
          let len = Int64.to_int (varint ()) in
          if len < 0 || !pos + len > n then raise Malformed;
          let s = String.sub data !pos len in
          pos := !pos + len;
          Bytes s
      | 5 ->
          if !pos + 4 > n then raise Malformed;
          pos := !pos + 4;
          Fixed32
      | _ -> raise Malformed
    in
    fields := (field, value) :: !fields
  done;
  List.rev !fields

let all_bytes field fields =
  List.filter_map
    (function f, Bytes s when f = field -> Some s | _ -> None)
    fields

let one_bytes field fields =
  match all_bytes field fields with [] -> "" | s :: _ -> s

let one_int field fields =
  List.fold_left
    (fun acc f -> match f with g, Varint v when g = field -> v | _ -> acc)
    0L fields

let map_entries field fields =
  List.map
    (fun entry ->
      let parsed = parse_fields entry in
      (one_int 1 parsed, parse_fields (one_bytes 2 parsed)))
    (all_bytes field fields)

let decode_stats stat_names raw =
  List.map
    (fun bytes ->
      let f = parse_fields bytes in
      let metadata_id = one_int 1 f in
      let name id = Option.value (List.assoc_opt id stat_names) ~default:"" in
      let value =
        let rec pick = function
          | [] -> Stat_int 0L
          | (2, Fixed64 v) :: _ -> Stat_float (Int64.float_of_bits v)
          | (3, Varint v) :: _ -> Stat_uint v
          | (4, Varint v) :: _ -> Stat_int v
          | (5, Bytes s) :: _ -> Stat_text s
          | (6, Bytes s) :: _ -> Stat_bytes s
          | (7, Varint v) :: _ -> Stat_text (name v)
          | _ :: rest -> pick rest
        in
        pick f
      in
      (name metadata_id, value))
    raw

let ps_to_ns ps = Int64.to_float (Int64.div ps 1000L)

let decode_plane bytes =
  let f = parse_fields bytes in
  let names_of field =
    List.map
      (fun (key, entry) -> (key, one_bytes 2 entry))
      (map_entries field f)
  in
  let event_names = names_of 4 and stat_names = names_of 5 in
  let decode_event timestamp raw =
    let e = parse_fields raw in
    let duration = ps_to_ns (one_int 3 e) in
    let start = timestamp +. ps_to_ns (one_int 2 e) in
    {
      event_name =
        Option.value (List.assoc_opt (one_int 1 e) event_names) ~default:"";
      start_ns = start;
      end_ns = start +. duration;
      duration_ns = duration;
      event_stats = decode_stats stat_names (all_bytes 4 e);
    }
  in
  let decode_line raw =
    let l = parse_fields raw in
    let timestamp = Int64.to_float (one_int 3 l) in
    {
      line_name = one_bytes 2 l;
      events = List.map (decode_event timestamp) (all_bytes 4 l);
    }
  in
  {
    plane_name = one_bytes 2 f;
    lines = List.map decode_line (all_bytes 3 f);
    plane_stats = decode_stats stat_names (all_bytes 6 f);
  }

let profile_data_from_serialized_xspace bytes =
  match parse_fields bytes with
  | fields -> { planes = List.map decode_plane (all_bytes 1 fields) }
  | exception Malformed -> failwith "Failed to parse XSpace from array"

let profile_data_from_file path =
  let contents =
    try In_channel.with_open_bin path In_channel.input_all
    with Sys_error _ ->
      raise
        (Errors.Jax_runtime_error
           (Printf.sprintf "NOT_FOUND: %s; No such file or directory" path))
  in
  profile_data_from_serialized_xspace contents

let find_plane_with_name data name =
  List.find_opt (fun plane -> plane.plane_name = name) data.planes

let add_varint buf v =
  let v = ref v in
  let go = ref true in
  while !go do
    let byte = Int64.to_int (Int64.logand !v 0x7fL) in
    v := Int64.shift_right_logical !v 7;
    if !v = 0L then (
      Buffer.add_char buf (Char.chr byte);
      go := false)
    else Buffer.add_char buf (Char.chr (byte lor 0x80))
  done

let add_tag buf field wire =
  add_varint buf (Int64.of_int ((field lsl 3) lor wire))

let add_int buf field v =
  add_tag buf field 0;
  add_varint buf v

let add_bytes buf field s =
  add_tag buf field 2;
  add_varint buf (Int64.of_int (String.length s));
  Buffer.add_string buf s

let message build =
  let buf = Buffer.create 32 in
  build buf;
  Buffer.contents buf

type recorded = {
  r_name : string;
  r_start_ns : int64;
  r_duration_ns : int64;
  r_stats : (string * stat_value) list;
}

type session = {
  log_dir : string;
  started_ns : int64;
  options : profile_options;
  mutable recorded : recorded list;
}

let now_ns () = Int64.of_float (Unix.gettimeofday () *. 1e9)

let intern table next name =
  match Hashtbl.find_opt table name with
  | Some id -> id
  | None ->
      let id = !next in
      incr next;
      Hashtbl.replace table name id;
      id

let encode_stat stat_ids next (name, value) =
  let id = intern stat_ids next name in
  message (fun b ->
      add_int b 1 (Int64.of_int id);
      match value with
      | Stat_float v ->
          add_tag b 2 1;
          Buffer.add_int64_le b (Int64.bits_of_float v)
      | Stat_uint v -> add_int b 3 v
      | Stat_int v -> add_int b 4 v
      | Stat_text s -> add_bytes b 5 s
      | Stat_bytes s -> add_bytes b 6 s)

let encode_xspace session =
  let event_ids = Hashtbl.create 16 and next_event = ref 1 in
  let stat_ids = Hashtbl.create 16 and next_stat = ref 1 in
  let encode_event r =
    let id = intern event_ids next_event r.r_name in
    let offset_ps =
      Int64.mul (Int64.sub r.r_start_ns session.started_ns) 1000L
    in
    message (fun b ->
        add_int b 1 (Int64.of_int id);
        add_int b 2 offset_ps;
        add_int b 3 (Int64.mul r.r_duration_ns 1000L);
        List.iter
          (fun stat -> add_bytes b 4 (encode_stat stat_ids next_stat stat))
          r.r_stats)
  in
  let events = List.rev_map encode_event session.recorded in
  let line =
    message (fun b ->
        add_int b 1 1L;
        add_bytes b 2 "ocaml";
        add_int b 3 session.started_ns;
        List.iter (add_bytes b 4) events)
  in
  let metadata field table =
    Hashtbl.fold
      (fun name id acc ->
        let body =
          message (fun b ->
              add_int b 1 (Int64.of_int id);
              add_bytes b 2 name)
        in
        message (fun b ->
            add_int b 1 (Int64.of_int id);
            add_bytes b 2 body)
        :: acc)
      table []
    |> List.map (fun entry -> (field, entry))
  in
  let plane =
    message (fun b ->
        add_int b 1 1L;
        add_bytes b 2 "/host:CPU";
        add_bytes b 3 line;
        List.iter
          (fun (field, entry) -> add_bytes b field entry)
          (metadata 4 event_ids @ metadata 5 stat_ids))
  in
  message (fun b -> add_bytes b 1 plane)

let active : session option ref = ref None

let record name start stats =
  match !active with
  | Some session when session.options.host_tracer_level > 0 ->
      session.recorded <-
        {
          r_name = name;
          r_start_ns = start;
          r_duration_ns = Int64.sub (now_ns ()) start;
          r_stats = stats;
        }
        :: session.recorded
  | _ -> ()

let trace_annotation ?(stats = []) name thunk =
  match !active with
  | None -> thunk ()
  | Some _ ->
      let start = now_ns () in
      Fun.protect ~finally:(fun () -> record name start stats) thunk

let step_trace_annotation name ~step_num thunk =
  trace_annotation
    ~stats:
      [ ("_r", Stat_int 1L); ("step_num", Stat_int (Int64.of_int step_num)) ]
    name thunk

let annotate_function f = f

let start_trace ?profiler_options log_dir =
  if Option.is_some !active then
    failwith
      "Profile has already been started. Only one profile may be run at a time.";
  let options =
    match profiler_options with Some o -> o | None -> profile_options ()
  in
  active := Some { log_dir; started_ns = now_ns (); options; recorded = [] }

let rec mkdir_p path =
  if not (Sys.file_exists path) then (
    mkdir_p (Filename.dirname path);
    try Sys.mkdir path 0o755 with Sys_error _ when Sys.file_exists path -> ())

let stop_trace () =
  match !active with
  | None -> failwith "No profile started"
  | Some session ->
      active := None;
      let t = Unix.gmtime (Int64.to_float session.started_ns /. 1e9) in
      let stamp =
        Printf.sprintf "%04d_%02d_%02d_%02d_%02d_%02d" (t.tm_year + 1900)
          (t.tm_mon + 1) t.tm_mday t.tm_hour t.tm_min t.tm_sec
      in
      let dir =
        List.fold_left Filename.concat session.log_dir
          [ "plugins"; "profile"; stamp ]
      in
      mkdir_p dir;
      let path = Filename.concat dir (Unix.gethostname () ^ ".xplane.pb") in
      Out_channel.with_open_bin path (fun oc ->
          output_string oc (encode_xspace session))

let trace ?profiler_options log_dir thunk =
  start_trace ?profiler_options log_dir;
  Fun.protect ~finally:stop_trace thunk

type profiler_server = { socket : Unix.file_descr; port : int }

let server : profiler_server option ref = ref None

let start_server ?requires_backend:_ port =
  if Option.is_some !server then
    invalid_arg "Only one profiler server can be active at a time.";
  let socket = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  (try
     Unix.setsockopt socket Unix.SO_REUSEADDR true;
     Unix.bind socket (Unix.ADDR_INET (Unix.inet_addr_any, port));
     Unix.listen socket 8
   with Unix.Unix_error (error, _, _) ->
     Unix.close socket;
     failwith
       (Printf.sprintf "Could not start the profiler server on port %d: %s" port
          (Unix.error_message error)));
  let port =
    match Unix.getsockname socket with
    | Unix.ADDR_INET (_, bound) -> bound
    | Unix.ADDR_UNIX _ -> port
  in
  let handle = { socket; port } in
  server := Some handle;
  handle

let stop_server () =
  match !server with
  | None -> invalid_arg "No active profiler server."
  | Some handle ->
      Unix.close handle.socket;
      server := None

let server_port handle = handle.port
let registered_subprocesses : int list ref = ref []

let register_subprocess pid port =
  if pid = Unix.getpid () then
    raise
      (Errors.Jax_runtime_error
         "INVALID_ARGUMENT: Cannot register subprocess with the same pid as \
          current process.");
  if List.mem pid !registered_subprocesses then
    raise
      (Errors.Jax_runtime_error
         (Printf.sprintf "ALREADY_EXISTS: Subprocess %d is already registered."
            pid));
  let socket = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close socket)
    (fun () ->
      try Unix.connect socket (Unix.ADDR_INET (Unix.inet_addr_loopback, port))
      with Unix.Unix_error _ ->
        raise
          (Errors.Jax_runtime_error
             (Printf.sprintf
                "DEADLINE_EXCEEDED: Timeout while connecting to localhost:%d"
                port)));
  registered_subprocesses := pid :: !registered_subprocesses;
  fun () ->
    registered_subprocesses :=
      List.filter (fun p -> p <> pid) !registered_subprocesses

let device_memory_profile () = ""

let save_device_memory_profile path =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out_noerr oc)
    (fun () -> output_string oc (device_memory_profile ()))
