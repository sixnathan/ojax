type t

external compile_ : Client.t -> string -> string -> t = "ojax_pjrt_compile"

external execute_ : t -> Buffer.raw array -> Buffer.raw array
  = "ojax_pjrt_execute"

external execute_devices_ :
  t -> Buffer.raw array array -> Buffer.raw array array
  = "ojax_pjrt_execute_devices"

external optimized_ : t -> string = "ojax_pjrt_executable_optimized"
external serialize_ : t -> string = "ojax_pjrt_executable_serialize"

type memory_stats = {
  generated_code_size_in_bytes : int;
  argument_size_in_bytes : int;
  output_size_in_bytes : int;
  alias_size_in_bytes : int;
  temp_size_in_bytes : int;
  host_generated_code_size_in_bytes : int;
  host_argument_size_in_bytes : int;
  host_output_size_in_bytes : int;
  host_alias_size_in_bytes : int;
  host_temp_size_in_bytes : int;
  peak_memory_in_bytes : int;
}

external memory_stats : t -> memory_stats = "ojax_pjrt_executable_memory_stats"

external deserialize_ : Client.t -> string -> t
  = "ojax_pjrt_executable_deserialize"

external num_outputs_ : t -> int = "ojax_pjrt_executable_num_outputs"
external destroy_ : t -> unit = "ojax_pjrt_executable_destroy"

let cache_dir = lazy (Sys.getenv_opt "OJAX_TEST_COMPILE_CACHE")
let plugin_hash = lazy (Option.get (Discover.verified_sha256 ()))

let cache_key client code options =
  String.concat "\000"
    [
      Lazy.force plugin_hash;
      Client.platform_name client;
      string_of_int (Client.device_count client);
      options;
      code;
    ]

let read_cache path key =
  match open_in_bin path with
  | exception Sys_error _ -> None
  | ic ->
      Fun.protect ~finally:(fun () -> close_in_noerr ic) @@ fun () ->
      let n = String.length key in
      if in_channel_length ic >= n && really_input_string ic n = key then
        Some (really_input_string ic (in_channel_length ic - n))
      else None

let write_cache dir path key exec =
  match serialize_ exec with
  | exception Abi.Error _ -> ()
  | bytes -> (
      (try Sys.mkdir dir 0o755 with Sys_error _ -> ());
      let temp = Printf.sprintf "%s.%d.tmp" path (Unix.getpid ()) in
      try
        Out_channel.with_open_bin temp (fun oc ->
            output_string oc key;
            output_string oc bytes);
        Sys.rename temp path
      with Sys_error _ -> ())

let compile_cached client code options =
  match Lazy.force cache_dir with
  | Some dir when Client.platform_name client = "cpu" -> (
      let key = cache_key client code options in
      let path = Filename.concat dir (Digest.to_hex (Digest.string key)) in
      let loaded =
        Option.bind (read_cache path key) (fun bytes ->
            try Some (deserialize_ client bytes) with Abi.Error _ -> None)
      in
      match loaded with
      | Some exec -> exec
      | None ->
          let exec = compile_ client code options in
          write_cache dir path key exec;
          exec)
  | _ -> compile_ client code options

let add_varint buf v =
  let v = ref v in
  let continue = ref true in
  while !continue do
    let byte = !v land 0x7f in
    v := !v lsr 7;
    if !v = 0 then (
      Stdlib.Buffer.add_char buf (Char.chr byte);
      continue := false)
    else Stdlib.Buffer.add_char buf (Char.chr (byte lor 0x80))
  done

let add_tag buf field wire = add_varint buf ((field lsl 3) lor wire)

let executable_build_options_proto =
  let b = Stdlib.Buffer.create 8 in
  add_tag b 4 0;
  add_varint b 1;
  add_tag b 5 0;
  add_varint b 1;
  Stdlib.Buffer.contents b

let compile_options_proto =
  let b = Stdlib.Buffer.create 8 in
  add_tag b 3 2;
  add_varint b (String.length executable_build_options_proto);
  Stdlib.Buffer.add_string b executable_build_options_proto;
  Stdlib.Buffer.contents b

type compiler_option =
  | Option_string of string
  | Option_bool of bool
  | Option_int of int
  | Option_float of float

let option_override_proto opt =
  let b = Stdlib.Buffer.create 8 in
  (match opt with
  | Option_string s ->
      add_tag b 1 2;
      add_varint b (String.length s);
      Stdlib.Buffer.add_string b s
  | Option_bool v ->
      add_tag b 2 0;
      add_varint b (if v then 1 else 0)
  | Option_int v ->
      add_tag b 3 0;
      add_varint b v
  | Option_float v ->
      add_tag b 4 1;
      Stdlib.Buffer.add_int64_le b (Int64.bits_of_float v));
  Stdlib.Buffer.contents b

let env_override_entry (key, opt) =
  let body = option_override_proto opt in
  let entry = Stdlib.Buffer.create 16 in
  add_tag entry 1 2;
  add_varint entry (String.length key);
  Stdlib.Buffer.add_string entry key;
  add_tag entry 2 2;
  add_varint entry (String.length body);
  Stdlib.Buffer.add_string entry body;
  let out = Stdlib.Buffer.create 16 in
  add_tag out 7 2;
  add_varint out (Stdlib.Buffer.length entry);
  Stdlib.Buffer.add_buffer out entry;
  Stdlib.Buffer.contents out

let compile ?(options = []) client code =
  let proto =
    compile_options_proto
    ^ String.concat "" (List.map env_override_entry options)
  in
  compile_cached client code proto

let device_assignment_proto devices =
  let b = Stdlib.Buffer.create 32 in
  add_tag b 1 0;
  add_varint b 1;
  add_tag b 2 0;
  add_varint b (Array.length devices);
  Array.iter
    (fun id ->
      let row = Stdlib.Buffer.create 4 in
      add_tag row 1 0;
      add_varint row id;
      add_tag b 3 2;
      add_varint b (Stdlib.Buffer.length row);
      Stdlib.Buffer.add_buffer b row)
    devices;
  Stdlib.Buffer.contents b

let placed_options_proto ~devices ~propagate_outputs =
  let inner = Stdlib.Buffer.create 32 in
  add_tag inner 4 0;
  add_varint inner 1;
  add_tag inner 5 0;
  add_varint inner (Array.length devices);
  if Array.length devices > 1 then (
    add_tag inner 6 0;
    add_varint inner 1;
    add_tag inner 19 0;
    add_varint inner 1);
  let assignment = device_assignment_proto devices in
  add_tag inner 9 2;
  add_varint inner (String.length assignment);
  Stdlib.Buffer.add_string inner assignment;
  if propagate_outputs then (
    add_tag inner 12 2;
    add_varint inner 1;
    add_varint inner 1);
  let b = Stdlib.Buffer.create 64 in
  add_tag b 3 2;
  add_varint b (Stdlib.Buffer.length inner);
  Stdlib.Buffer.add_buffer b inner;
  Stdlib.Buffer.contents b

let compile_placed client code ~devices ~propagate_outputs =
  compile_cached client code (placed_options_proto ~devices ~propagate_outputs)

type output_sharding =
  | Replicated
  | Tiled of { dims : int array; grid : int array }

let read_varint text pos =
  let rec go shift acc =
    let byte = Char.code text.[!pos] in
    incr pos;
    let acc = acc lor ((byte land 0x7f) lsl shift) in
    if byte land 0x80 = 0 then acc else go (shift + 7) acc
  in
  go 0 0

let proto_fields text =
  let pos = ref 0 in
  let out = ref [] in
  while !pos < String.length text do
    let tag = read_varint text pos in
    let field = tag lsr 3 in
    match tag land 7 with
    | 0 -> out := (field, `Int (read_varint text pos)) :: !out
    | 1 -> pos := !pos + 8
    | 2 ->
        let n = read_varint text pos in
        out := (field, `Bytes (String.sub text !pos n)) :: !out;
        pos := !pos + n
    | 5 -> pos := !pos + 4
    | wire ->
        failwith (Printf.sprintf "Pjrt.Executable: proto wire type %d" wire)
  done;
  List.rev !out

let packed_ints field fields =
  List.concat_map
    (fun (f, v) ->
      if f <> field then []
      else
        match v with
        | `Int i -> [ i ]
        | `Bytes b ->
            let pos = ref 0 in
            let acc = ref [] in
            while !pos < String.length b do
              acc := read_varint b pos :: !acc
            done;
            List.rev !acc)
    fields
  |> Array.of_list

let iota_grid dims perm =
  let total = Array.fold_left ( * ) 1 dims in
  let rank = Array.length dims in
  let strides = Array.make rank 1 in
  for i = rank - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * dims.(i + 1)
  done;
  let moved = Array.map (fun p -> dims.(p)) perm in
  Array.init total (fun flat ->
      let rem = ref flat in
      let coords = Array.make rank 0 in
      for i = rank - 1 downto 0 do
        coords.(i) <- !rem mod moved.(i);
        rem := !rem / moved.(i)
      done;
      let value = ref 0 in
      Array.iteri (fun i p -> value := !value + (coords.(i) * strides.(p))) perm;
      !value)

let decode_sharding fields =
  let kind = match List.assoc_opt 1 fields with Some (`Int k) -> k | _ -> 0 in
  match kind with
  | 0 -> Replicated
  | 3 ->
      let dims = packed_ints 3 fields in
      let devices = packed_ints 4 fields in
      let reshape = packed_ints 9 fields in
      let perm = packed_ints 10 fields in
      let grid =
        if Array.length reshape > 0 then iota_grid reshape perm else devices
      in
      Tiled { dims; grid }
  | k -> failwith (Printf.sprintf "Pjrt.Executable: OpSharding type %d" k)

let output_shardings exec =
  let module_fields =
    match List.assoc_opt 1 (proto_fields (optimized_ exec)) with
    | Some (`Bytes b) -> proto_fields b
    | _ -> failwith "Pjrt.Executable: optimized program has no module"
  in
  match List.assoc_opt 12 module_fields with
  | None -> failwith "Pjrt.Executable: optimized program has no output sharding"
  | Some (`Int _) -> failwith "Pjrt.Executable: malformed output sharding"
  | Some (`Bytes b) ->
      let fields = proto_fields b in
      if List.assoc_opt 1 fields = Some (`Int 2) then
        List.filter_map
          (fun (f, v) ->
            match (f, v) with
            | 5, `Bytes sub -> Some (decode_sharding (proto_fields sub))
            | _ -> None)
          fields
      else [ decode_sharding fields ]

let optimized_module exec =
  match List.assoc_opt 1 (proto_fields (optimized_ exec)) with
  | Some (`Bytes b) -> b
  | _ -> failwith "Pjrt.Executable: optimized program has no module"

let cost_analysis exec = Hlo_cost.of_module (optimized_module exec)

let execute exec args =
  Array.map
    (fun raw -> Buffer.Single raw)
    (execute_ exec (Array.map Buffer.single_raw args))

let execute_devices exec lists = execute_devices_ exec lists
let num_outputs exec = num_outputs_ exec
let destroy exec = destroy_ exec
