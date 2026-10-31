type t

external compile_ : Client.t -> string -> string -> t = "ojax_pjrt_compile"
external execute_ : t -> Buffer.t array -> Buffer.t array = "ojax_pjrt_execute"
external num_outputs_ : t -> int = "ojax_pjrt_executable_num_outputs"
external destroy_ : t -> unit = "ojax_pjrt_executable_destroy"

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
  compile_ client code proto

let execute exec args = execute_ exec args
let num_outputs exec = num_outputs_ exec
let destroy exec = destroy_ exec
