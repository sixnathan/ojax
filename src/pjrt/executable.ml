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

let compile client code = compile_ client code compile_options_proto
let execute exec args = execute_ exec args
let num_outputs exec = num_outputs_ exec
let destroy exec = destroy_ exec
