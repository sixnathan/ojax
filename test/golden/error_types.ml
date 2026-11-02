let both_raise = "both_raise"
let mismatch_prefix = "both_raise_type:"
let mismatch_code = -8

let authored_stands_for =
  [
    "ValueError";
    "TypeError";
    "IndexError";
    "OverflowError";
    "NotImplementedError";
  ]

let authored_classes = [ "Failure"; "Invalid_argument" ]

let incidental_exact =
  [
    "index out of bounds";
    "hd";
    "tl";
    "nth";
    "int_of_string";
    "float_of_string";
    "bool_of_string";
  ]

let incidental_prefixes =
  [
    "Array.";
    "List.";
    "String.";
    "Bytes.";
    "Option.";
    "Seq.";
    "Hashtbl.";
    "Buffer.";
    "Bigarray.";
    "Float.Array.";
    "Stdlib.";
    "Ndarray.";
    "parity:";
    "coverage:";
    "fuzz:";
  ]

let ojax_class e =
  let slot = Printexc.exn_slot_name e in
  match String.rindex_opt slot '.' with
  | Some i -> String.sub slot (i + 1) (String.length slot - i - 1)
  | None -> slot

let message_of e =
  match e with
  | Failure m | Invalid_argument m -> m
  | e -> (
      match Ojax.Py_exn.message_of e with
      | Some m -> m
      | None -> Printexc.to_string e)

let is_incidental message =
  List.mem message incidental_exact
  || List.exists
       (fun prefix -> String.starts_with ~prefix message)
       incidental_prefixes

let python_class = Ojax.Py_exn.class_name

let tier ~jax e =
  if python_class e = Some jax then Some "typed"
  else if
    List.mem jax authored_stands_for
    && List.mem (ojax_class e) authored_classes
    && not (is_incidental (message_of e))
  then Some "authored"
  else None

let one_line message =
  String.map (fun c -> if c = '\t' || c = '\n' then ' ' else c) message

let mismatch ~jax e =
  Printf.sprintf "%s%s!=%s:%s" mismatch_prefix jax (ojax_class e)
    (let line = one_line (message_of e) in
     if String.length line > 120 then String.sub line 0 120 else line)

let detail ~jax e =
  Printf.sprintf "%s\t%s\t%s\t%s"
    (Option.value (tier ~jax e) ~default:"mismatch")
    jax (ojax_class e)
    (String.escaped (message_of e))
