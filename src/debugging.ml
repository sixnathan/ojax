open Types

let render v =
  match v with
  | Concrete _ | Device _ -> Numpy.Extras.array_str v
  | Tracer _ -> "<traced>"

let format_message fmt args =
  let buffer = Buffer.create (String.length fmt + 16) in
  let remaining = ref args in
  let n = String.length fmt in
  let i = ref 0 in
  while !i < n do
    (match fmt.[!i] with
    | '{' when !i + 1 < n && fmt.[!i + 1] = '{' ->
        Buffer.add_char buffer '{';
        incr i
    | '}' when !i + 1 < n && fmt.[!i + 1] = '}' ->
        Buffer.add_char buffer '}';
        incr i
    | '{' -> (
        (match String.index_from_opt fmt !i '}' with
        | Some close -> i := close
        | None -> invalid_arg "debug.print: unmatched brace");
        match !remaining with
        | v :: rest ->
            remaining := rest;
            Buffer.add_string buffer (render v)
        | [] -> invalid_arg "debug.print: not enough arguments")
    | c -> Buffer.add_char buffer c);
    incr i
  done;
  Buffer.contents buffer

let callback f args =
  let host values =
    f values;
    []
  in
  ignore (Custom_derivatives.pure_callback host ~result_avals:[] args)

let print fmt args =
  callback (fun values -> print_endline (format_message fmt values)) args

let log ?(level = "INFO") fmt args =
  callback
    (fun values ->
      prerr_endline (Printf.sprintf "%s: %s" level (format_message fmt values)))
    args

let default_breakpoint values =
  List.iteri
    (fun i v -> prerr_endline (Printf.sprintf "%d: %s" i (render v)))
    values

let breakpoint_handler = ref default_breakpoint
let set_breakpoint_handler f = breakpoint_handler := f
let breakpoint args = callback (fun values -> !breakpoint_handler values) args

let inspect_array_sharding v ~callback =
  match v with
  | Tracer _ -> ()
  | Concrete _ | Device _ ->
      callback (Spmd.sharding_to_string (Spmd.array_sharding v))

let visualize_sharding ?scale ?min_width ?max_width shape sharding =
  print_string
    (Spmd.visualize_sharding_string ?scale ?min_width ?max_width shape sharding)

let visualize_array_sharding ?scale ?min_width ?max_width v =
  visualize_sharding ?scale ?min_width ?max_width (Core.get_aval v).shape
    (Spmd.array_sharding v)
