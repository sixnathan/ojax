let default_sink category message =
  prerr_endline (Printf.sprintf "%s: %s" category message)

let sink : (string -> string -> unit) ref = ref default_sink
let warn ?(category = "UserWarning") message = !sink category message

let catch f =
  let saved = !sink in
  let seen = ref [] in
  (sink := fun category message -> seen := (category, message) :: !seen);
  match f () with
  | result ->
      sink := saved;
      (result, List.rev !seen)
  | exception e ->
      sink := saved;
      raise e
