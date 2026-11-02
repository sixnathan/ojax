open Types

type t = {
  aliases : int option array;
  buffer_donor : bool array;
  unusable : bool array;
}

let size (a : aval) = Array.fold_left ( * ) 1 a.shape

let set_up_aliases (avals_in : aval list) (avals_out : aval list)
    (donated : bool list) =
  let avals_in = Array.of_list avals_in in
  let avals_out = Array.of_list avals_out in
  let n = Array.length avals_in in
  let aliases = Array.make n None in
  let unusable = Array.of_list donated in
  let buffer_donor = Array.make n false in
  let key (a : aval) = (a.shape, a.dtype) in
  let taken = Array.make n false in
  Array.iteri
    (fun i out ->
      let rec find j =
        if j >= n then None
        else if unusable.(j) && (not taken.(j)) && key avals_in.(j) = key out
        then Some j
        else find (j + 1)
      in
      match find 0 with
      | Some j ->
          taken.(j) <- true;
          unusable.(j) <- false;
          aliases.(j) <- Some i
      | None -> ())
    avals_out;
  let aliased =
    Array.to_list aliases |> List.filter_map Fun.id |> List.sort_uniq compare
  in
  let not_matched =
    Array.to_list (Array.mapi (fun i a -> (i, a)) avals_out)
    |> List.filter (fun (i, _) -> not (List.mem i aliased))
    |> List.map (fun (i, a) -> (i, size a))
  in
  let remaining = ref not_matched in
  for j = 0 to n - 1 do
    if unusable.(j) then
      let wanted = size avals_in.(j) in
      match List.find_opt (fun (_, s) -> s = wanted) !remaining with
      | Some (i, _) ->
          remaining := List.filter (fun (k, _) -> k <> i) !remaining;
          unusable.(j) <- false;
          buffer_donor.(j) <- true
      | None -> ()
  done;
  { aliases; buffer_donor; unusable }

let arg_attribute t i =
  match t.aliases.(i) with
  | Some k -> Some (Printf.sprintf "tf.aliasing_output = %d : i32" k)
  | None -> if t.buffer_donor.(i) then Some "jax.buffer_donor = true" else None

let warn_unusable (avals_in : aval list) t =
  let unused =
    List.filteri (fun i _ -> t.unusable.(i)) avals_in
    |> List.map Ref_state.aval_string
  in
  if unused <> [] then
    Py_warnings.warn
      (Printf.sprintf
         "Some donated buffers were not usable: %s.\n\
          See an explanation at \
          https://docs.jax.dev/en/latest/faq.html#buffer-donation."
         (String.concat ", " unused))
