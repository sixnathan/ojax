let canonicalize_axis axis num_dims =
  if axis < -num_dims || axis >= num_dims then
    Py_exn.value_error
      (Printf.sprintf "axis %d is out of bounds for array of dimension %d" axis
         num_dims);
  if axis < 0 then axis + num_dims else axis

let python_tuple shape =
  match Array.to_list shape with
  | [ d ] -> Printf.sprintf "(%d,)" d
  | dims -> "(" ^ String.concat ", " (List.map string_of_int dims) ^ ")"

let canonicalize_axis_tuple ?(allow_duplicate = false) axes ndim =
  let axes = Array.map (fun a -> canonicalize_axis a ndim) axes in
  let distinct = List.sort_uniq compare (Array.to_list axes) in
  if (not allow_duplicate) && List.length distinct <> Array.length axes then
    Py_exn.value_error ("repeated axis: " ^ python_tuple axes);
  axes

let tuple_index tuple i =
  let n = Array.length tuple in
  if i < -n || i >= n then Py_exn.index_error "tuple index out of range";
  tuple.(if i < 0 then i + n else i)

let rfft_empty_verifier_message ~function_name ~location ~element shape =
  let tensor dims =
    let dims = Array.to_list dims |> List.map string_of_int in
    Printf.sprintf "tensor<%sx%s>" (String.concat "x" dims) element
  in
  let result = Array.copy shape in
  result.(Array.length shape - 1) <- 1;
  let returned = tensor shape in
  Printf.sprintf
    "Cannot lower jaxpr with verifier errors:\n\
     \ttype of return operand 0 ('%s') doesn't match function result type \
     ('%s') in function @%s\n\
     \t\tat %s\n\
     \tsee current operation: \"func.return\"(%%0) : (%s) -> ()\n\
     \t\tat %s\n\
     Define JAX_DUMP_IR_TO to dump the module."
    returned (tensor result) function_name location returned location

let unpack_last_two shape =
  let n = Array.length shape in
  if n < 2 then
    Py_exn.value_error
      (Printf.sprintf "not enough values to unpack (expected 2, got %d)" n);
  (shape.(n - 2), shape.(n - 1))

let unpack_exact_two shape =
  match shape with
  | [| m; n |] -> (m, n)
  | _ when Array.length shape > 2 ->
      Py_exn.value_error "too many values to unpack (expected 2)"
  | _ ->
      Py_exn.value_error
        (Printf.sprintf "not enough values to unpack (expected 2, got %d)"
           (Array.length shape))

let list_assign_index length i =
  if i < -length || i >= length then
    Py_exn.index_error "list assignment index out of range";
  if i < 0 then i + length else i

let slice_indices ?(start = 0) ?stop length =
  let clamp v = if v < 0 then max (v + length) 0 else min v length in
  (clamp start, match stop with None -> length | Some v -> clamp v)

let require_ndim_at_least ndim ~min message =
  if ndim < min then Py_exn.value_error message

let require_ndim ndim ~ndim:wanted message =
  if ndim <> wanted then Py_exn.value_error message

let python_shapes shapes =
  "[" ^ String.concat ", " (List.map python_tuple shapes) ^ "]"

let broadcast_shapes shapes =
  let rank = List.fold_left (fun acc s -> max acc (Array.length s)) 0 shapes in
  let padded =
    List.map
      (fun s -> Array.append (Array.make (rank - Array.length s) 1) s)
      shapes
  in
  Array.init rank (fun i ->
      let sizes = List.map (fun s -> s.(i)) padded in
      let others = List.filter (fun d -> d <> 1) sizes in
      match List.sort_uniq compare others with
      | [] -> 1
      | [ d ] -> d
      | _ ->
          Py_exn.value_error
            ("Incompatible shapes for broadcasting: shapes="
           ^ python_shapes shapes))

let lax_broadcast_shapes ~name shapes =
  let shapes = List.filter (fun s -> Array.length s > 0) shapes in
  match shapes with
  | [] -> [||]
  | first :: _ ->
      let ranks = List.sort_uniq compare (List.map Array.length shapes) in
      if List.length ranks <> 1 then broadcast_shapes shapes
      else
        Array.init (Array.length first) (fun i ->
            match
              List.sort_uniq compare
                (List.filter
                   (fun d -> d <> 1)
                   (List.map (fun s -> s.(i)) shapes))
            with
            | [] -> 1
            | [ d ] -> d
            | _ ->
                Py_exn.type_error
                  (Printf.sprintf
                     "%s got incompatible shapes for broadcasting: %s." name
                     (String.concat ", " (List.map python_tuple shapes))))

let require_same_shape a b message = if a <> b then Py_exn.value_error message
