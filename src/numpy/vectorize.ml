module T = Types
module C = Core
module LN = Lax_numpy

let get_aval = C.get_aval
let shape_of v = (get_aval v).T.shape

let parse_side (s : string) : string list list =
  let n = String.length s in
  let groups = ref [] in
  let i = ref 0 in
  while !i < n do
    if s.[!i] = '(' then (
      let j = String.index_from s !i ')' in
      let inner = String.sub s (!i + 1) (j - !i - 1) in
      let dims =
        if String.trim inner = "" then []
        else List.map String.trim (String.split_on_char ',' inner)
      in
      groups := dims :: !groups;
      i := j + 1)
    else incr i
  done;
  List.rev !groups

let input_core_dims (s : string) : string list list =
  match String.index_opt s '-' with
  | Some k -> parse_side (String.sub s 0 k)
  | None -> []

let python_core_dims = function
  | [ d ] -> Printf.sprintf "('%s',)" d
  | dims ->
      "(" ^ String.concat ", " (List.map (Printf.sprintf "'%s'") dims) ^ ")"

let check_core_dims excluded signature names shapes =
  let context =
    Printf.sprintf "on vectorized function with excluded=%s and signature='%s'"
      (match excluded with
      | [] -> "frozenset()"
      | l ->
          "frozenset({" ^ String.concat ", " (List.map string_of_int l) ^ "})")
      signature
  in
  let sizes = Hashtbl.create 4 in
  List.iter2
    (fun core_dims shape ->
      let rank = Array.length shape and count = List.length core_dims in
      if rank < count then
        Py_exn.value_error
          (Printf.sprintf
             "input with shape %s does not have enough dimensions for all core \
              dimensions %s %s"
             (Validate.python_tuple shape)
             (python_core_dims core_dims)
             context);
      List.iteri
        (fun i dim ->
          let size = shape.(rank - count + i) in
          match Hashtbl.find_opt sizes dim with
          | None -> Hashtbl.add sizes dim size
          | Some seen when seen <> size ->
              Py_exn.value_error
                (Printf.sprintf
                   "inconsistent size for core dimension '%s': %d vs %d %s" dim
                   size seen context)
          | Some _ -> ())
        core_dims)
    names shapes

let parse_signature (s : string) : int list * int list =
  let arrow =
    match String.index_opt s '-' with
    | Some k when k + 1 < String.length s && s.[k + 1] = '>' -> k
    | _ -> Py_exn.value_error ("not a valid gufunc signature: " ^ s)
  in
  let lhs = String.sub s 0 arrow in
  let rhs = String.sub s (arrow + 2) (String.length s - arrow - 2) in
  (List.map List.length (parse_side lhs), List.map List.length (parse_side rhs))

let check_rank_promotion ~signature shapes core_ndims =
  match Config.value Config.numpy_rank_promotion with
  | "allow" -> ()
  | mode ->
      let ranks =
        List.filter_map
          (fun (sh, nc) ->
            if Array.length sh = 0 then None else Some (Array.length sh - nc))
          (List.combine shapes core_ndims)
      in
      if List.length (List.sort_uniq compare ranks) > 1 then begin
        let msg =
          Printf.sprintf
            "operands with shapes [%s] require rank promotion for \
             jnp.vectorize function with signature %s. Set the \
             jax_numpy_rank_promotion config option to 'allow' to disable this \
             message; for more information, see \
             https://docs.jax.dev/en/latest/rank_promotion_warning.html."
            (String.concat ", " (List.map Validate.python_tuple shapes))
            (Option.value signature ~default:"None")
        in
        if mode = "warn" then Py_warnings.warn msg
        else if mode = "raise" then Py_exn.value_error msg
      end

let vectorize_many ?(excluded = []) ?signature pyfunc raw_args =
  let n = List.length raw_args in
  let args_arr = Array.of_list raw_args in
  let is_excl i = List.mem i excluded in
  let dynamic = List.filteri (fun i _ -> not (is_excl i)) raw_args in
  let call_full dyn_slices =
    let dyn = ref dyn_slices in
    let full =
      List.init n (fun i ->
          if is_excl i then args_arr.(i)
          else
            match !dyn with
            | x :: r ->
                dyn := r;
                x
            | [] -> assert false)
    in
    pyfunc full
  in
  let input_ncs, output_ncs =
    match signature with
    | Some s -> parse_signature s
    | None -> (List.map (fun _ -> 0) dynamic, [ 0 ])
  in
  let dyn_arr = Array.of_list dynamic in
  let ncs = Array.of_list input_ncs in
  let excluded_repr =
    match excluded with
    | [] -> "frozenset()"
    | items ->
        "frozenset({"
        ^ String.concat ", " (List.map string_of_int (List.sort compare items))
        ^ "})"
  in
  let error_context () =
    Printf.sprintf "on vectorized function with excluded=%s and signature=%s"
      excluded_repr
      (match signature with Some s -> "'" ^ s ^ "'" | None -> "None")
  in
  let core_dims idx =
    match signature with
    | None -> "()"
    | Some s -> python_core_dims (List.nth (input_core_dims s) idx)
  in
  if Array.length dyn_arr <> Array.length ncs then
    Py_exn.type_error
      (Printf.sprintf
         "wrong number of positional arguments: expected %d, got %d %s"
         (Array.length ncs) (Array.length dyn_arr) (error_context ()));
  ignore output_ncs;
  (match signature with
  | Some s ->
      check_core_dims excluded s (input_core_dims s) (List.map shape_of dynamic)
  | None -> ());
  let batch_shapes =
    Array.to_list
      (Array.mapi
         (fun idx v ->
           let sh = shape_of v in
           let ndim = Array.length sh in
           let nc = ncs.(idx) in
           if ndim < nc then
             Py_exn.value_error
               (Printf.sprintf
                  "input with shape %s does not have enough dimensions for all \
                   core dimensions %s %s"
                  (Validate.python_tuple sh) (core_dims idx) (error_context ()));
           Array.sub sh 0 (ndim - nc))
         dyn_arr)
  in
  let broadcast_shape = LN.broadcast_shapes_n batch_shapes in
  check_rank_promotion ~signature
    (List.map shape_of dynamic)
    (Array.to_list ncs);
  let nb = Array.length broadcast_shape in
  let pieces =
    Array.to_list
      (Array.mapi
         (fun idx v ->
           let sh = shape_of v in
           let noncore = Array.sub sh 0 (Array.length sh - ncs.(idx)) in
           let filled =
             Array.append (Array.make (nb - Array.length noncore) 1) noncore
           in
           let squeeze_axes =
             List.filter
               (fun i -> noncore.(i) = 1)
               (List.init (Array.length noncore) Fun.id)
           in
           let squeezed =
             if squeeze_axes = [] then v
             else LN.squeeze ~axis:(Array.of_list squeeze_axes) v
           in
           (filled, squeezed))
         dyn_arr)
  in
  let filled = List.map fst pieces and squeezed = List.map snd pieces in
  let func = ref call_full and expand = ref [] in
  for pos = nb - 1 downto 0 do
    let in_axes =
      List.map (fun s -> if s.(pos) = 1 then None else Some 0) filled
    in
    if List.for_all Option.is_none in_axes then expand := pos :: !expand
    else begin
      let inner = !func in
      func := fun xs -> Interpreters.Batching.vmap inner in_axes xs
    end
  done;
  let results = !func squeezed in
  match !expand with
  | [] -> results
  | dims -> List.map (fun r -> LN.expand_dims r (Array.of_list dims)) results

let vectorize ?excluded ?signature pyfunc raw_args =
  (match signature with
  | Some s when List.length (snd (parse_signature s)) <> 1 ->
      Py_exn.not_implemented_error
        "vectorize: only single-output gufuncs supported"
  | _ -> ());
  List.hd
    (vectorize_many ?excluded ?signature (fun xs -> [ pyfunc xs ]) raw_args)
