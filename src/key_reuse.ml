open Types

type cell = { typed : bool; batch : int array; mutable consumed : bool array }

module Nd_table = Ephemeron.K1.Make (struct
  type t = Ndarray.t

  let equal = ( == )

  let hash nd =
    let shape = Ndarray.shape nd in
    if Ndarray.dtype nd = Dtype.Uint32 && Array.for_all (fun d -> d > 0) shape
    then
      Hashtbl.hash
        (shape, Ndarray.get_i64 nd (Array.make (Array.length shape) 0))
    else Hashtbl.hash shape
end)

module Tracer_table = Ephemeron.K1.Make (struct
  type t = tracer

  let equal = ( == )
  let hash (t : tracer) = t.id
end)

module Buffer_table = Ephemeron.K1.Make (struct
  type t = Pjrt.Buffer.t

  let equal = ( == )
  let hash _ = 0
end)

let nds : cell Nd_table.t = Nd_table.create 64
let tracers : cell Tracer_table.t = Tracer_table.create 64
let buffers : cell Buffer_table.t = Buffer_table.create 8
let any = ref false
let suspension = ref 0
let idle () = (not !any) || !suspension > 0

let suspended thunk =
  incr suspension;
  Fun.protect ~finally:(fun () -> decr suspension) thunk

let register v cell =
  any := true;
  match v with
  | Concrete nd -> Nd_table.replace nds nd cell
  | Tracer t -> Tracer_table.replace tracers t cell
  | Device b -> Buffer_table.replace buffers b cell

let direct = function
  | Concrete nd -> Nd_table.find_opt nds nd
  | Tracer t -> Tracer_table.find_opt tracers t
  | Device b -> Buffer_table.find_opt buffers b

let rec lookup v =
  match direct v with
  | Some _ as found -> found
  | None -> (
      match v with
      | Tracer { payload = Batch { v = inner; _ }; _ } -> lookup inner
      | Tracer { payload = JVP { primal; _ }; _ } -> lookup primal
      | Tracer { payload = PE { pval = { pv_const = Some c; _ }; _ }; _ } ->
          lookup c
      | _ -> None)

let rec root = function
  | Tracer { payload = Batch { v = inner; _ }; _ } -> root inner
  | Tracer { payload = JVP { primal; _ }; _ } -> root primal
  | Tracer { payload = PE { pval = { pv_const = Some c; _ }; _ }; _ } -> root c
  | v -> v

let shape_dtype = function
  | Concrete nd -> (Ndarray.shape nd, Ndarray.dtype nd)
  | Tracer t -> (t.aval.shape, t.aval.dtype)
  | Device b -> (Pjrt.Buffer.dimensions b, Pjrt.Buffer.element_type b)

let key_shaped v =
  let shape, dtype = shape_dtype v in
  let n = Array.length shape in
  dtype = Dtype.Uint32 && n >= 1 && shape.(n - 1) = 2

let batch_of v =
  let shape, _ = shape_dtype v in
  Array.sub shape 0 (Array.length shape - 1)

let size batch = Array.fold_left ( * ) 1 batch

let fresh_cell ?(consumed = false) ?(typed = true) v =
  let batch = batch_of v in
  { typed; batch; consumed = Array.make (size batch) consumed }

let typed_cell v =
  match lookup v with Some cell when cell.typed -> Some cell | _ -> None

let mark_typed v =
  if key_shaped v then register v (fresh_cell v);
  v

let mark_raw v =
  if key_shaped v then register v (fresh_cell ~typed:false v);
  v

let is_typed v = key_shaped v && Option.is_some (typed_cell v)

let known_raw v =
  key_shaped v
  &&
  match lookup v with
  | Some cell -> not cell.typed
  | None -> ( match root v with Tracer _ -> false | _ -> true)

let checking () = Config.value Config.debug_key_reuse

let is_static v =
  match root v with
  | Tracer { trace = { kind = KJaxpr; _ }; _ } -> true
  | _ -> false

let report ~eager_name ~static_name ~index v =
  let message =
    if is_static v then
      Printf.sprintf "In %s, argument %d is already consumed." static_name index
    else
      Printf.sprintf "Previously-consumed key passed to %s at index %d"
        eager_name index
  in
  raise (Errors.key_reuse_error message)

let overlaps cell mask =
  match mask with
  | None -> Array.exists Fun.id cell.consumed
  | Some m ->
      Array.length m = Array.length cell.consumed
      && Array.exists Fun.id (Array.map2 (fun a b -> a && b) m cell.consumed)

let mark cell mask =
  match mask with
  | Some m when Array.length m = Array.length cell.consumed ->
      cell.consumed <- Array.map2 ( || ) m cell.consumed
  | _ -> cell.consumed <- Array.map (fun _ -> true) cell.consumed

let sink ~eager_name ~static_name ?mask index v =
  if checking () then
    match typed_cell v with
    | None -> ()
    | Some cell ->
        if overlaps cell mask then report ~eager_name ~static_name ~index v;
        mark cell mask

let consume name v =
  let static_name = if name = "jit-compiled function" then "jit" else name in
  sink ~eager_name:name ~static_name 0 v

let any_consumed cell = Array.exists Fun.id cell.consumed

let forward ~share cell out =
  if key_shaped out then
    let batch = batch_of out in
    let same = size batch = Array.length cell.consumed in
    if same && (share || match out with Tracer _ -> true | _ -> false) then
      register out cell
    else
      let consumed =
        if same then Array.copy cell.consumed
        else Array.make (size batch) (any_consumed cell)
      in
      register out { typed = true; batch; consumed }

let source out = ignore (mark_typed out)

let consumed_out out =
  if key_shaped out then register out (fresh_cell ~consumed:true out)

let slice_mask batch ~start ~limit ~strides =
  let rank = Array.length batch in
  let n = size batch in
  let coords = Array.make rank 0 in
  Array.init n (fun flat ->
      let r = ref flat in
      for d = rank - 1 downto 0 do
        coords.(d) <- !r mod batch.(d);
        r := !r / batch.(d)
      done;
      let inside = ref true in
      for d = 0 to rank - 1 do
        let c = coords.(d) in
        if
          not
            (c >= start.(d)
            && c < limit.(d)
            && (c - start.(d)) mod strides.(d) = 0)
        then inside := false
      done;
      !inside)

let apply prim args run =
  if idle () || not (List.exists is_typed args) then run ()
  else
    let prim_name = Pretty_printer.primitive_name prim in
    let sink_arg ?mask index v =
      sink ~eager_name:prim_name ~static_name:prim_name ?mask index v
    in
    let sink_all () = List.iteri (fun i v -> sink_arg i v) args in
    let on_outs f =
      let outs = run () in
      f outs;
      outs
    in
    let head = List.hd args in
    let forward_head ~share =
      match typed_cell head with
      | None -> run ()
      | Some cell ->
          on_outs (function out :: _ -> forward ~share cell out | [] -> ())
    in
    match prim with
    | Random_bits _ ->
        sink ~eager_name:"jit-compiled function" ~static_name:"jit" 0 head;
        run ()
    | Random_split _ ->
        sink_arg 0 head;
        on_outs (List.iter source)
    | Random_fold_in ->
        if is_typed head then on_outs (List.iter source) else run ()
    | Random_wrap | Random_unwrap -> forward_head ~share:true
    | Broadcast_in_dim _ | Copy | Convert_element_type _ | Reshape _
    | Reshape_perm _ | Squeeze _ | Dynamic_slice _ | Gather _ ->
        forward_head ~share:false
    | Eq | Ne -> run ()
    | Dynamic_update_slice ->
        (match args with _ :: update :: _ -> sink_arg 1 update | _ -> ());
        forward_head ~share:false
    | Scatter _ ->
        (match args with
        | _ :: _ :: updates :: _ -> sink_arg 2 updates
        | _ -> ());
        forward_head ~share:false
    | Slice { start_indices; limit_indices; strides } ->
        let shape, _ = shape_dtype head in
        let rank = Array.length shape in
        let whole_key =
          rank > 0
          && start_indices.(rank - 1) = 0
          && limit_indices.(rank - 1) = 2
        in
        if not whole_key then run ()
        else
          let strides =
            match strides with Some s -> s | None -> Array.make rank 1
          in
          let cut a = Array.sub a 0 (rank - 1) in
          let mask =
            slice_mask (cut shape) ~start:(cut start_indices)
              ~limit:(cut limit_indices) ~strides:(cut strides)
          in
          sink_arg ~mask 0 head;
          on_outs (List.iter source)
    | Unstack _ | Split _ | Stack _ ->
        sink_all ();
        on_outs (List.iter source)
    | Concatenate _ when List.length args = 1 -> forward_head ~share:false
    | Concatenate _ ->
        sink_all ();
        on_outs (List.iter source)
    | _ ->
        sink_all ();
        on_outs (List.iter consumed_out)

type arg_kind = Kind_typed | Kind_raw | Kind_other
type out_rule = Out_raw | Out_source | Out_consumed | Out_forward of int

type signature = {
  mutable sinks : (int * bool array) list;
  mutable out_rules : out_rule list;
}

let arg_kinds args =
  List.map
    (fun v ->
      if not (key_shaped v) then Kind_other
      else if is_typed v then Kind_typed
      else if known_raw v then Kind_raw
      else Kind_other)
    args

let kinds_key kinds =
  if List.for_all (fun k -> k = Kind_other) kinds then ""
  else
    String.concat ""
      (List.map
         (function Kind_typed -> "k" | Kind_raw -> "r" | Kind_other -> "_")
         kinds)

let new_signature () = { sinks = []; out_rules = [] }

let instrument signature kinds f tracers =
  let cells =
    List.map2
      (fun kind v ->
        match kind with
        | Kind_typed ->
            let cell = fresh_cell v in
            register v cell;
            Some cell
        | Kind_raw ->
            register v (fresh_cell ~typed:false v);
            None
        | Kind_other -> None)
      kinds tracers
  in
  let outs = f tracers in
  signature.sinks <-
    List.concat
      (List.mapi
         (fun i cell ->
           match cell with
           | Some c when any_consumed c -> [ (i, Array.copy c.consumed) ]
           | _ -> [])
         cells);
  let input_index cell =
    let rec go i = function
      | [] -> None
      | Some c :: _ when c == cell -> Some i
      | _ :: rest -> go (i + 1) rest
    in
    go 0 cells
  in
  signature.out_rules <-
    List.map
      (fun out ->
        match typed_cell out with
        | None -> Out_raw
        | Some cell -> (
            match input_index cell with
            | Some i -> Out_forward i
            | None -> if any_consumed cell then Out_consumed else Out_source))
      outs;
  outs

let call_with_signature signature args run =
  if !suspension > 0 then run ()
  else (
    List.iter
      (fun (i, mask) ->
        sink ~eager_name:"jit-compiled function" ~static_name:"jit" ~mask i
          (List.nth args i))
      signature.sinks;
    let outs = suspended run in
    List.iter2
      (fun rule out ->
        match rule with
        | Out_raw -> ()
        | Out_source -> source out
        | Out_consumed -> consumed_out out
        | Out_forward i -> (
            match typed_cell (List.nth args i) with
            | Some cell -> forward ~share:false cell out
            | None -> source out))
      signature.out_rules outs;
    outs)
