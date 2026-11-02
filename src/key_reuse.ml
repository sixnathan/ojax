open Types

type cell = {
  typed : bool;
  batch : int array;
  mutable consumed : bool array;
  birth : int;
}

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
let generation = ref 0
let journal : (cell * bool array) list ref list ref = ref []

let fresh_cell ?(consumed = false) ?(typed = true) v =
  let batch = batch_of v in
  {
    typed;
    batch;
    consumed = Array.make (size batch) consumed;
    birth = !generation;
  }

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
  (match !journal with
  | j :: _ -> j := (cell, Array.copy cell.consumed) :: !j
  | [] -> ());
  match mask with
  | Some m when Array.length m = Array.length cell.consumed ->
      cell.consumed <- Array.map2 ( || ) m cell.consumed
  | _ -> cell.consumed <- Array.map (fun _ -> true) cell.consumed

let region_stack : (int * (cell * value) list ref) list ref = ref []

let enter_closure_view cell v =
  match (!region_stack, !journal) with
  | (start, touched) :: _, log :: _
    when cell.birth < start
         && not (List.exists (fun (c, _) -> c == cell) !touched) ->
      touched := (cell, v) :: !touched;
      log := (cell, Array.copy cell.consumed) :: !log;
      cell.consumed <- Array.map (fun _ -> false) cell.consumed
  | _ -> ()

let sink ~eager_name ~static_name ?mask index v =
  if checking () then
    match typed_cell v with
    | None -> ()
    | Some cell ->
        enter_closure_view cell v;
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
      register out { typed = true; batch; consumed; birth = !generation }

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
    let prim_name =
      match prim with
      | Xla_call _ -> "jit"
      | _ -> Pretty_printer.primitive_name prim
    in
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

type out_rule =
  | Out_raw
  | Out_source
  | Out_consumed
  | Out_forward of int
  | Out_share of cell

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

let instrument_from ?start signature kinds f tracers =
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
            | None -> (
                match start with
                | Some s when cell.birth < s -> Out_share cell
                | _ -> if any_consumed cell then Out_consumed else Out_source)))
      outs;
  outs

let instrument signature kinds f tracers =
  instrument_from signature kinds f tracers

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
        | Out_share cell -> forward ~share:false cell out
        | Out_forward i -> (
            match typed_cell (List.nth args i) with
            | Some cell -> forward ~share:false cell out
            | None -> source out))
      signature.out_rules outs;
    outs)

type region = { r_signature : signature; r_closure : (value * bool array) list }

let moved_slots : bool list option ref list ref = ref []

let note_moved flags =
  match !moved_slots with
  | ({ contents = None } as slot) :: _ -> slot := Some flags
  | _ -> ()

let with_moved_slot run =
  let slot = ref None in
  moved_slots := slot :: !moved_slots;
  let outs =
    Fun.protect ~finally:(fun () -> moved_slots := List.tl !moved_slots) run
  in
  (outs, !slot)

let traced_region ~level holder kinds f tracers =
  let saved = !suspension in
  suspension := level;
  incr generation;
  let start = !generation in
  let log = ref [] in
  let touched = ref [] in
  journal := log :: !journal;
  region_stack := (start, touched) :: !region_stack;
  let signature = new_signature () in
  Fun.protect
    ~finally:(fun () ->
      journal := List.tl !journal;
      region_stack := List.tl !region_stack;
      suspension := saved;
      List.iter (fun (c, old) -> c.consumed <- old) !log)
    (fun () ->
      let outs = instrument_from ~start signature kinds f tracers in
      let closure =
        List.rev_map
          (fun ((c : cell), v) -> (v, Array.copy c.consumed))
          !touched
      in
      holder := Some { r_signature = signature; r_closure = closure };
      outs)

let sink_operands ~name ~offset operands sinks =
  List.iter
    (fun (i, mask) ->
      sink ~eager_name:name ~static_name:name ~mask (i + offset)
        (List.nth operands i))
    sinks

let apply_out_rule operands rule out =
  match rule with
  | Out_raw -> ()
  | Out_source -> source out
  | Out_consumed -> consumed_out out
  | Out_share cell -> forward ~share:false cell out
  | Out_forward i -> (
      match typed_cell (List.nth operands i) with
      | Some cell -> forward ~share:false cell out
      | None -> source out)

let cond_regions branches operands run =
  if idle () then run branches
  else
    let level = !suspension in
    let kinds = arg_kinds operands in
    let holders = List.map (fun _ -> ref None) branches in
    let wrapped =
      List.map2 (fun h b -> traced_region ~level h kinds b) holders branches
    in
    let outs = suspended (fun () -> run wrapped) in
    let regions = List.filter_map (fun h -> !h) holders in
    if List.length regions <> List.length branches then outs
    else
      let merged =
        List.fold_left
          (fun acc r ->
            List.fold_left
              (fun acc (i, mask) ->
                match List.assoc_opt i acc with
                | Some m ->
                    (i, Array.map2 ( || ) m mask) :: List.remove_assoc i acc
                | None -> (i, mask) :: acc)
              acc r.r_signature.sinks)
          [] regions
      in
      let closures =
        List.fold_left
          (fun acc r ->
            List.fold_left
              (fun acc (v, mask) ->
                let same w =
                  match (typed_cell v, typed_cell w) with
                  | Some a, Some b -> a == b
                  | _ -> false
                in
                if List.exists (fun (w, _) -> same w) acc then
                  List.map
                    (fun (w, m) ->
                      if same w then (w, Array.map2 ( || ) m mask) else (w, m))
                    acc
                else acc @ [ (v, mask) ])
              acc r.r_closure)
          [] regions
      in
      List.iteri
        (fun j (v, mask) ->
          sink ~eager_name:"cond" ~static_name:"cond" ~mask (j + 1) v)
        closures;
      sink_operands ~name:"cond"
        ~offset:(1 + List.length closures)
        operands (List.sort compare merged);
      let rules = List.map (fun r -> r.r_signature.out_rules) regions in
      List.iteri
        (fun j out ->
          let at = List.map (fun l -> List.nth l j) rules in
          let first = List.hd at in
          let same =
            List.for_all
              (fun r ->
                match (r, first) with
                | Out_forward a, Out_forward b -> a = b
                | Out_share a, Out_share b -> a == b
                | _ -> false)
              at
          in
          if same then apply_out_rule operands first out
          else if List.exists (fun r -> r = Out_source) at then source out
          else if List.for_all (fun r -> r = Out_raw) at then ()
          else consumed_out out)
        outs;
      outs

let signature_repr ~moved ?(body = true) r =
  let rules = r.r_signature.out_rules in
  let is_moved i =
    match List.nth_opt moved i with Some m -> m | None -> false
  in
  let count_before pred i =
    List.length (List.filter pred (List.init i Fun.id))
  in
  let n_carry = List.length moved in
  let n_moved = List.length (List.filter Fun.id moved) in
  let n_closure = List.length r.r_closure in
  let map_in i =
    if not body then i - count_before is_moved (min i n_carry) + n_closure
    else if is_moved i then count_before is_moved i
    else n_moved + n_closure + (i - count_before is_moved (min i n_carry))
  in
  let map_out j = j - count_before is_moved (min j n_carry) in
  let sinks =
    List.mapi (fun j _ -> j + if body then n_moved else 0) r.r_closure
    @ List.map (fun (i, _) -> map_in i) r.r_signature.sinks
  in
  let sources =
    List.concat
      (List.mapi
         (fun j rule ->
           match rule with
           | Out_source when not (is_moved j) -> [ map_out j ]
           | _ -> [])
         rules)
  in
  let forwards =
    List.concat
      (List.mapi
         (fun j rule ->
           match rule with
           | Out_forward i when not (is_moved j) -> [ (map_in i, map_out j) ]
           | _ -> [])
         rules)
  in
  let items =
    List.map (Printf.sprintf "Sink(%d)") (List.sort compare sinks)
    @ List.map (Printf.sprintf "Source(%d)") (List.sort compare sources)
    @ List.map
        (fun (a, b) -> Printf.sprintf "Forward(%d, %d)" a b)
        (List.sort compare forwards)
  in
  match items with
  | [ one ] -> Printf.sprintf "KeyReuseSignature(%s,)" one
  | l -> Printf.sprintf "KeyReuseSignature(%s)" (String.concat ", " l)

let finish_loop ~name operands r outs =
  sink_operands ~name ~offset:0 operands r.r_signature.sinks;
  List.iter2 (apply_out_rule operands) r.r_signature.out_rules outs;
  outs

let moved_of ~num_carry slot =
  match slot with
  | Some flags when List.length flags = num_carry -> flags
  | _ -> List.init num_carry (fun _ -> false)

let scan_region ~num_carry body operands run =
  if idle () then run body
  else
    let level = !suspension in
    let holder = ref None in
    let wrapped = traced_region ~level holder (arg_kinds operands) body in
    let outs, slot =
      with_moved_slot (fun () -> suspended (fun () -> run wrapped))
    in
    let moved = moved_of ~num_carry slot in
    match !holder with
    | None -> outs
    | Some r ->
        let sinks = r.r_signature.sinks in
        let is_moved i =
          match List.nth_opt moved i with Some m -> m | None -> false
        in
        if not (checking ()) then finish_loop ~name:"scan" operands r outs
        else if
          r.r_closure <> [] || List.exists (fun (i, _) -> is_moved i) sinks
        then
          raise
            (Errors.key_reuse_error
               (Printf.sprintf
                  "scan body function leads to key reuse when repeatedly \
                   executed, because key constants are repeatedly consumed:\n\
                  \  signature=%s"
                  (signature_repr ~moved r)))
        else if
          not
            (List.for_all
               (fun (i, _) ->
                 i >= num_carry
                 ||
                 match List.nth_opt r.r_signature.out_rules i with
                 | Some Out_source -> true
                 | _ -> false)
               sinks)
        then
          raise
            (Errors.key_reuse_error
               (Printf.sprintf
                  "scan body function leads to key reuse when repeatedly \
                   executed, because consumed inputs don't match sourced \
                   outputs:\n\
                  \  signature=%s"
                  (signature_repr ~moved r)))
        else finish_loop ~name:"scan" operands r outs

let while_regions cond_fun body operands run =
  if idle () then run cond_fun body
  else
    let level = !suspension in
    let kinds = arg_kinds operands in
    let cond_holder = ref None and body_holder = ref None in
    let cond_wrapped = traced_region ~level cond_holder kinds cond_fun in
    let body_wrapped = traced_region ~level body_holder kinds body in
    let outs, slot =
      with_moved_slot (fun () ->
          suspended (fun () -> run cond_wrapped body_wrapped))
    in
    let num_carry = List.length operands in
    let moved = moved_of ~num_carry slot in
    match (!cond_holder, !body_holder) with
    | Some c, Some b ->
        if not (checking ()) then finish_loop ~name:"while" operands b outs
        else
          let cond_repr = signature_repr ~moved ~body:false c in
          let body_repr = signature_repr ~moved b in
          let fail which repr =
            raise
              (Errors.key_reuse_error
                 (Printf.sprintf
                    "while_loop %s function leads to key reuse when repeatedly \
                     executed:   %s_signature=%s"
                    which which repr))
          in
          let is_moved i =
            match List.nth_opt moved i with Some m -> m | None -> false
          in
          let sourced (r : region) =
            List.for_all
              (fun (i, _) ->
                match List.nth_opt b.r_signature.out_rules i with
                | Some Out_source -> true
                | _ -> false)
              r.r_signature.sinks
          in
          if c.r_closure <> [] then fail "cond" cond_repr
          else if
            b.r_closure <> []
            || List.exists (fun (i, _) -> is_moved i) b.r_signature.sinks
          then fail "body" body_repr
          else if not (sourced c) then fail "cond" cond_repr
          else if not (sourced b) then fail "body" body_repr
          else if
            List.exists
              (fun (i, _) -> List.mem_assoc i c.r_signature.sinks)
              b.r_signature.sinks
          then
            raise
              (Errors.key_reuse_error
                 (Printf.sprintf
                    "while_loop cond and body functions both use the same \
                     key:   cond_signature=%s\n\
                    \  body_signature=%s"
                    cond_repr body_repr))
          else finish_loop ~name:"while" operands b outs
    | _ -> outs
