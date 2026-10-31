open Types

type kind = Default | No_grad_no_remat

type cell = {
  cell_id : int;
  mutable content : value;
  cell_aval : aval;
  level : int;
  depth : int;
  kind : kind;
  mutable consumed : bool;
}

type mode = Bare | Lift of (aval -> value) | Forbid of string
type entry = { cell : cell; saved : value; was_consumed : bool; lifted : bool }
type scope = { mode : mode; mutable entries : entry list }

let scopes : scope list ref = ref []
let current_depth () = List.length !scopes

let make_cell ~kind content =
  {
    cell_id = Core.fresh_id ();
    content;
    cell_aval = Core.get_aval content;
    level = Core.current_level ();
    depth = current_depth ();
    kind;
    consumed = false;
  }

let in_trace () = !scopes <> []
let is_local cell = cell.depth >= current_depth ()
let tracked scope cell = List.exists (fun e -> e.cell == cell) scope.entries

let add_entry scope cell ~lifted saved =
  scope.entries <-
    { cell; saved; was_consumed = cell.consumed; lifted } :: scope.entries

let touch cell =
  match !scopes with
  | scope :: _ when (not (is_local cell)) && not (tracked scope cell) -> (
      match scope.mode with
      | Forbid message -> failwith message
      | Bare -> add_entry scope cell ~lifted:false cell.content
      | Lift new_arg ->
          add_entry scope cell ~lifted:true cell.content;
          cell.content <- new_arg cell.cell_aval)
  | _ -> ()

let pin cells values =
  match !scopes with
  | [] -> failwith "Ref: pin outside a trace"
  | scope :: _ ->
      List.iter2
        (fun cell v ->
          if not (tracked scope cell) then
            add_entry scope cell ~lifted:false cell.content;
          cell.content <- v)
        cells values

let with_scope mode f =
  let scope = { mode; entries = [] } in
  scopes := scope :: !scopes;
  Fun.protect
    ~finally:(fun () ->
      scopes := List.tl !scopes;
      List.iter
        (fun e ->
          e.cell.content <- e.saved;
          e.cell.consumed <- e.was_consumed)
        scope.entries)
    (fun () -> f scope)

let lifted scope =
  List.rev
    (List.filter_map
       (fun e -> if e.lifted then Some e.cell else None)
       scope.entries)

let contents cells = List.map (fun cell -> cell.content) cells

let dtype_name dt =
  match dt with
  | Dtype.F16 -> "float16"
  | Dtype.Bf16 -> "bfloat16"
  | Dtype.F32 -> "float32"
  | Dtype.F64 -> "float64"
  | Dtype.I8 -> "int8"
  | Dtype.I16 -> "int16"
  | Dtype.I32 -> "int32"
  | Dtype.I64 -> "int64"
  | Dtype.Uint8 -> "uint8"
  | Dtype.Uint16 -> "uint16"
  | Dtype.Uint32 -> "uint32"
  | Dtype.Uint64 -> "uint64"
  | Dtype.Bool -> "bool"
  | Dtype.Complex64 -> "complex64"
  | Dtype.Complex128 -> "complex128"

let aval_string (a : aval) =
  Printf.sprintf "%s%s[%s]"
    (if a.weak_type then "~" else "")
    (dtype_name a.dtype)
    (String.concat "," (Array.to_list (Array.map string_of_int a.shape)))

let cell_string cell = "Ref{" ^ aval_string cell.cell_aval ^ "}"

let check_writable ~operation cell v =
  match v with
  | Tracer { trace = { kind = KJVP; level; _ }; _ } when level > cell.level ->
      failwith
        (Printf.sprintf
           "performing %s operation with a differentiated value on a \
            non-differentiated array reference of type %s. Move the array \
            reference to be an argument of the differentiated function?"
           operation (cell_string cell))
  | Tracer { trace = { kind = KBatch; level; _ }; _ } when level > cell.level ->
      failwith
        (Printf.sprintf
           "performing %s operation with vmapped value on an unbatched array \
            reference of type %s. Move the array reference to be an argument \
            to the vmapped function?"
           operation (cell_string cell))
  | _ -> ()

let check_live ~operation cell =
  if cell.consumed then
    invalid_arg
      (Printf.sprintf "`%s` must be called on `Ref` types: %s." operation
         (aval_string cell.cell_aval))

let read ?(operation = "get") cell =
  touch cell;
  check_live ~operation cell;
  cell.content

let write ?(operation = "a set/swap") cell v =
  touch cell;
  check_writable ~operation cell v;
  cell.content <- v

let consume ~operation cell =
  if in_trace () then begin
    if not (is_local cell) then
      invalid_arg
        (Printf.sprintf
           "%s of a Ref created outside the traced function is not supported"
           operation);
    check_live ~operation cell;
    cell.consumed <- true
  end
