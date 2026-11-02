open Types

type t = Ref_state.cell
type kind = Ref_state.kind = Default | No_grad_no_remat

type index_item =
  | At of int
  | Range of int option * int option
  | All
  | Dynamic of value

let stopped kind v =
  match kind with
  | Default -> v
  | No_grad_no_remat -> Custom_derivatives.stop_gradient v

let new_ref ?(kind = Default) v = Ref_state.make_cell ~kind (stopped kind v)

let empty_ref (a : aval) =
  if not (Ref_state.in_trace ()) then
    failwith "Evaluation rule for 'empty_ref' not implemented";
  Ref_state.make_cell ~kind:Default
    (Ad_util.zeros_like_aval { a with weak_type = false })

let free_ref cell = Ref_state.consume ~operation:"free_ref" cell

let freeze (cell : t) =
  let v = Ref_state.read ~operation:"freeze" cell in
  Ref_state.consume ~operation:"freeze" cell;
  v

let aval (cell : t) = cell.cell_aval
let shape cell = (aval cell).shape
let dtype cell = (aval cell).dtype
let ndim cell = Array.length (shape cell)
let size cell = Array.fold_left ( * ) 1 (shape cell)
let to_string = Ref_state.cell_string

type resolved = {
  starts : [ `Static of int | `Traced of value ] array;
  sizes : int array;
  dropped : bool array;
}

let shape_string s =
  match Array.to_list s with
  | [] -> "()"
  | [ d ] -> Printf.sprintf "(%d,)" d
  | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"

let traced_start n v =
  let i = Core.bind1 (Convert_element_type Dtype.I32) [ v ] in
  let scalar x =
    Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int x |])
  in
  let negative = Core.bind1 Lt [ i; scalar 0 ] in
  let wrapped = Core.bind1 Add [ i; scalar n ] in
  Core.bind1 Select_n [ negative; i; wrapped ]

let resolve shape index =
  let rank = Array.length shape in
  if List.length index > rank then
    raise
      (Py_exn.Index_error
         (Printf.sprintf
            "Too many indices: array is %d-dimensional, but %d were indexed"
            rank (List.length index)));
  let starts = Array.make rank (`Static 0) in
  let sizes = Array.copy shape in
  let dropped = Array.make rank false in
  List.iteri
    (fun d item ->
      let n = shape.(d) in
      match item with
      | All -> ()
      | At i ->
          let j = if i < 0 then i + n else i in
          if j < 0 || j >= n then
            invalid_arg
              (Printf.sprintf "Out of bound indexer: idx=%d, dim=%d." i n);
          starts.(d) <- `Static j;
          sizes.(d) <- 1;
          dropped.(d) <- true
      | Range (lo, hi) ->
          let clamp x =
            let x = if x < 0 then x + n else x in
            max 0 (min n x)
          in
          let a = clamp (Option.value lo ~default:0) in
          let b = clamp (Option.value hi ~default:n) in
          starts.(d) <- `Static a;
          sizes.(d) <- max 0 (b - a)
      | Dynamic v ->
          starts.(d) <- `Traced (traced_start n v);
          sizes.(d) <- 1;
          dropped.(d) <- true)
    index;
  { starts; sizes; dropped }

let kept_shape r =
  Array.of_list
    (List.filteri (fun d _ -> not r.dropped.(d)) (Array.to_list r.sizes))

let start_values r =
  Array.to_list
    (Array.map
       (function
         | `Static i ->
             Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int i |])
         | `Traced v -> v)
       r.starts)

let read_at content r =
  let block =
    Core.bind1
      (Dynamic_slice { slice_sizes = r.sizes })
      (content :: start_values r)
  in
  Core.bind1 (Reshape (kept_shape r)) [ block ]

let write_at content r v =
  let block = Core.bind1 (Reshape r.sizes) [ v ] in
  Core.bind1 Dynamic_update_slice (content :: block :: start_values r)

let check_value ~operation (cell : t) r v =
  let a = Core.get_aval v in
  let expected = kept_shape r in
  if a.dtype <> cell.cell_aval.dtype then
    invalid_arg
      (Printf.sprintf "Invalid dtype for `%s`. Ref dtype: %s. Value dtype: %s."
         operation
         (Ref_state.dtype_name cell.cell_aval.dtype)
         (Ref_state.dtype_name a.dtype));
  if a.shape <> expected then
    invalid_arg
      (Printf.sprintf
         "Invalid shape for `%s`. Ref shape: %s. Expected shape: %s. Value \
          shape: %s."
         operation
         (shape_string cell.cell_aval.shape)
         (shape_string expected) (shape_string a.shape))

let get ?(index = []) cell =
  let content = Ref_state.read cell in
  if index = [] then content else read_at content (resolve (shape cell) index)

let swap ?(index = []) (cell : t) v =
  let content = Ref_state.read ~operation:"swap" cell in
  let r = resolve (shape cell) index in
  check_value ~operation:"swap" cell r v;
  let v = stopped cell.kind v in
  if index = [] then begin
    Ref_state.write cell v;
    content
  end
  else begin
    let old = read_at content r in
    Ref_state.write cell (write_at content r v);
    old
  end

let set ?index cell v = ignore (swap ?index cell v)

let addupdate ?(index = []) (cell : t) v =
  let content = Ref_state.read ~operation:"addupdate" cell in
  let r = resolve (shape cell) index in
  check_value ~operation:"addupdate" cell r v;
  let v = stopped cell.kind v in
  let operation = "an addupdate" in
  if index = [] then
    Ref_state.write ~operation cell (Core.bind1 Add [ content; v ])
  else
    let sum = Core.bind1 Add [ read_at content r; v ] in
    Ref_state.write ~operation cell (write_at content r sum)
