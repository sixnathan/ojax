module C = Core
module T = Types
module D = Dtype
module L = Lax_api
module LC = Lax_config

type index =
  | Int of int
  | Slice of { start : int option; stop : int option; step : int option }
  | Newaxis
  | Ellipsis
  | Arr of T.value
  | Bool of bool

type typ = Tnone | Tslice | Tellipsis | Tinteger | Tboolean | Tarray

type item =
  | User of index
  | Host of Ndarray.t
  | Jarr of T.value
  | Pint of int
  | Pbool of bool

type parsed = { item : item; typ : typ; axes : int list }

let shape v = (C.get_aval v).T.shape
let dtype v = (C.get_aval v).T.dtype
let ndim v = Array.length (shape v)
let is_integer dt = Dtypes.issubdtype dt `Integer
let is_unsigned dt = Dtypes.issubdtype dt `Unsigned
let prod = Array.fold_left ( * ) 1
let index_error s = raise (Py_exn.Index_error s)

let py_tuple l =
  match l with
  | [ x ] -> "(" ^ string_of_int x ^ ",)"
  | _ -> "(" ^ String.concat ", " (List.map string_of_int l) ^ ")"

let py_list l = "[" ^ String.concat ", " (List.map string_of_int l) ^ "]"

let index_type = function
  | Newaxis -> Tnone
  | Ellipsis -> Tellipsis
  | Slice _ -> Tslice
  | Int _ -> Tinteger
  | Bool _ -> Tboolean
  | Arr v ->
      let dt = dtype v in
      if dt = D.Bool then Tboolean
      else if is_integer dt then Tarray
      else
        raise
          (Py_exn.Type_error
             ("Indexer must have integer or boolean type, got indexer with \
               type " ^ D.numpy_name dt))

let item_ndim = function
  | User (Arr v) | Jarr v -> ndim v
  | Host nd -> Array.length (Ndarray.shape nd)
  | _ -> 0

let canon = function
  | T.Concrete nd ->
      T.Concrete
        (Ndarray.canonicalize (Dtypes.canonicalize_dtype (Ndarray.dtype nd)) nd)
  | v -> v

let parse indices sh =
  let indices = List.map (function Arr v -> Arr (canon v) | i -> i) indices in
  let rank = Array.length sh in
  let types = List.map index_type indices in
  let consumed =
    List.map2
      (fun idx typ ->
        match typ with
        | Tnone | Tellipsis -> 0
        | Tboolean -> item_ndim (User idx)
        | _ -> 1)
      indices types
  in
  let ellipses = List.filteri (fun _ t -> t = Tellipsis) types |> List.length in
  if ellipses > 1 then
    index_error "an index can only have a single ellipsis ('...')";
  let total = List.fold_left ( + ) 0 consumed in
  if total > rank then
    index_error
      (Printf.sprintf
         "Too many indices: array is %d-dimensional, but %d were indexed" rank
         total);
  let consumed =
    List.map2
      (fun t n -> if t = Tellipsis then rank - total else n)
      types consumed
  in
  let cur = ref 0 in
  List.map2
    (fun (idx, typ) n ->
      let axes = List.init n (fun i -> !cur + i) in
      cur := !cur + n;
      { item = User idx; typ; axes })
    (List.combine indices types)
    consumed

let slice_indices size start stop step =
  let step = Option.value step ~default:1 in
  if step = 0 then Py_exn.value_error "slice step cannot be zero";
  let lower, upper = if step < 0 then (-1, size - 1) else (0, size) in
  let fix d = function
    | None -> d
    | Some v ->
        let v = if v < 0 then v + size else v in
        if v < 0 && v < lower then lower
        else if v < lower then lower
        else if v > upper then upper
        else v
  in
  let start = fix (if step < 0 then upper else lower) start in
  let stop = fix (if step < 0 then lower else upper) stop in
  (start, stop, step)

let parse_mode = function
  | None -> LC.Fill_or_drop
  | Some "clip" -> LC.Clip
  | Some ("fill" | "drop") -> LC.Fill_or_drop
  | Some "promise_in_bounds" -> LC.Promise_in_bounds
  | Some "one_hot" -> LC.One_hot
  | Some s -> Py_exn.value_error (Printf.sprintf "Unknown gather mode \"%s\"" s)

let validate_static_indices ~normalize sh parsed =
  List.iter
    (fun p ->
      match (p.typ, p.item) with
      | Tinteger, User (Int i) ->
          let axis = List.hd p.axes in
          let size = sh.(axis) in
          let normed = if normalize && i < 0 then i + size else i in
          if not (0 <= normed && normed < size) then
            index_error
              (Printf.sprintf
                 "index %d out of bounds for axis %d with size %d \
                  (normalize_indices=%s)"
                 i axis size
                 (if normalize then "True" else "False"))
      | _ -> ())
    parsed

let expand_ellipses sh parsed =
  let consumed = ref 0 in
  let full axis =
    {
      item = User (Slice { start = None; stop = None; step = None });
      typ = Tslice;
      axes = [ axis ];
    }
  in
  let out =
    List.concat_map
      (fun p ->
        consumed := !consumed + List.length p.axes;
        if p.typ = Tellipsis then List.map full p.axes else [ p ])
      parsed
  in
  out @ List.init (Array.length sh - !consumed) (fun i -> full (!consumed + i))

let rec position_type_error kind pos p =
  let ty =
    match p.item with
    | User (Arr _) | Jarr _ -> "<class 'jaxlib._jax.ArrayImpl'>"
    | Host _ -> "<class 'numpy.ndarray'>"
    | User (Bool _) | Pbool _ -> "<class 'bool'>"
    | _ ->
        ignore (position_type_error, pos);
        "<class 'object'>"
  in
  raise
    (Py_exn.Type_error
       (Printf.sprintf "%s Got index of type %s at position %d" kind ty pos))

type static_slice = {
  starts : int array;
  limits : int array;
  strides : int array option;
  rev_axes : int array;
  squeeze_axes : int array;
  newaxis_dims : int array;
}

let to_static_slice ~normalize ~mode sh parsed =
  let m = match mode with None -> LC.Promise_in_bounds | m -> parse_mode m in
  if m <> LC.Promise_in_bounds && m <> LC.Clip then
    Py_exn.value_error
      "static_slice requires mode='promise_in_bounds' or mode='clip'";
  if m = LC.Promise_in_bounds then validate_static_indices ~normalize sh parsed;
  List.iteri
    (fun pos p ->
      match p.typ with
      | Tarray | Tboolean ->
          position_type_error
            "static_slice: indices must be static scalars or slices." pos p
      | _ -> ())
    parsed;
  let starts = ref [] and limits = ref [] and strides = ref [] in
  let rev = ref [] and squeeze = ref [] and newaxis = ref [] in
  List.iter
    (fun p ->
      match (p.typ, p.item) with
      | Tnone, _ ->
          newaxis :=
            (List.length !starts - List.length !squeeze + List.length !newaxis)
            :: !newaxis
      | Tinteger, User (Int i) ->
          let axis = List.hd p.axes in
          if sh.(axis) = 0 then
            index_error
              (Printf.sprintf "index is out of bounds for axis %d with size 0"
                 axis);
          let s = if normalize && i < 0 then i + sh.(axis) else i in
          let s = min (max s 0) (sh.(axis) - 1) in
          starts := s :: !starts;
          limits := (s + 1) :: !limits;
          strides := 1 :: !strides;
          squeeze := axis :: !squeeze
      | Tslice, User (Slice { start; stop; step }) ->
          let axis = List.hd p.axes in
          let size = sh.(axis) in
          let start, stop, stride = slice_indices size start stop step in
          if stride < 0 then begin
            let new_start =
              min size (stop + 1 + (abs (start - stop - 1) mod abs stride))
            in
            starts := new_start :: !starts;
            limits := max new_start (start + 1) :: !limits;
            strides := abs stride :: !strides;
            rev := axis :: !rev
          end
          else begin
            starts := start :: !starts;
            limits := max start stop :: !limits;
            strides := stride :: !strides
          end
      | _ -> invalid_arg "static_slice")
    (expand_ellipses sh parsed);
  let a l = Array.of_list (List.rev l) in
  let strides = a !strides in
  {
    starts = a !starts;
    limits = a !limits;
    strides =
      (if Array.for_all (fun s -> s = 1) strides then None else Some strides);
    rev_axes = a !rev;
    squeeze_axes = a !squeeze;
    newaxis_dims = a !newaxis;
  }

let finish x ~rev ~squeeze ~newaxis =
  let x = if rev = [||] then x else L.rev x rev in
  let x = if squeeze = [||] then x else L.squeeze x squeeze in
  if newaxis = [||] then x else L.expand_dims x newaxis

let static_slice x s =
  let sh = shape x in
  let trivial =
    s.strides = None
    && Array.length sh = Array.length s.starts
    && Array.for_all Fun.id
         (Array.mapi (fun i st -> st = 0 && s.limits.(i) = sh.(i)) s.starts)
  in
  let x =
    if trivial then x else L.slice ?strides:s.strides x s.starts s.limits
  in
  finish x ~rev:s.rev_axes ~squeeze:s.squeeze_axes ~newaxis:s.newaxis_dims

let index_dtype sh =
  if Array.exists (fun d -> d > 2147483647) sh then D.I64 else D.I32

let py_int i = Jnp_util.py_int i
let as_strong v dt = C.convert_like ~weak:false v dt

let array_start i d =
  let dt = dtype i in
  if is_unsigned dt then i
  else
    let d_arr = T.Concrete (Ndarray.of_floats dt [||] [| float_of_int d |]) in
    L.select (Ufuncs.less i (py_int 0)) (Ufuncs.add i d_arr) i

let dynamic_slice ~normalize ~mode x parsed =
  let sh = shape x in
  (match mode with
  | None -> ()
  | m ->
      let m = parse_mode m in
      if m <> LC.Promise_in_bounds && m <> LC.Clip then
        Py_exn.value_error
          "dynamic_slice requires mode='promise_in_bounds' or mode='clip'");
  List.iteri
    (fun pos p ->
      match (p.typ, p.item) with
      | Tslice, User (Slice { step = Some s; start; stop })
        when s <> 1 && s <> -1 ->
          let o = function None -> "None" | Some v -> string_of_int v in
          raise
            (Py_exn.Type_error
               (Printf.sprintf
                  "dynamic_slice: only unit steps supported in slice. Got \
                   slice(%s, %s, %d) at position %d"
                  (o start) (o stop) s pos))
      | Tarray, _ when item_ndim p.item <> 0 ->
          position_type_error "dynamic_slice: only scalar indices allowed." pos
            p
      | Tboolean, _ ->
          position_type_error
            "dynamic_slice: indices must be scalars or slices." pos p
      | _ -> ())
    parsed;
  let starts = ref [] and sizes = ref [] and rev = ref [] in
  let squeeze = ref [] and newaxis = ref [] and trivial = ref true in
  List.iter
    (fun p ->
      match (p.typ, p.item) with
      | Tnone, _ ->
          newaxis :=
            (List.length !starts - List.length !squeeze + List.length !newaxis)
            :: !newaxis
      | (Tinteger | Tarray), it ->
          trivial := false;
          let v =
            match it with
            | User (Int i) -> `V (py_int i)
            | User (Arr v) -> `V v
            | _ -> invalid_arg "dynamic_slice"
          in
          let axis = List.hd p.axes in
          if sh.(axis) = 0 then
            index_error
              (Printf.sprintf "index is out of bounds for axis %d with size 0"
                 axis);
          starts := v :: !starts;
          sizes := 1 :: !sizes;
          squeeze := axis :: !squeeze
      | Tslice, User (Slice { start; stop; step }) ->
          if not (start = None && stop = None && step = None) then
            trivial := false;
          let axis = List.hd p.axes in
          let size = sh.(axis) in
          let start, stop, stride = slice_indices size start stop step in
          if stride < 0 then begin
            let new_start =
              stop + 1 + (abs (start - stop - 1) mod abs stride)
            in
            starts := `I new_start :: !starts;
            sizes := max 0 (start + 1 - new_start) :: !sizes;
            rev := axis :: !rev
          end
          else begin
            starts := `I start :: !starts;
            sizes := max 0 (stop - start) :: !sizes
          end
      | _ -> invalid_arg "dynamic_slice")
    (expand_ellipses sh parsed);
  let starts = List.rev !starts in
  let starts =
    if List.length starts > 1 then
      let dt = index_dtype sh in
      List.map
        (function `I i -> as_strong (py_int i) dt | `V v -> as_strong v dt)
        starts
    else List.map (function `I i -> py_int i | `V v -> v) starts
  in
  let a l = Array.of_list (List.rev l) in
  let y =
    if !trivial then x
    else
      let starts =
        if not normalize then starts
        else List.mapi (fun i st -> array_start st sh.(i)) starts
      in
      L.dynamic_slice_normalized x starts (a !sizes)
  in
  finish y ~rev:(a !rev) ~squeeze:(a !squeeze) ~newaxis:(a !newaxis)

let rec concrete_nd v =
  match v with
  | T.Concrete nd -> Some nd
  | T.Device b -> Some (Pjrt.Buffer.to_host b)
  | T.Tracer { payload = T.JVP { primal; _ }; _ } -> concrete_nd primal
  | T.Tracer _ -> None

let host_int_dtype () = if Config.value Config.enable_x64 then D.I64 else D.I32

let where_true nd =
  let sh = Ndarray.shape nd in
  let r = Array.length sh in
  let hits = ref [] in
  let n = prod sh in
  for flat = 0 to n - 1 do
    let ix = Array.make r 0 in
    let rem = ref flat in
    for d = r - 1 downto 0 do
      ix.(d) <- !rem mod sh.(d);
      rem := !rem / sh.(d)
    done;
    if Ndarray.get_f nd ix <> 0.0 then hits := ix :: !hits
  done;
  let hits = Array.of_list (List.rev !hits) in
  let k = Array.length hits in
  List.init r (fun d ->
      Ndarray.of_floats (host_int_dtype ()) [| k |]
        (Array.map (fun ix -> float_of_int ix.(d)) hits))

let expand_bool_indices sh parsed =
  List.concat
    (List.mapi
       (fun pos p ->
         if p.typ <> Tboolean then [ p ]
         else
           match p.item with
           | User (Bool b) -> [ { p with item = Pbool b } ]
           | User (Arr v) -> (
               match concrete_nd v with
               | None ->
                   raise
                     (Errors.non_concrete_boolean_index_error
                        ~tracer:(Lax_shape.aval_string (C.get_aval v))
                        ())
               | Some nd ->
                   let ish = Ndarray.shape nd in
                   if Array.length ish = 0 then
                     [ { p with item = Pbool (Ndarray.get_f nd [||] <> 0.0) } ]
                   else begin
                     let expected = List.map (fun i -> sh.(i)) p.axes in
                     if
                       not
                         (List.for_all2
                            (fun s1 s2 -> s1 = 0 || s1 = s2)
                            (List.filteri
                               (fun i _ -> i < List.length expected)
                               (Array.to_list ish))
                            (List.filteri
                               (fun i _ -> i < Array.length ish)
                               expected))
                     then
                       index_error
                         (Printf.sprintf
                            "boolean index did not match shape of indexed \
                             array in index %d: got %s, expected %s"
                            pos
                            (py_tuple (Array.to_list ish))
                            (py_list expected));
                     List.map2
                       (fun w axis ->
                         { item = Host w; typ = Tarray; axes = [ axis ] })
                       (where_true nd) p.axes
                   end)
           | _ -> [ p ])
       parsed)

type gather_indexer = {
  slice_shape : int array;
  gather_slice_shape : int array;
  gather_indices : T.value;
  dnums : T.gather_dims;
  unique : bool;
  sorted : bool;
  reversed_y_dims : int array;
  newaxis_dims : int array;
  scalar_bool_dims : int array;
}

let host_value nd = T.Concrete nd

let normalize_item sh p =
  let p =
    match p.item with
    | User (Arr v) -> { p with item = User (Arr (canon v)) }
    | _ -> p
  in
  match (p.typ, p.item) with
  | Tinteger, User (Int i) ->
      let size = sh.(List.hd p.axes) in
      { p with item = Pint (if i < 0 then i + size else i) }
  | Tarray, User (Arr v) when is_unsigned (dtype v) -> { p with item = Jarr v }
  | Tarray, User (Arr v) ->
      let size = sh.(List.hd p.axes) in
      let dt = dtype v in
      let neg =
        match concrete_nd v with
        | Some nd ->
            host_value
              (Ndarray.map D.Bool (fun x -> if x < 0.0 then 1.0 else 0.0) nd)
        | None -> Ufuncs.less v (py_int 0)
      in
      let sz = Ndarray.of_floats dt [||] [| float_of_int size |] in
      let sz =
        T.Concrete
          (if (C.get_aval v).T.weak_type then Ndarray.as_weak sz else sz)
      in
      { p with item = Jarr (L.select neg (L.add v sz) v) }
  | Tarray, Host nd ->
      let size = sh.(List.hd p.axes) in
      let dt = Ndarray.dtype nd in
      let neg = Ndarray.map D.Bool (fun x -> if x < 0.0 then 1.0 else 0.0) nd in
      let sz = T.Concrete (Ndarray.of_floats dt [||] [| float_of_int size |]) in
      {
        p with
        item =
          Jarr
            (L.select (host_value neg)
               (L.add (host_value nd) sz)
               (host_value nd));
      }
  | _ -> p

let as_jnp = function
  | User (Int i) | Pint i -> py_int i
  | User (Arr v) | Jarr v -> v
  | Host nd ->
      C.convert_like ~weak:false (host_value nd)
        (Dtypes.canonicalize_dtype (Ndarray.dtype nd))
  | _ -> invalid_arg "as_jnp"

let is_advanced_int parsed =
  List.exists
    (fun p -> (p.typ = Tarray || p.typ = Tboolean) && item_ndim p.item > 0)
    parsed

let index_to_gather ~normalize sh parsed =
  let adv_pos =
    List.filteri (fun _ p -> p.typ = Tarray || p.typ = Tinteger) parsed
    |> ignore;
    List.concat
      (List.mapi
         (fun i p -> if p.typ = Tarray || p.typ = Tinteger then [ i ] else [])
         parsed)
  in
  let contiguous =
    let rec ok = function
      | a :: (b :: _ as t) -> b - a = 1 && ok t
      | _ -> true
    in
    ok adv_pos
  in
  let parsed = expand_ellipses sh parsed in
  let scalar_bool_dims =
    List.concat
      (List.mapi (fun n p -> if p.typ = Tboolean then [ n ] else []) parsed)
  in
  let sh =
    let l = ref (Array.to_list sh) in
    List.iteri
      (fun i p ->
        if p.typ = Tboolean then
          l :=
            List.filteri (fun j _ -> j < i) !l
            @ [ 1 ]
            @ List.filteri (fun j _ -> j >= i) !l)
      parsed;
    Array.of_list !l
  in
  let cur = ref 0 in
  let parsed =
    List.map
      (fun p ->
        match (p.typ, p.item) with
        | Tboolean, Pbool b ->
            let a = !cur in
            incr cur;
            {
              item =
                Host
                  (Ndarray.of_floats (host_int_dtype ())
                     [| (if b then 1 else 0) |]
                     (if b then [| 0.0 |] else [||]));
              typ = Tarray;
              axes = [ a ];
            }
        | _ ->
            let n = List.length p.axes in
            let axes = List.init n (fun i -> !cur + i) in
            cur := !cur + n;
            { p with axes })
      parsed
  in
  let parsed =
    if normalize then List.map (normalize_item sh) parsed else parsed
  in
  let advanced = is_advanced_int parsed in
  let idx_adv, x_adv, adv_arrays =
    if not advanced then ([], [], [])
    else
      let without_none =
        List.filter
          (fun (_, p) -> p.typ <> Tnone)
          (List.mapi (fun i p -> (i, p)) parsed)
      in
      let triples =
        List.concat
          (List.mapi
             (fun j (i, p) ->
               if p.typ = Tarray || p.typ = Tinteger then
                 [ (i, j, as_jnp p.item) ]
               else [])
             without_none)
      in
      ( List.map (fun (i, _, _) -> i) triples,
        List.map (fun (_, j, _) -> j) triples,
        List.map (fun (_, _, a) -> a) triples )
  in
  let index_dtype = index_dtype sh in
  let x_axis = ref 0 and y_axis = ref 0 and collapsed_y = ref 0 in
  let offset_dims = ref [] and collapsed = ref [] and start_map = ref [] in
  let gather_indices = ref [] and gshape = ref [] in
  let slice_shape = ref [] and newaxis = ref [] and reversed = ref [] in
  let gslice = ref [] in
  let push r v = r := !r @ [ v ] in
  List.iteri
    (fun idx_pos p ->
      if
        adv_arrays <> []
        && ((contiguous && idx_pos = List.hd idx_adv)
           || ((not contiguous) && idx_pos = 0))
      then begin
        let arrs = Jnp_util.broadcast_arrays adv_arrays in
        let bshape = shape (List.hd arrs) in
        let start_dim = List.length !gshape in
        List.iter
          (fun a -> push gather_indices (as_strong a index_dtype, start_dim))
          arrs;
        gshape := !gshape @ Array.to_list bshape;
        start_map := !start_map @ x_adv;
        collapsed := !collapsed @ x_adv;
        slice_shape := !slice_shape @ Array.to_list bshape;
        y_axis := !y_axis + Array.length bshape;
        collapsed_y := !collapsed_y + Array.length bshape
      end;
      if List.mem idx_pos idx_adv then begin
        incr x_axis;
        push gslice 1
      end
      else
        match (p.typ, p.item) with
        | (Tinteger | Tarray), it when item_ndim it = 0 ->
            if sh.(!x_axis) = 0 then
              index_error
                (Printf.sprintf "index is out of bounds for axis %d with size 0"
                   !x_axis);
            let v =
              match it with
              | User (Int i) | Pint i -> as_strong (py_int i) index_dtype
              | _ -> as_strong (as_jnp it) index_dtype
            in
            push gather_indices (v, List.length !gshape);
            push collapsed !x_axis;
            push gslice 1;
            push start_map !x_axis;
            incr x_axis
        | Tnone, _ ->
            push slice_shape 1;
            push newaxis !y_axis;
            incr y_axis
        | Tslice, User (Slice { start; stop; step }) ->
            let size = sh.(!x_axis) in
            let step = Option.value step ~default:1 in
            if step = 0 then Py_exn.value_error "slice step cannot be zero";
            let gt = step > 0 in
            let clamp i =
              if i >= 0 then if gt then min size i else min (size - 1) i
              else if gt then max 0 (size + i)
              else max (-1) (size + i)
            in
            let start =
              match start with
              | None -> if gt then 0 else size - 1
              | Some s -> clamp s
            in
            let stop =
              match stop with
              | None -> if gt then size else -1
              | Some s -> clamp s
            in
            let gap = abs step in
            let distance = if gt then stop - start else start - stop in
            let slice_size = max 0 (distance + gap - 1) / gap in
            push slice_shape slice_size;
            if step = 1 then begin
              if slice_size <> size then begin
                push gather_indices
                  (as_strong (py_int start) index_dtype, List.length !gshape);
                push start_map !x_axis
              end;
              push gslice slice_size;
              push offset_dims !collapsed_y
            end
            else begin
              let c v =
                T.Concrete
                  (Ndarray.of_floats index_dtype [||] [| float_of_int v |])
              in
              let indices =
                Ufuncs.add (c start)
                  (Ufuncs.multiply (c step) (L.iota index_dtype slice_size))
              in
              let indices =
                if step < 0 then begin
                  push reversed !collapsed_y;
                  L.rev indices [| 0 |]
                end
                else indices
              in
              push gslice 1;
              push gather_indices (indices, List.length !gshape);
              push start_map !x_axis;
              push gshape slice_size;
              push collapsed !x_axis
            end;
            incr collapsed_y;
            incr y_axis;
            incr x_axis
        | _ ->
            index_error
              (Printf.sprintf "Got unsupported indexer at position %d" idx_pos))
    parsed;
  let gather_array =
    match !gather_indices with
    | [] -> T.Concrete (Ndarray.of_floats index_dtype [| 0 |] [||])
    | [ (g, _) ] -> L.expand_dims g [| ndim g |]
    | gs ->
        let last = List.length !gshape in
        let target = Array.of_list (!gshape @ [ 1 ]) in
        L.concatenate
          (List.map
             (fun (g, i) ->
               L.broadcast_in_dim g target
                 (Array.init (ndim g) (fun d -> i + d)))
             gs)
          last
  in
  let a = Array.of_list in
  {
    slice_shape = a !slice_shape;
    gather_slice_shape = a !gslice;
    gather_indices = gather_array;
    dnums =
      {
        T.offset_dims = a !offset_dims;
        collapsed_slice_dims = a (List.sort compare !collapsed);
        start_index_map = a !start_map;
        g_operand_batching_dims = [||];
        g_start_indices_batching_dims = [||];
        g_sorted_unique = false;
      };
    unique = adv_arrays = [];
    sorted = adv_arrays = [];
    reversed_y_dims = a !reversed;
    newaxis_dims = a !newaxis;
    scalar_bool_dims = a scalar_bool_dims;
  }

let is_empty sh = Array.exists (fun d -> d = 0) sh

let python_float_repr f =
  if Float.is_nan f then "nan"
  else if Float.is_integer f && Float.abs f < 1e16 then Printf.sprintf "%.1f" f
  else
    let rec go p =
      let s = Printf.sprintf "%.*g" p f in
      if p >= 17 || float_of_string s = f then s else go (p + 1)
    in
    go 1

let fill_of v =
  if ndim v <> 0 then
    Py_exn.value_error "fill_value argument to indexed get() must be a scalar";
  match concrete_nd v with
  | None ->
      Py_exn.value_error "fill_value argument to indexed get() must be concrete"
  | Some nd ->
      let f = Ndarray.get_f nd [||] in
      let repr =
        if is_integer (Ndarray.dtype nd) then Printf.sprintf "%.0f" f
        else python_float_repr f
      in
      (repr, f)

let gather ~sorted ~unique ~mode ~fill_value ~normalize x parsed =
  let ix = index_to_gather ~normalize (shape x) parsed in
  let fill_value = Option.map fill_of fill_value in
  let y = x in
  let y =
    if ix.scalar_bool_dims = [||] then y
    else L.expand_dims y ix.scalar_bool_dims
  in
  if is_empty ix.slice_shape then L.full_like ~shape:ix.slice_shape y 0.0
  else
    let y =
      if is_empty (shape ix.gather_indices) then y
      else
        let indices_are_sorted = sorted || ix.sorted in
        let unique_indices = unique || ix.unique in
        let m =
          match mode with None -> LC.Promise_in_bounds | m -> parse_mode m
        in
        match (m, fill_value) with
        | LC.Fill_or_drop, Some (repr, f) ->
            let prim =
              T.Gather
                {
                  dimension_numbers = ix.dnums;
                  slice_sizes = ix.gather_slice_shape;
                }
            in
            let prim = Np_ints.mark_gather_fill prim repr (Some f) in
            let prim =
              if indices_are_sorted || unique_indices then
                Np_ints.mark_gather_flags ~sorted:indices_are_sorted
                  ~unique:unique_indices prim
              else prim
            in
            C.bind1 prim [ y; ix.gather_indices ]
        | _ ->
            L.gather ~mode:m ~indices_are_sorted ~unique_indices y
              ix.gather_indices ix.dnums ix.gather_slice_shape
    in
    let y =
      if ix.reversed_y_dims = [||] then y else L.rev y ix.reversed_y_dims
    in
    L.expand_dims y ix.newaxis_dims

let take ?(indices_are_sorted = false) ?(unique_indices = false) ?mode
    ?fill_value ?(normalize_indices = true) x indices =
  let parsed = parse indices (shape x) in
  if
    Config.value Config.check_static_indices
    && (mode = None || parse_mode mode = LC.Promise_in_bounds)
  then validate_static_indices ~normalize:normalize_indices (shape x) parsed;
  let attempt f =
    try Some (f ())
    with Py_exn.Type_error _ | Py_exn.Value_error _ | Py_exn.Index_error _ ->
      None
  in
  match
    attempt (fun () ->
        to_static_slice ~normalize:normalize_indices ~mode (shape x) parsed)
  with
  | Some s -> static_slice x s
  | None -> (
      match
        attempt (fun () ->
            dynamic_slice ~normalize:normalize_indices ~mode x parsed)
      with
      | Some y -> y
      | None ->
          let parsed = expand_bool_indices (shape x) parsed in
          gather ~sorted:indices_are_sorted ~unique:unique_indices ~mode
            ~fill_value ~normalize:normalize_indices x parsed)

type update =
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?mode:string ->
  ?normalize_indices:bool ->
  T.value ->
  index list ->
  T.value ->
  T.value

type scatter_op =
  | Set
  | Add
  | Sub
  | Mul
  | Min
  | Max
  | Apply of (T.value -> T.value)

let scatter_apply ~indices_are_sorted ~unique_indices ?mode x g y f
    (dn : T.scatter_dims) =
  let dt = dtype x in
  let scalar = { T.dtype = dt; shape = [||]; weak_type = false } in
  let jaxpr =
    Jaxpr.make_jaxpr [ scalar; scalar ] (function
      | [ a; _ ] -> [ f a ]
      | _ -> invalid_arg "scatter_apply")
  in
  let host v =
    let s = T.Concrete (Ndarray.of_floats dt [||] [| v |]) in
    match Jaxpr.eval_closed_jaxpr jaxpr [ s; s ] with
    | [ r ] -> (
        match concrete_nd r with
        | Some nd -> Ndarray.get_f nd [||]
        | None -> invalid_arg "scatter_apply")
    | _ -> invalid_arg "scatter_apply"
  in
  let unused = L.full (shape y) 0.0 dt in
  let dn =
    match mode with Some LC.Clip -> { dn with T.s_clip = true } | _ -> dn
  in
  let prim = T.Scatter { dimension_numbers = dn; unique_indices } in
  let prim =
    if indices_are_sorted || unique_indices then
      Np_ints.mark_scatter_flags ~sorted:indices_are_sorted ~unique:false prim
    else prim
  in
  let prim =
    match mode with
    | Some LC.Promise_in_bounds -> Np_ints.mark_prim prim
    | Some LC.One_hot -> Np_ints.mark_scatter_one_hot prim
    | _ -> prim
  in
  C.bind1 (Scatter_apply.mark prim jaxpr host) [ x; g; unused ]

let scatter_update ?(indices_are_sorted = false) ?(unique_indices = false) ?mode
    ?(normalize_indices = true) op x indices y =
  let x_dt = dtype x and weak = (C.get_aval x).T.weak_type in
  let parsed = expand_bool_indices (shape x) (parse indices (shape x)) in
  let ix = index_to_gather ~normalize:normalize_indices (shape x) parsed in
  if is_empty ix.slice_shape then x
  else
    let dt, w =
      Dtypes.result_type [ (x_dt, weak); (dtype y, (C.get_aval y).T.weak_type) ]
    in
    let x = C.convert_like ~weak:w x dt and y = C.convert_like ~weak:w y dt in
    let y = Jnp_util.broadcast_to y ix.slice_shape in
    let y = Lax_numpy.squeeze ~axis:ix.newaxis_dims y in
    let y =
      if ix.reversed_y_dims = [||] then y else L.rev y ix.reversed_y_dims
    in
    let x =
      if ix.scalar_bool_dims = [||] then x
      else L.expand_dims x ix.scalar_bool_dims
    in
    let dn =
      {
        T.update_window_dims = ix.dnums.offset_dims;
        inserted_window_dims = ix.dnums.collapsed_slice_dims;
        scatter_dims_to_operand_dims = ix.dnums.start_index_map;
        s_operand_batching_dims = [||];
        s_scatter_indices_batching_dims = [||];
        s_sorted_unique = false;
        s_clip = false;
      }
    in
    let mode = Option.map (fun m -> parse_mode (Some m)) mode in
    let indices_are_sorted = ix.sorted || indices_are_sorted in
    let unique_indices = ix.unique || unique_indices in
    let g = ix.gather_indices in
    let out =
      match op with
      | Set -> L.scatter ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Add -> L.scatter_add ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Sub -> L.scatter_sub ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Mul -> L.scatter_mul ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Min -> L.scatter_min ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Max -> L.scatter_max ?mode ~indices_are_sorted ~unique_indices x g y dn
      | Apply f ->
          scatter_apply ~indices_are_sorted ~unique_indices ?mode x g y f dn
    in
    let out =
      if ix.scalar_bool_dims = [||] then out
      else L.squeeze out ix.scalar_bool_dims
    in
    C.convert_like ~weak out x_dt

let scalar_values x y =
  match y with
  | T.Concrete nd
    when Ndarray.weak nd
         && Array.length (Ndarray.shape nd) = 0
         && is_integer (Ndarray.dtype nd)
         && is_integer (dtype x) ->
      let v = Ndarray.get_f nd [||] in
      let info = Dtypes.iinfo (dtype x) in
      if info.imin <= v && v <= info.imax then
        T.Concrete (Ndarray.of_floats (dtype x) [||] [| v |])
      else y
  | _ -> y

let at_set ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx y
    =
  scatter_update ?indices_are_sorted ?unique_indices ?mode ?normalize_indices
    Set x idx (scalar_values x y)

let at_op op ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx
    y =
  scatter_update ?indices_are_sorted ?unique_indices ?mode ?normalize_indices op
    x idx (scalar_values x y)

let at_add = at_op Add
let at_subtract = at_op Sub
let at_multiply = at_op Mul
let at_min = at_op Min
let at_max = at_op Max

let at_divide ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx
    y =
  let ones = Array_creation.ones_like x in
  Ufuncs.divide x
    (scatter_update ?indices_are_sorted ?unique_indices ?mode ?normalize_indices
       Mul ones idx (scalar_values ones y))

let at_power ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx
    y =
  let ones = Array_creation.ones_like x in
  Ufuncs.power x
    (scatter_update ?indices_are_sorted ?unique_indices ?mode ?normalize_indices
       Mul ones idx (scalar_values ones y))

let at_apply ?indices_are_sorted ?unique_indices ?mode ?normalize_indices x idx
    f =
  scatter_update ?indices_are_sorted ?unique_indices ?mode ?normalize_indices
    (Apply f) x idx
    (T.Concrete (Ndarray.of_floats (dtype x) [||] [| 0.0 |]))

let at_inplace_warning =
  "Because JAX arrays are immutable, jnp.ufunc.at() cannot operate inplace like\n\
   np.ufunc.at(). Instead, you can pass inplace=False and capture the result; \
   e.g.\n\
   >>> arr = jnp.add.at(arr, ind, val, inplace=False)\n"

let check_inplace inplace =
  if inplace then raise (Py_exn.Not_implemented_error at_inplace_warning)

let bool_via_int32 a b f =
  if dtype a = D.Bool then
    let a = Lax_numpy.astype a D.I32 in
    let b = Lax_numpy.astype (C.convert_like ~weak:false b D.Bool) D.I32 in
    Lax_numpy.astype (f a b) D.Bool
  else f a b

let ufunc_jit name f a idx b =
  let leaves =
    List.concat
      (List.mapi
         (fun k item ->
           let bad ty =
             raise
               (Py_exn.Type_error
                  (Printf.sprintf
                     "Error interpreting argument to <function ufunc.at> as an \
                      abstract array. The problematic value is of type <class \
                      '%s'> and was passed to the function at path indices[%d].\n\
                      This typically means that a jit-wrapped function was \
                      called with a non-array argument, and this argument was \
                      not marked as static using the static_argnums or \
                      static_argnames parameters of jax.jit."
                     ty k))
           in
           match item with
           | Slice _ -> bad "slice"
           | Ellipsis -> bad "ellipsis"
           | Newaxis -> []
           | Int i -> [ py_int i ]
           | Bool b ->
               [
                 T.Concrete
                   (Ndarray.of_floats D.Bool [||]
                      [| (if b then 1.0 else 0.0) |]);
               ]
           | Arr v -> [ canon v ])
         idx)
  in
  let shape_key =
    String.concat ","
      (List.map
         (function
           | Newaxis -> "n"
           | Int _ -> "i"
           | Bool _ -> "b"
           | Arr _ -> "a"
           | _ -> "?")
         idx)
  in
  Jnp_util.jit1 "at"
    ~key:(name ^ ":" ^ shape_key)
    (fun args ->
      match args with
      | a :: rest ->
          let rest = ref rest in
          let next () =
            match !rest with
            | v :: tl ->
                rest := tl;
                v
            | [] -> invalid_arg "ufunc.at"
          in
          let idx =
            List.map (function Newaxis -> Newaxis | _ -> Arr (next ())) idx
          in
          f a idx (next ())
      | [] -> invalid_arg "ufunc.at")
    ((a :: leaves) @ [ b ])

let ufunc_add_at ?(inplace = true) a idx b =
  ufunc_jit "add"
    (fun a idx b ->
      check_inplace inplace;
      bool_via_int32 a b (fun a b -> at_add a idx b))
    a idx b

let ufunc_multiply_at ?(inplace = true) a idx b =
  ufunc_jit "multiply"
    (fun a idx b ->
      check_inplace inplace;
      bool_via_int32 a b (fun a b -> at_multiply a idx b))
    a idx b

let ufunc_subtract_at ?(inplace = true) a idx b =
  ufunc_jit "subtract"
    (fun a idx b ->
      check_inplace inplace;
      at_subtract a idx b)
    a idx b

let ufunc_at_operands ~name idx =
  let leaves =
    List.concat
      (List.mapi
         (fun k item ->
           let bad ty =
             raise
               (Py_exn.Type_error
                  (Printf.sprintf
                     "Error interpreting argument to <function ufunc.at> as an \
                      abstract array. The problematic value is of type <class \
                      '%s'> and was passed to the function at path indices[%d].\n\
                      This typically means that a jit-wrapped function was \
                      called with a non-array argument, and this argument was \
                      not marked as static using the static_argnums or \
                      static_argnames parameters of jax.jit."
                     ty k))
           in
           match item with
           | Slice _ -> bad "slice"
           | Ellipsis -> bad "ellipsis"
           | Newaxis -> []
           | Int i -> [ py_int i ]
           | Bool b ->
               [
                 T.Concrete
                   (Ndarray.of_floats D.Bool [||]
                      [| (if b then 1.0 else 0.0) |]);
               ]
           | Arr v -> [ canon v ])
         idx)
  in
  let key =
    String.concat ","
      (List.map
         (function
           | Newaxis -> "n"
           | Int _ -> "i"
           | Bool _ -> "b"
           | Arr _ -> "a"
           | _ -> "?")
         idx)
  in
  (leaves, name ^ ":scan:" ^ key)

let at_via_scan ufn a idx args =
  let one v = Array_creation.ones_like ~shape:[||] v in
  let dt =
    match
      Api.eval_shape
        (fun leaves ->
          Tree_util.Leaf
            (ufn (List.map (function Tree_util.Leaf v -> v | _ -> a) leaves)))
        (List.map (fun v -> Tree_util.Leaf (one v)) (a :: args))
    with
    | Tree_util.Leaf av -> av.T.dtype
    | _ -> invalid_arg "ufunc.at"
  in
  let a = Lax_numpy.astype a dt in
  let args = List.map (fun v -> Lax_numpy.astype v dt) args in
  if idx = [] then a
  else
    let shapes =
      List.filter_map
        (function Arr v -> Some (shape v) | Slice _ -> None | _ -> Some [||])
        idx
    in
    let bshape =
      match shapes with [] -> [||] | l -> Extras.broadcast_shapes l
    in
    if bshape = [||] then at_set a idx (ufn (take a idx :: args))
    else
      let n = prod bshape in
      let nb = Array.length bshape in
      let args =
        List.map
          (fun arg ->
            let s = shape arg in
            let tail =
              if Array.length s > nb then Array.sub s nb (Array.length s - nb)
              else [||]
            in
            Lax_numpy.reshape
              (Jnp_util.broadcast_to arg (Array.append bshape tail))
              (Array.append [| n |] tail))
          args
      in
      let idx =
        List.map
          (function
            | Arr v -> Arr (Lax_numpy.ravel (Jnp_util.broadcast_to v bshape))
            | item -> item)
          idx
      in
      match
        L.scan ~length:n
          (function
            | [ i; acc ] ->
                let pick = function
                  | Arr v -> Arr (take v [ Arr i ])
                  | item -> item
                in
                let idx_i = List.map pick idx in
                let current = take acc idx_i in
                let vals = List.map (fun arg -> take arg [ Arr i ]) args in
                let updated = at_set acc idx_i (ufn (current :: vals)) in
                let next_i = C.bind1 T.Add [ i; py_int 1 ] in
                [ next_i; updated ]
            | _ -> invalid_arg "ufunc.at scan")
          [ py_int 0; a ]
          []
      with
      | [ _; r ] -> r
      | _ -> invalid_arg "ufunc.at scan"

let ufunc_at ?(inplace = true) ~name ufn a idx args =
  let leaves, key = ufunc_at_operands ~name idx in
  Jnp_util.jit1 "at" ~key
    (fun operands ->
      check_inplace inplace;
      match operands with
      | a :: rest ->
          let rest = ref rest in
          let next () =
            match !rest with
            | v :: tl ->
                rest := tl;
                v
            | [] -> invalid_arg "ufunc.at"
          in
          let idx =
            List.map (function Newaxis -> Newaxis | _ -> Arr (next ())) idx
          in
          at_via_scan ufn a idx !rest
      | [] -> invalid_arg "ufunc.at")
    ((a :: leaves) @ args)
