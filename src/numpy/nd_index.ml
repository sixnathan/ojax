module C = Core
module T = Types
module D = Dtype
module L = Lax_api
module UF = Ufuncs

type index =
  | Int of int
  | Range of { start : int option; stop : int option }
  | Arr of T.value

let shape v = (C.get_aval v).T.shape
let dtype v = (C.get_aval v).T.dtype
let rank v = Array.length (shape v)
let full = Range { start = None; stop = None }

let expand idx n =
  let k = List.length idx in
  if k > n then
    raise
      (Py_exn.Index_error
         (Printf.sprintf
            "Too many indices: array is %d-dimensional, but %d were indexed" n k));
  idx @ List.init (n - k) (fun _ -> full)

let bound size v =
  let v = if v < 0 then v + size else v in
  max 0 (min v size)

let range_of size start stop =
  let lo = match start with None -> 0 | Some v -> bound size v in
  let hi = match stop with None -> size | Some v -> bound size v in
  (lo, max lo hi)

let index_dtype sh =
  if Array.exists (fun d -> d >= 1 lsl 31) sh then D.I64 else D.I32

let const dt v = T.Concrete (Ndarray.of_floats dt [||] [| float_of_int v |])

let squeeze_axes idx =
  List.mapi (fun axis i -> match i with Int _ -> Some axis | _ -> None) idx
  |> List.filter_map Fun.id |> Array.of_list

let static_take x idx =
  let sh = shape x in
  let starts = Array.make (Array.length sh) 0 in
  let limits = Array.make (Array.length sh) 0 in
  List.iteri
    (fun axis i ->
      match i with
      | Int v ->
          let v = if v < 0 then v + sh.(axis) else v in
          let v = max 0 (min v (sh.(axis) - 1)) in
          starts.(axis) <- v;
          limits.(axis) <- v + 1
      | Range { start; stop } ->
          let lo, hi = range_of sh.(axis) start stop in
          starts.(axis) <- lo;
          limits.(axis) <- hi
      | Arr _ -> assert false)
    idx;
  let trivial = Array.for_all (fun v -> v = 0) starts && limits = sh in
  let out = if trivial then x else L.slice x starts limits in
  L.squeeze out (squeeze_axes idx)

let dynamic_take x idx =
  let sh = shape x in
  let dt = index_dtype sh in
  let starts = ref [] in
  let sizes = ref [] in
  List.iteri
    (fun axis i ->
      match i with
      | Int v ->
          starts := const dt v :: !starts;
          sizes := 1 :: !sizes
      | Arr v ->
          starts :=
            (if List.length idx > 1 then C.convert_like ~weak:false v dt else v)
            :: !starts;
          sizes := 1 :: !sizes
      | Range { start; stop } ->
          let lo, hi = range_of sh.(axis) start stop in
          starts := const dt lo :: !starts;
          sizes := (hi - lo) :: !sizes)
    idx;
  let squeeze =
    List.mapi
      (fun axis i ->
        match i with Int _ | Arr _ -> Some axis | Range _ -> None)
      idx
    |> List.filter_map Fun.id |> Array.of_list
  in
  let starts = List.rev !starts in
  let normalised =
    List.mapi
      (fun axis s ->
        let d = dtype s in
        let negative = UF.less s (const d 0) in
        let shifted = L.add s (const d sh.(axis)) in
        C.bind1 T.Select_n [ negative; s; shifted ])
      starts
  in
  let out =
    L.dynamic_slice_normalized x normalised (Array.of_list (List.rev !sizes))
  in
  L.squeeze out squeeze

type indexer = {
  slice_shape : int array;
  gather_slice_shape : int array;
  gather_indices : T.value;
  offset_dims : int array;
  collapsed_slice_dims : int array;
  start_index_map : int array;
  plain : bool;
}

let to_gather sh idx =
  let normalise axis = function
    | Arr v ->
        let d = dtype v in
        let negative = UF.less v (const d 0) in
        let shifted = L.add v (const d sh.(axis)) in
        Arr (C.bind1 T.Select_n [ negative; v; shifted ])
    | i -> i
  in
  let idx = List.mapi normalise idx in
  let is_adv = function Arr v -> rank v > 0 | _ -> false in
  let any_adv = List.exists is_adv idx in
  let is_adv_pos = function Arr _ | Int _ -> any_adv | Range _ -> false in
  let adv_positions =
    List.mapi (fun p i -> if is_adv_pos i then Some p else None) idx
    |> List.filter_map Fun.id
  in
  let contiguous =
    match adv_positions with
    | [] -> true
    | first :: _ ->
        List.for_all2 ( = ) adv_positions
          (List.mapi (fun i _ -> first + i) adv_positions)
  in
  if not contiguous then
    Py_exn.not_implemented_error "Nd_index: separated advanced indices";
  let dt = index_dtype sh in
  let gather = ref [] in
  let gather_shape = ref [] in
  let offset = ref [] in
  let collapsed = ref [] in
  let start_map = ref [] in
  let slice_shape = ref [] in
  let gather_slice = ref [] in
  let collapsed_y = ref 0 in
  let as_value = function
    | Arr v -> v
    | Int v -> const (Dtypes.default_int_dtype ()) v
    | Range _ -> assert false
  in
  List.iteri
    (fun pos i ->
      let x_axis = pos in
      if any_adv && pos = List.hd adv_positions then begin
        let arrs =
          List.filteri (fun p _ -> List.mem p adv_positions) idx
          |> List.map as_value
        in
        let target = Array.of_list (List.map shape arrs) in
        let same = Array.for_all (fun s -> s = target.(0)) target in
        let arrs = if same then arrs else Jnp_util.broadcast_arrays arrs in
        let bshape = shape (List.hd arrs) in
        let start_dim = List.length !gather_shape in
        List.iter
          (fun a ->
            gather := (C.convert_like ~weak:false a dt, start_dim) :: !gather)
          arrs;
        gather_shape := !gather_shape @ Array.to_list bshape;
        start_map := !start_map @ adv_positions;
        collapsed := !collapsed @ adv_positions;
        slice_shape := !slice_shape @ Array.to_list bshape;
        collapsed_y := !collapsed_y + Array.length bshape
      end;
      if List.mem pos adv_positions then gather_slice := 1 :: !gather_slice
      else
        match i with
        | Int v ->
            let v = if v < 0 then v + sh.(x_axis) else v in
            gather := (const dt v, List.length !gather_shape) :: !gather;
            collapsed := !collapsed @ [ x_axis ];
            gather_slice := 1 :: !gather_slice;
            start_map := !start_map @ [ x_axis ]
        | Arr v ->
            gather :=
              (C.convert_like ~weak:false v dt, List.length !gather_shape)
              :: !gather;
            collapsed := !collapsed @ [ x_axis ];
            gather_slice := 1 :: !gather_slice;
            start_map := !start_map @ [ x_axis ]
        | Range { start; stop } ->
            let lo, hi = range_of sh.(x_axis) start stop in
            let size = hi - lo in
            slice_shape := !slice_shape @ [ size ];
            if size <> sh.(x_axis) then begin
              gather := (const dt lo, List.length !gather_shape) :: !gather;
              start_map := !start_map @ [ x_axis ]
            end;
            gather_slice := size :: !gather_slice;
            offset := !offset @ [ !collapsed_y ];
            incr collapsed_y)
    idx;
  let parts = List.rev !gather in
  let gather_indices =
    match parts with
    | [] -> T.Concrete (Ndarray.of_floats dt [| 0 |] [||])
    | [ (g, _) ] -> L.expand_dims g [| rank g |]
    | _ ->
        let last = List.length !gather_shape in
        let target = Array.of_list (!gather_shape @ [ 1 ]) in
        L.concatenate
          (List.map
             (fun (g, i) ->
               L.broadcast_in_dim g target
                 (Array.init (rank g) (fun d -> i + d)))
             parts)
          last
  in
  {
    slice_shape = Array.of_list !slice_shape;
    gather_slice_shape = Array.of_list (List.rev !gather_slice);
    gather_indices;
    offset_dims = Array.of_list !offset;
    collapsed_slice_dims = Array.of_list (List.sort compare !collapsed);
    start_index_map = Array.of_list !start_map;
    plain = not any_adv;
  }

let empty sh = Array.exists (fun d -> d = 0) sh

let gather_take ?(clip = false) x idx =
  let ix = to_gather (shape x) idx in
  if empty ix.slice_shape then L.full ix.slice_shape 0.0 (dtype x)
  else
    let prim =
      T.Gather
        {
          dimension_numbers =
            {
              offset_dims = ix.offset_dims;
              collapsed_slice_dims = ix.collapsed_slice_dims;
              start_index_map = ix.start_index_map;
              g_operand_batching_dims = [||];
              g_start_indices_batching_dims = [||];
              g_sorted_unique = false;
            };
          slice_sizes = ix.gather_slice_shape;
        }
    in
    let prim =
      if clip then Np_ints.mark_gather_clip ~sorted:false prim else prim
    in
    C.bind1 prim [ x; ix.gather_indices ]

let check_scalar_indices sh idx =
  let advanced = function Arr v -> rank v > 0 | _ -> false in
  if not (List.exists advanced idx) then
    List.iteri
      (fun axis i ->
        match i with
        | (Int _ | Arr _) when sh.(axis) = 0 ->
            raise
              (Py_exn.Index_error
                 (Printf.sprintf
                    "index is out of bounds for axis %d with size 0" axis))
        | _ -> ())
      idx

let take_hook : (string option -> T.value -> index list -> T.value) option ref =
  ref None

let set_hook : (T.value -> index list -> T.value -> T.value) option ref =
  ref None

let set_stepped_hook :
    (T.value -> start:int -> step:int -> T.value -> T.value) option ref =
  ref None

let take_clip x idx =
  match !take_hook with
  | Some f -> f (Some "clip") x idx
  | None ->
      let idx = expand idx (rank x) in
      check_scalar_indices (shape x) idx;
      gather_take ~clip:true x idx

let static_in_bounds x idx =
  let sh = shape x in
  List.for_all Fun.id
    (List.mapi
       (fun axis i ->
         match i with
         | Int v ->
             let v = if v < 0 then v + sh.(axis) else v in
             0 <= v && v < sh.(axis)
         | _ -> true)
       idx)

let take x idx =
  match !take_hook with
  | Some f -> f None x idx
  | None ->
      let idx = expand idx (rank x) in
      check_scalar_indices (shape x) idx;
      let is_arr = function Arr _ -> true | _ -> false in
      if not (List.exists is_arr idx) then
        if static_in_bounds x idx then static_take x idx else dynamic_take x idx
      else if List.for_all (function Arr v -> rank v = 0 | _ -> true) idx then
        dynamic_take x idx
      else gather_take x idx

let set x idx y =
  match !set_hook with
  | Some f -> f x idx y
  | None ->
      let idx = expand idx (rank x) in
      check_scalar_indices (shape x) idx;
      let x_aval = C.get_aval x in
      let ix = to_gather x_aval.shape idx in
      if empty ix.slice_shape then x
      else
        let dt, is_weak =
          Dtypes.result_type
            [
              (x_aval.dtype, x_aval.weak_type);
              (dtype y, (C.get_aval y).T.weak_type);
            ]
        in
        let x' = C.convert_like ~weak:is_weak x dt in
        let y' = C.convert_like ~weak:is_weak y dt in
        let y' =
          if shape y' = ix.slice_shape then y'
          else
            let sy = shape y' in
            let lead = Array.length ix.slice_shape - Array.length sy in
            L.broadcast_in_dim y' ix.slice_shape
              (Array.init (Array.length sy) (fun d -> d + lead))
        in
        let out =
          C.bind1
            (T.Scatter
               {
                 dimension_numbers =
                   {
                     update_window_dims = ix.offset_dims;
                     inserted_window_dims = ix.collapsed_slice_dims;
                     scatter_dims_to_operand_dims = ix.start_index_map;
                     s_operand_batching_dims = [||];
                     s_scatter_indices_batching_dims = [||];
                     s_sorted_unique = ix.plain;
                     s_clip = false;
                   };
                 unique_indices = ix.plain;
               })
            [ x'; ix.gather_indices; y' ]
        in
        C.convert_like ~weak:x_aval.weak_type out x_aval.dtype

let set_stepped x ~start ~step y =
  match !set_stepped_hook with
  | Some f -> f x ~start ~step y
  | None ->
      let sh = shape x in
      let dt = index_dtype sh in
      if start >= sh.(0) then x
      else
        let n = ((sh.(0) - start - 1) / step) + 1 in
        let positions =
          UF.add (const dt start) (UF.multiply (const dt step) (L.iota dt n))
        in
        C.bind1
          (T.Scatter
             {
               dimension_numbers =
                 {
                   update_window_dims = [||];
                   inserted_window_dims = [| 0 |];
                   scatter_dims_to_operand_dims = [| 0 |];
                   s_operand_batching_dims = [||];
                   s_scatter_indices_batching_dims = [||];
                   s_sorted_unique = true;
                   s_clip = false;
                 };
               unique_indices = true;
             })
          [ x; L.expand_dims positions [| 1 |]; y ]
