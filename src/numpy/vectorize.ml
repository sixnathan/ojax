module T = Types
module C = Core
module Nd = Ndarray
module D = Dtype
module LN = Lax_numpy

let get_aval = C.get_aval
let shape_of v = (get_aval v).T.shape

let concrete = function
  | T.Concrete nd -> nd
  | _ -> failwith "vectorize: forward-eval requires concrete inputs"

let is_complex = function D.Complex64 | D.Complex128 -> true | _ -> false

let unravel lin sh =
  let n = Array.length sh in
  let idx = Array.make n 0 in
  let rem = ref lin in
  for d = n - 1 downto 0 do
    idx.(d) <- (if sh.(d) = 0 then 0 else !rem mod sh.(d));
    rem := if sh.(d) = 0 then 0 else !rem / sh.(d)
  done;
  idx

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

let parse_signature (s : string) : int list * int list =
  let arrow =
    match String.index_opt s '-' with
    | Some k when k + 1 < String.length s && s.[k + 1] = '>' -> k
    | _ -> invalid_arg ("not a valid gufunc signature: " ^ s)
  in
  let lhs = String.sub s 0 arrow in
  let rhs = String.sub s (arrow + 2) (String.length s - arrow - 2) in
  (List.map List.length (parse_side lhs), List.map List.length (parse_side rhs))

let extract_slice nd nc bidx =
  let sh = Nd.shape nd in
  let ndim = Array.length sh in
  let nb = ndim - nc in
  let batch_shape = Array.sub sh 0 nb in
  let core_shape = Array.sub sh nb nc in
  let bb = Array.length bidx in
  let batch_idx =
    Array.init nb (fun i ->
        if batch_shape.(i) = 1 then 0 else bidx.(bb - nb + i))
  in
  let dt = Nd.dtype nd in
  let csize = Array.fold_left ( * ) 1 core_shape in
  let full_idx = Array.make ndim 0 in
  Array.blit batch_idx 0 full_idx 0 nb;
  if is_complex dt then (
    let buf = Array.make csize Complex.zero in
    for lin = 0 to csize - 1 do
      Array.blit (unravel lin core_shape) 0 full_idx nb nc;
      buf.(lin) <- Nd.get_c nd full_idx
    done;
    Nd.of_complex dt core_shape buf)
  else
    let buf = Array.make csize 0.0 in
    for lin = 0 to csize - 1 do
      Array.blit (unravel lin core_shape) 0 full_idx nb nc;
      buf.(lin) <- Nd.get_f nd full_idx
    done;
    Nd.of_floats dt core_shape buf

let flat_f nd =
  let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
  let a = Array.make n 0.0 in
  ignore
    (Nd.fold
       (fun i x ->
         a.(i) <- x;
         i + 1)
       0 nd);
  a

let flat_c nd =
  let sh = Nd.shape nd in
  let n = Array.fold_left ( * ) 1 sh in
  Array.init n (fun lin -> Nd.get_c nd (unravel lin sh))

let vectorize ?(excluded = []) ?signature pyfunc raw_args =
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
  if Array.length dyn_arr <> Array.length ncs then
    invalid_arg "vectorize: wrong number of positional arguments";
  if List.length output_ncs <> 1 then
    failwith "vectorize: only single-output gufuncs supported";
  let batch_shapes =
    Array.to_list
      (Array.mapi
         (fun idx v ->
           let sh = shape_of v in
           let ndim = Array.length sh in
           let nc = ncs.(idx) in
           if ndim < nc then
             invalid_arg
               "vectorize: input has too few dimensions for its core dimensions";
           Array.sub sh 0 (ndim - nc))
         dyn_arr)
  in
  let broadcast_shape = LN.broadcast_shapes_n batch_shapes in
  let bsize = Array.fold_left ( * ) 1 broadcast_shape in
  let concretes = Array.map concrete dyn_arr in
  let fbuf = ref [||] in
  let cbuf = ref [||] in
  let out_dt = ref D.F32 in
  let out_shape = ref [||] in
  let out_csize = ref 0 in
  let cplx = ref false in
  for lin = 0 to bsize - 1 do
    let bidx = unravel lin broadcast_shape in
    let dyn_slices =
      Array.to_list
        (Array.mapi
           (fun idx nd -> T.Concrete (extract_slice nd ncs.(idx) bidx))
           concretes)
    in
    let out = concrete (call_full dyn_slices) in
    if lin = 0 then (
      out_dt := Nd.dtype out;
      out_shape := Nd.shape out;
      out_csize := Array.fold_left ( * ) 1 !out_shape;
      cplx := is_complex !out_dt;
      if !cplx then cbuf := Array.make (bsize * !out_csize) Complex.zero
      else fbuf := Array.make (bsize * !out_csize) 0.0);
    let base = lin * !out_csize in
    if !cplx then Array.blit (flat_c out) 0 !cbuf base !out_csize
    else Array.blit (flat_f out) 0 !fbuf base !out_csize
  done;
  let final_shape = Array.append broadcast_shape !out_shape in
  if !cplx then T.Concrete (Nd.of_complex !out_dt final_shape !cbuf)
  else T.Concrete (Nd.of_floats !out_dt final_shape !fbuf)
