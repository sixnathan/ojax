module T = Types
module NL = Lax_numpy
module Batching = Interpreters.Batching

type t = {
  func : T.value list -> T.value list;
  nin : int;
  nout : int;
  identity : float option;
}

let frompyfunc ?identity func ~nin ~nout = { func; nin; nout; identity }
let shape v = (Core.get_aval v).T.shape

let call u args =
  if List.length args <> u.nin then
    invalid_arg
      (Printf.sprintf "ufunc: expected %d inputs, got %d" u.nin
         (List.length args));
  let target = NL.broadcast_shapes_n (List.map shape args) in
  let count = Array.fold_left ( * ) 1 target in
  let flat =
    List.map (fun v -> NL.reshape (NL.broadcast_to v target) [| count |]) args
  in
  let outs = Batching.vmap u.func (List.map (fun _ -> Some 0) flat) flat in
  List.map (fun v -> NL.reshape v target) outs

let single u args =
  match call u args with
  | [ x ] -> x
  | _ -> invalid_arg "ufunc: this method needs a single output function"

let check_binary u name =
  if u.nin <> 2 || u.nout <> 1 then
    invalid_arg (name ^ " only supported for functions with nin=2 and nout=1")

let slice_along axis v i = Lax_api.index_in_dim ~axis ~keepdims:false v i

let reduce ?(axis = 0) u a =
  check_binary u "reduce";
  let sh = shape a in
  let rank = Array.length sh in
  let axis = if axis < 0 then axis + rank else axis in
  let n = sh.(axis) in
  let first =
    if n > 0 then slice_along axis a 0
    else
      match u.identity with
      | Some x ->
          let rest =
            Array.of_list
              (List.filteri (fun i _ -> i <> axis) (Array.to_list sh))
          in
          Lax_api.full rest x (Core.get_aval a).T.dtype
      | None -> invalid_arg "reduce: zero-size input needs an identity"
  in
  List.fold_left
    (fun acc i -> single u [ acc; slice_along axis a i ])
    first
    (List.init (Stdlib.max 0 (n - 1)) (fun i -> i + 1))

let accumulate ?(axis = 0) u a =
  check_binary u "accumulate";
  let sh = shape a in
  let rank = Array.length sh in
  let axis = if axis < 0 then axis + rank else axis in
  let n = sh.(axis) in
  let _, items =
    List.fold_left
      (fun (acc, out) i ->
        let x = slice_along axis a i in
        let next =
          match acc with None -> x | Some carry -> single u [ carry; x ]
        in
        (Some next, next :: out))
      (None, []) (List.init n Fun.id)
  in
  NL.stack ~axis (List.rev items)

let outer u a b =
  check_binary u "outer";
  let sa = shape a and sb = shape b in
  let ones n = Array.make n 1 in
  single u
    [
      NL.reshape a (Array.append sa (ones (Array.length sb)));
      NL.reshape b (Array.append (ones (Array.length sa)) sb);
    ]
