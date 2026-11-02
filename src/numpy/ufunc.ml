module T = Types
module D = Dtype
module NL = Lax_numpy
module Batching = Interpreters.Batching

type t = {
  id : int;
  func : T.value list -> T.value list;
  nin : int;
  nout : int;
  identity : float option;
}

let next_id = ref 0

let frompyfunc ?identity func ~nin ~nout =
  incr next_id;
  { id = !next_id; func; nin; nout; identity }

let shape v = (Core.get_aval v).T.shape
let dtype v = (Core.get_aval v).T.dtype

let call_flat u args =
  let target = NL.broadcast_shapes_n (List.map shape args) in
  let count = Array.fold_left ( * ) 1 target in
  let flat =
    List.map (fun v -> NL.reshape (NL.broadcast_to v target) [| count |]) args
  in
  let outs = Batching.vmap u.func (List.map (fun _ -> Some 0) flat) flat in
  List.map (fun v -> NL.reshape v target) outs

let call u args =
  if List.length args <> u.nin then
    invalid_arg
      (Printf.sprintf "ufunc: expected %d inputs, got %d" u.nin
         (List.length args));
  if u.nout = 1 then
    [
      Jnp_util.jit1 "_call_vectorized" ~key:(string_of_int u.id)
        (Vectorize.vectorize (fun inner ->
             match u.func inner with
             | [ x ] -> x
             | _ -> invalid_arg "ufunc: expected a single output"))
        args;
    ]
  else call_flat u args

let single u args =
  match call u args with
  | [ x ] -> x
  | _ -> invalid_arg "ufunc: this method needs a single output function"

let check_binary u name =
  if u.nin <> 2 || u.nout <> 1 then
    invalid_arg (name ^ " only supported for functions with nin=2 and nout=1")

let one_of dt : T.aval = { shape = [||]; dtype = dt; weak_type = false }

let result_dtype u a =
  let dt = dtype a in
  let closed =
    Jaxpr.make_jaxpr [ one_of dt; one_of dt ] (fun args -> u.func args)
  in
  match closed.jaxpr.outs with
  | [ T.A_var v ] -> v.vaval.dtype
  | [ T.A_lit nd ] -> Ndarray.dtype nd
  | [ T.DropVar aval ] -> aval.dtype
  | _ -> invalid_arg "ufunc: expected a single output"

let as_dtype v dt = if dtype v = dt then v else NL.astype v dt

let canonical_axis axis a =
  let rank = Array.length (shape a) in
  if axis < 0 then axis + rank else axis

let front axis a = if axis = 0 then a else NL.moveaxis [| axis |] [| 0 |] a
let int_scalar v = Core.splat (Dtypes.default_int_dtype ()) [||] v

let weak_int v =
  match Core.splat (Dtypes.default_int_dtype ()) [||] v with
  | T.Concrete nd -> T.Concrete (Ndarray.as_weak nd)
  | other -> other

let at arr i =
  let dims = shape arr in
  let rank = Array.length dims in
  let starts =
    if rank > 1 then
      Core.convert_like ~weak:false i D.I32
      :: List.init (rank - 1) (fun _ -> Core.splat D.I32 [||] 0.0)
    else [ i ]
  in
  let normalize s d =
    let dt = dtype s in
    let negative = Core.bind1 T.Lt [ s; Core.splat dt [||] 0.0 ] in
    let shifted =
      Core.bind1 T.Add
        [
          Core.convert_like ~weak:false s dt;
          Core.splat dt [||] (float_of_int d);
        ]
    in
    Core.bind1 T.Select_n [ negative; s; shifted ]
  in
  let starts = List.mapi (fun d s -> normalize s dims.(d)) starts in
  let sizes = Array.mapi (fun d n -> if d = 0 then 1 else n) dims in
  Core.bind1 (T.Squeeze [| 0 |])
    [ Core.bind1 (T.Dynamic_slice { slice_sizes = sizes }) (arr :: starts) ]

let first arr =
  let dims = shape arr in
  let limit = Array.mapi (fun d n -> if d = 0 then 1 else n) dims in
  Core.bind1 (T.Squeeze [| 0 |])
    [
      Core.bind1
        (T.Slice
           {
             start_indices = Array.map (fun _ -> 0) dims;
             limit_indices = limit;
             strides = None;
           })
        [ arr ];
    ]

let bump i = Core.bind1 T.Add [ i; weak_int 1.0 ]

let reduce ?(axis = 0) u a =
  check_binary u "reduce";
  Eager_jit.one ~jit:"reduce" "numpy.ufunc.reduce" (u.id, axis) [] [ a ]
    (fun _ vals ->
      let a = List.hd vals in
      let axis = canonical_axis axis a in
      let dt = result_dtype u a in
      let arr = front axis a in
      let n = (shape arr).(0) in
      let rest = Array.sub (shape arr) 1 (Array.length (shape arr) - 1) in
      if n = 0 then
        match u.identity with
        | Some x -> Lax_api.full rest x dt
        | None -> invalid_arg "reduce: zero-size input needs an identity"
      else
        let start_index, start_value =
          match u.identity with
          | Some x -> (0, Lax_api.full rest x dt)
          | None -> (1, as_dtype (first arr) dt)
        in
        let body i = function
          | [ v ] -> [ single u [ v; as_dtype (at arr i) dt ] ]
          | _ -> invalid_arg "ufunc.reduce: arity"
        in
        match
          Lax_api.fori_loop
            (weak_int (float_of_int start_index))
            (weak_int (float_of_int n))
            body [ start_value ]
        with
        | [ result ] -> result
        | _ -> invalid_arg "ufunc.reduce: arity")

let accumulate ?(axis = 0) u a =
  check_binary u "accumulate";
  Eager_jit.one ~jit:"accumulate" "numpy.ufunc.accumulate" (u.id, axis) [] [ a ]
    (fun _ vals ->
      let a = List.hd vals in
      let axis = canonical_axis axis a in
      let dt = result_dtype u a in
      if Array.mem 0 (shape a) then
        Core.bind1
          (T.Broadcast_in_dim { shape = shape a; dims = [||] })
          [ Core.splat dt [||] 0.0 ]
      else
        let arr = front axis a in
        let body = function
          | [ i; x ] ->
              let is_first = Core.bind1 T.Eq [ i; weak_int 0.0 ] in
              let head = as_dtype (first arr) dt in
              let y =
                NL.where_ is_first head
                  (single u [ as_dtype x dt; as_dtype (at arr i) dt ])
              in
              [ bump i; y; y ]
          | _ -> invalid_arg "ufunc.accumulate: arity"
        in
        match
          Lax_api.scan
            ~length:(shape arr).(0)
            body
            [ weak_int 0.0; as_dtype (first arr) dt ]
            []
        with
        | [ _; _; stacked ] ->
            if axis = 0 then stacked else NL.moveaxis [| 0 |] [| axis |] stacked
        | _ -> invalid_arg "ufunc.accumulate: arity")

let outer u a b =
  check_binary u "outer";
  Eager_jit.one ~jit:"outer" "numpy.ufunc.outer" u.id [] [ a; b ] (fun _ vals ->
      match vals with
      | [ a; b ] ->
          let sa = shape a and sb = shape b in
          let ravel x =
            let count = Array.fold_left ( * ) 1 (shape x) in
            Lax_api.reshape x [| count |]
          in
          let inner args = [ single u args ] in
          let result =
            Batching.vmap
              (Batching.vmap inner [ None; Some 0 ])
              [ Some 0; None ]
              [ ravel a; ravel b ]
          in
          Lax_api.reshape (List.hd result) (Array.append sa sb)
      | _ -> assert false)
