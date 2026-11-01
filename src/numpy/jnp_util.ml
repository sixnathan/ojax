module C = Core
module T = Types
module D = Dtype

let shape v = (C.get_aval v).T.shape
let dtype v = (C.get_aval v).T.dtype
let weak v = (C.get_aval v).T.weak_type
let jit name ?key f args = Jaxpr.named_jit ~name ?key f args

let jit1 name ?key f args =
  match Jaxpr.named_jit ~name ?key (fun a -> [ f a ]) args with
  | [ v ] -> v
  | _ -> invalid_arg "Jnp_util.jit1: arity"

let broadcast_to v sh =
  let s = shape v in
  if s = sh then v
  else begin
    let nlead = Array.length sh - Array.length s in
    if nlead < 0 then invalid_arg "broadcast_to: incompatible shapes";
    let kept = ref [] and squeezed = ref [] in
    Array.iteri
      (fun i d ->
        if d = sh.(i + nlead) then kept := (i + nlead) :: !kept
        else if d = 1 then squeezed := i :: !squeezed
        else invalid_arg "broadcast_to: incompatible shapes")
      s;
    let operand =
      match !squeezed with
      | [] -> v
      | dims -> C.bind1 (T.Squeeze (Array.of_list (List.rev dims))) [ v ]
    in
    C.bind1
      (T.Broadcast_in_dim { shape = sh; dims = Array.of_list (List.rev !kept) })
      [ operand ]
  end

let broadcast_arrays vs =
  match vs with
  | [] -> []
  | first :: rest ->
      if List.for_all (fun v -> shape v = shape first) rest then vs
      else
        let sh = C.elementwise_shape (List.map shape vs) in
        List.map (fun v -> broadcast_to v sh) vs

let where c x y =
  jit1 "_where"
    (function
      | [ c; x; y ] -> (
          let c =
            if dtype c = D.Bool then c
            else C.bind1 T.Ne [ c; C.splat (dtype c) [||] 0.0 ]
          in
          let dt, _ =
            Dtypes.result_type [ (dtype x, weak x); (dtype y, weak y) ]
          in
          let conv v =
            if dtype v = dt then v
            else C.bind1 (T.Convert_element_type dt) [ v ]
          in
          let x = conv x in
          let y = conv y in
          if Array.length (shape c) = 0 then
            match broadcast_arrays [ x; y ] with
            | [ a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false
          else
            match broadcast_arrays [ c; x; y ] with
            | [ c; a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false)
      | _ -> invalid_arg "where: arity")
    [ c; x; y ]
