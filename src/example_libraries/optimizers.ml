open Types
module Tree = Tree_util
module L = Lax_api

type schedule = int -> float
type leaf_state = value list

type optimizer = {
  init : value Tree.t -> leaf_state Tree.t;
  update : int -> value Tree.t -> leaf_state Tree.t -> leaf_state Tree.t;
  get_params : leaf_state Tree.t -> value Tree.t;
}

let const_like v x = L.full_like v x
let zeros_like v = const_like v 0.0
let scale s v = L.mul (const_like v s) v
let add = L.add
let sub = L.sub
let mul = L.mul
let div = L.div
let sqrt_ = L.sqrt
let constant c _ = c

let exponential_decay step_size decay_steps decay_rate i =
  step_size *. (decay_rate ** (float_of_int i /. float_of_int decay_steps))

let inverse_time_decay ?(staircase = false) step_size decay_steps decay_rate i =
  let ratio = float_of_int i /. float_of_int decay_steps in
  let ratio = if staircase then Float.floor ratio else ratio in
  step_size /. (1.0 +. (decay_rate *. ratio))

let polynomial_decay ?(power = 1.0) step_size decay_steps final_step_size i =
  let step = Stdlib.min i decay_steps in
  let step_frac = 1.0 -. (float_of_int step /. float_of_int decay_steps) in
  ((step_size -. final_step_size) *. (step_frac ** power)) +. final_step_size

let piecewise_constant boundaries values i =
  let rec go bs vs =
    match (bs, vs) with
    | b :: brest, v :: vrest -> if i < b then v else go brest vrest
    | [], v :: _ -> v
    | _, [] -> invalid_arg "values must be one longer than boundaries"
  in
  go boundaries values

let of_leaf_rules ~init ~update ~get =
  {
    init = (fun params -> Tree.tree_map init params);
    update =
      (fun i grads state ->
        Tree.tree_map2 (fun g s -> update i g s) grads state);
    get_params = (fun state -> Tree.tree_map get state);
  }

let head = function x :: _ -> x | [] -> invalid_arg "optimizer: empty state"

let sgd (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x ])
    ~update:(fun i g s -> [ sub (head s) (scale (step_size i) g) ])
    ~get:head

let momentum (step_size : schedule) ~mass =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; v ] ->
          let v' = add (scale mass v) g in
          [ sub x (scale (step_size i) v'); v' ]
      | _ -> invalid_arg "momentum: state")
    ~get:head

let nesterov (step_size : schedule) ~mass =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; v ] ->
          let v' = add (scale mass v) g in
          [ sub x (scale (step_size i) (add (scale mass v') g)); v' ]
      | _ -> invalid_arg "nesterov: state")
    ~get:head

let adagrad ?(momentum = 0.9) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; gsq; m ] ->
          let gsq' = add gsq (mul g g) in
          let m' =
            add
              (scale (1.0 -. momentum) (div g (sqrt_ gsq')))
              (scale momentum m)
          in
          [ sub x (scale (step_size i) m'); gsq'; m' ]
      | _ -> invalid_arg "adagrad: state")
    ~get:head

let rmsprop ?(gamma = 0.9) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; avg ] ->
          let avg' = add (scale gamma avg) (scale (1.0 -. gamma) (mul g g)) in
          let denom = sqrt_ (add avg' (const_like avg' eps)) in
          [ sub x (scale (step_size i) (div g denom)); avg' ]
      | _ -> invalid_arg "rmsprop: state")
    ~get:head

let rmsprop_momentum ?(gamma = 0.9) ?(eps = 1e-8) ?(momentum = 0.9)
    (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; avg; mom ] ->
          let avg' = add (scale gamma avg) (scale (1.0 -. gamma) (mul g g)) in
          let denom = sqrt_ (add avg' (const_like avg' eps)) in
          let mom' =
            add (scale momentum mom) (scale (step_size i) (div g denom))
          in
          [ sub x mom'; avg'; mom' ]
      | _ -> invalid_arg "rmsprop_momentum: state")
    ~get:head

let adam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; m; v ] ->
          let m' = add (scale (1.0 -. b1) g) (scale b1 m) in
          let v' = add (scale (1.0 -. b2) (mul g g)) (scale b2 v) in
          let t = float_of_int (i + 1) in
          let mhat = scale (1.0 /. (1.0 -. (b1 ** t))) m' in
          let vhat = scale (1.0 /. (1.0 -. (b2 ** t))) v' in
          let denom = add (sqrt_ vhat) (const_like vhat eps) in
          [ sub x (scale (step_size i) (div mhat denom)); m'; v' ]
      | _ -> invalid_arg "adam: state")
    ~get:head

let adamax ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zeros_like x; zeros_like x ])
    ~update:(fun i g s ->
      match s with
      | [ x; m; u ] ->
          let m' = add (scale (1.0 -. b1) g) (scale b1 m) in
          let u' = L.max (scale b2 u) (L.abs g) in
          let t = float_of_int (i + 1) in
          let factor = step_size i /. (1.0 -. (b1 ** t)) in
          let denom = add u' (const_like u' eps) in
          [ sub x (scale factor (div m' denom)); m'; u' ]
      | _ -> invalid_arg "adamax: state")
    ~get:head

let sm3 ?(momentum = 0.9) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x ->
      let a = Core.get_aval x in
      x :: zeros_like x
      :: List.map
           (fun size -> L.full [| size |] 0.0 a.dtype)
           (Array.to_list a.shape))
    ~update:(fun i g s ->
      match s with
      | x :: m :: vs ->
          let shape = (Core.get_aval g).shape in
          let rank = Array.length shape in
          let expand axis v = L.broadcast_in_dim v shape [| axis |] in
          let smallest =
            match List.mapi expand vs with
            | first :: rest -> List.fold_left L.min first rest
            | [] -> zeros_like g
          in
          let accum = add smallest (mul g g) in
          let inverse =
            L.select
              (L.gt accum (zeros_like accum))
              (L.rsqrt accum) (zeros_like accum)
          in
          let m' =
            add (scale (1.0 -. momentum) (mul g inverse)) (scale momentum m)
          in
          let x' = sub x (scale (step_size i) m') in
          let vs' =
            List.init rank (fun j ->
                let axes =
                  Array.of_list
                    (List.filter (fun d -> d <> j) (List.init rank Fun.id))
                in
                if Array.length axes = 0 then accum else L.reduce_max accum axes)
          in
          x' :: m' :: vs'
      | _ -> invalid_arg "sm3: state")
    ~get:head

let l2_norm tree =
  let total =
    List.fold_left
      (fun acc g ->
        let rank = Array.length (Core.get_aval g).shape in
        let s = L.reduce_sum (mul g g) (Array.init rank Fun.id) in
        match acc with None -> Some s | Some a -> Some (add a s))
      None (Tree.tree_leaves tree)
  in
  match total with
  | Some t -> L.sqrt t
  | None -> invalid_arg "l2_norm: empty tree"

let clip_grads grad_tree max_norm =
  let norm = l2_norm grad_tree in
  let limit = const_like norm max_norm in
  let factor =
    L.select (L.lt norm limit) (const_like norm 1.0) (div limit norm)
  in
  Tree.tree_map
    (fun g ->
      let rank = Array.length (Core.get_aval g).shape in
      ignore rank;
      mul g (L.broadcast_in_dim factor (Core.get_aval g).shape [||]))
    grad_tree

type optimizer_state = {
  packed_state : value list list;
  tree_def : Tree.treedef;
  subtree_defs : Tree.treedef list;
}

type join_point = Join_point of value Tree.t

let optimizer_state_of state =
  let states_flat, tree_def = Tree.tree_flatten state in
  let subtrees =
    List.map
      (fun leaf ->
        Tree.tree_flatten (Tree.Tuple (List.map (fun x -> Tree.Leaf x) leaf)))
      states_flat
  in
  {
    packed_state = List.map fst subtrees;
    tree_def;
    subtree_defs = List.map snd subtrees;
  }

let leaf_state_of opt_state =
  Tree.tree_unflatten opt_state.tree_def opt_state.packed_state

let optimizer_state_leaves opt_state = List.concat opt_state.packed_state

let unpack_optimizer_state opt_state =
  let sentinels =
    List.map2
      (fun def leaves -> Join_point (Tree.tree_unflatten def leaves))
      opt_state.subtree_defs opt_state.packed_state
  in
  Tree.tree_unflatten opt_state.tree_def sentinels

let pack_optimizer_state marked =
  let sentinels, tree_def = Tree.tree_flatten marked in
  let subtrees =
    List.map (fun (Join_point subtree) -> Tree.tree_flatten subtree) sentinels
  in
  {
    packed_state = List.map fst subtrees;
    tree_def;
    subtree_defs = List.map snd subtrees;
  }

let make_schedule = function
  | `Schedule schedule -> schedule
  | `Scalar step_size -> constant step_size
  | `Value v -> (
      if Array.length (Core.get_aval v).shape <> 0 then
        invalid_arg "make_schedule: expected a scalar or a schedule";
      match v with
      | Concrete nd -> constant (Ndarray.get_f nd [||])
      | Device b -> constant (Ndarray.get_f (Pjrt.Buffer.to_host b) [||])
      | Tracer _ ->
          invalid_arg "make_schedule: a traced step size is not supported")
