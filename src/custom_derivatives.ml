open Types

let custom_jvp primal ~jvp =
  let id = Core.register_custom_rule (Core.Jvp_rule { primal; jvp }) in
  fun args -> Core.bind (Custom_jvp_call id) args

let custom_vjp primal ~fwd ~bwd =
  let id = Core.register_custom_rule (Core.Vjp_rule { primal; fwd; bwd }) in
  fun args -> Core.bind (Custom_vjp_call id) args

let custom_gradient f =
  let pullbacks = Hashtbl.create 16 in
  let handle = ref 0 in
  let primal args = fst (f args) in
  let fwd args =
    let outs, pullback = f args in
    let key = !handle in
    incr handle;
    Hashtbl.replace pullbacks key pullback;
    let residual =
      Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int key |])
    in
    (outs, [ residual ])
  in
  let bwd res cts =
    match res with
    | [ Concrete nd ] ->
        let key = int_of_float (Ndarray.get_f nd [||]) in
        (Hashtbl.find pullbacks key) cts
    | _ -> failwith "custom_gradient: residual was transformed"
  in
  custom_vjp primal ~fwd ~bwd

let linear_call ~forward ~transpose =
  let id =
    Core.register_custom_rule (Core.Linear_rule { forward; transpose })
  in
  fun res linear ->
    let out_avals =
      let traced =
        Jaxpr.make_jaxpr
          (List.map Core.get_aval (res @ linear))
          (fun args ->
            match Util.split_list args [ List.length res ] with
            | [ r; x ] -> forward r x
            | _ -> failwith "linear_call: arity")
      in
      List.map
        (function
          | A_var v -> v.vaval
          | A_lit nd ->
              {
                shape = Ndarray.shape nd;
                dtype = Ndarray.dtype nd;
                weak_type = false;
              }
          | DropVar a -> a)
        traced.jaxpr.outs
    in
    Core.bind
      (Custom_lin { rule = id; n_res = List.length res; out_avals })
      (res @ linear)
