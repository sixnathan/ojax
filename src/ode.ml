module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module NLIN = Numpy.Linalg
module JU = Numpy.Jnp_util
module G = Numpy.Getitem
module R = Numpy.Reductions
module TU = Tree_util

type mxstep = Mx_int of int | Mx_float of float

let dtype_of v = (C.get_aval v).T.dtype
let shape_of v = (C.get_aval v).T.shape
let weak_of v = (C.get_aval v).T.weak_type
let pf = JU.py_float
let pi = JU.py_int
let ( +! ) = U.add
let ( -! ) = U.subtract
let ( *! ) = U.multiply
let ( /! ) = U.true_divide

let astype v dt =
  if dtype_of v = dt && not (weak_of v) then v else LN.astype v dt

let dot a b = TC.dot a b
let norm v = NLIN.norm v
let where = JU.where
let all = G.Slice { start = None; stop = None; step = None }
let take v ix = G.take v ix

let rec concat_tree arrs =
  if List.length arrs > 16 then
    let rec chunks = function
      | [] -> []
      | l ->
          let a = List.filteri (fun i _ -> i < 16) l in
          let b = List.filteri (fun i _ -> i >= 16) l in
          Lax_api.concatenate a 0 :: chunks b
    in
    concat_tree (chunks arrs)
  else Lax_api.concatenate arrs 0

let array_list vs =
  concat_tree (List.map (fun v -> Lax_api.expand_dims v [| 0 |]) vs)

let staged v =
  match !Core.dynamic_trace with
  | Some t when t.T.kind = T.KJaxpr && t.T.level > 0 -> Core.full_raise t v
  | _ -> v

let vec dt xs = staged (T.Concrete (Nd.of_floats dt [| Array.length xs |] xs))

let promote_dtypes_inexact = function
  | [ a; b ] ->
      let target = LN.result_type [ a; b ] in
      let target =
        if Dtypes.issubdtype target `Inexact then target
        else Dtypes.default_float_dtype ()
      in
      let conv v = if dtype_of v = target then v else LN.astype v target in
      [ conv a; conv b ]
  | l -> l

let fit_4th_order_polynomial y0 y1 y_mid dy0 dy1 dt =
  let dt = astype dt (dtype_of y0) in
  let a =
    (pf (-2.) *! dt *! dy0)
    +! (pf 2. *! dt *! dy1)
    -! (pf 8. *! y0)
    -! (pf 8. *! y1)
    +! (pf 16. *! y_mid)
  in
  let b =
    (pf 5. *! dt *! dy0)
    -! (pf 3. *! dt *! dy1)
    +! (pf 18. *! y0)
    +! (pf 14. *! y1)
    -! (pf 32. *! y_mid)
  in
  let c =
    (pf (-4.) *! dt *! dy0)
    +! (dt *! dy1)
    -! (pf 11. *! y0)
    -! (pf 5. *! y1)
    +! (pf 16. *! y_mid)
  in
  let d = dt *! dy0 in
  let e = y0 in
  [ a; b; c; d; e ]

let interp_fit_dopri y0 y1 k dt =
  let dps_c_mid =
    vec (dtype_of y0)
      [|
        6025192743. /. 30085553152. /. 2.;
        0.;
        51252292925. /. 65400821598. /. 2.;
        -2691868925. /. 45128329728. /. 2.;
        187940372067. /. 1594534317056. /. 2.;
        -1776094331. /. 19743644256. /. 2.;
        11237099. /. 235043384. /. 2.;
      |]
  in
  let y_mid = y0 +! (astype dt (dtype_of y0) *! dot dps_c_mid k) in
  array_list
    (fit_4th_order_polynomial y0 y1 y_mid (take k [ G.Int 0 ])
       (take k [ G.Int (-1) ]) dt)

let initial_step_size fun_ t0 y0 order rtol atol f0 =
  let y0, f0 =
    match promote_dtypes_inexact [ y0; f0 ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let dtype = dtype_of y0 in
  let scale = pf atol +! (U.abs y0 *! pf rtol) in
  let d0 = norm (y0 /! astype scale dtype) in
  let d1 = norm (f0 /! astype scale dtype) in
  let h0 =
    where
      (U.bitwise_or (U.less d0 (pf 1e-5)) (U.less d1 (pf 1e-5)))
      (pf 1e-6)
      (pf 0.01 *! d0 /! d1)
  in
  let y1 = y0 +! (astype h0 dtype *! f0) in
  let f1 = fun_ y1 (t0 +! h0) in
  let d2 = norm ((f1 -! f0) /! astype scale dtype) /! h0 in
  let h1 =
    where
      (U.bitwise_and (U.less_equal d1 (pf 1e-15)) (U.less_equal d2 (pf 1e-15)))
      (U.maximum (pf 1e-6) (h0 *! pf 1e-3))
      (U.power
         (pf 0.01 /! U.maximum d1 d2)
         (pf (1. /. (float_of_int order +. 1.))))
  in
  U.minimum (pf 100. *! h0) h1

let runge_kutta_step func y0 f0 t0 dt =
  let fdt = dtype_of f0 in
  let alpha =
    vec (dtype_of dt) [| 1. /. 5.; 3. /. 10.; 4. /. 5.; 8. /. 9.; 1.; 1.; 0. |]
  in
  let beta =
    staged
    @@ T.Concrete
         (Nd.of_floats fdt [| 6; 7 |]
            [|
              1. /. 5.;
              0.;
              0.;
              0.;
              0.;
              0.;
              0.;
              3. /. 40.;
              9. /. 40.;
              0.;
              0.;
              0.;
              0.;
              0.;
              44. /. 45.;
              -56. /. 15.;
              32. /. 9.;
              0.;
              0.;
              0.;
              0.;
              19372. /. 6561.;
              -25360. /. 2187.;
              64448. /. 6561.;
              -212. /. 729.;
              0.;
              0.;
              0.;
              9017. /. 3168.;
              -355. /. 33.;
              46732. /. 5247.;
              49. /. 176.;
              -5103. /. 18656.;
              0.;
              0.;
              35. /. 384.;
              0.;
              500. /. 1113.;
              125. /. 192.;
              -2187. /. 6784.;
              11. /. 84.;
              0.;
            |])
  in
  let c_sol =
    vec fdt
      [|
        35. /. 384.;
        0.;
        500. /. 1113.;
        125. /. 192.;
        -2187. /. 6784.;
        11. /. 84.;
        0.;
      |]
  in
  let c_error =
    vec fdt
      [|
        (35. /. 384.) -. (1951. /. 21600.);
        0.;
        (500. /. 1113.) -. (22642. /. 50085.);
        (125. /. 192.) -. (451. /. 720.);
        (-2187. /. 6784.) -. (-12231. /. 42400.);
        (11. /. 84.) -. (649. /. 6300.);
        -1. /. 60.;
      |]
  in
  let body_fun i = function
    | [ k ] ->
        let ti = t0 +! (dt *! take alpha [ G.Arr (i -! pi 1) ]) in
        let yi =
          y0 +! (astype dt fdt *! dot (take beta [ G.Arr (i -! pi 1); all ]) k)
        in
        let ft = func yi ti in
        [ G.at_set k [ G.Arr i; all ] ft ]
    | _ -> assert false
  in
  let k =
    G.at_set
      (Numpy.Array_creation.zeros ~dtype:fdt [| 7; (shape_of f0).(0) |])
      [ G.Int 0; all ] f0
  in
  let k =
    match Lax_api.fori_loop (pi 1) (pi 7) body_fun [ k ] with
    | [ k ] -> k
    | _ -> assert false
  in
  let y1 = (astype dt fdt *! dot c_sol k) +! y0 in
  let y1_error = astype dt fdt *! dot c_error k in
  let f1 = take k [ G.Int (-1) ] in
  (y1, f1, y1_error, k)

let abs2 x =
  if Dtypes.issubdtype (dtype_of x) `Complex then
    U.power (U.real x) (pi 2) +! U.power (U.imag x) (pi 2)
  else U.power x (pi 2)

let mean_error_ratio error_estimate rtol atol y0 y1 =
  let err_tol = pf atol +! (pf rtol *! U.maximum (U.abs y0) (U.abs y1)) in
  let err_ratio = error_estimate /! astype err_tol (dtype_of error_estimate) in
  U.sqrt (R.mean (abs2 err_ratio))

let optimal_step_size last_step mean_error_ratio =
  let safety = 0.9 and ifactor = 10.0 and dfactor = 0.2 and order = 5.0 in
  let dfactor = where (U.less mean_error_ratio (pi 1)) (pf 1.0) (pf dfactor) in
  let factor =
    U.minimum (pf ifactor)
      (U.maximum
         (U.power mean_error_ratio (pf (-1.0 /. order)) *! pf safety)
         dfactor)
  in
  where
    (U.equal mean_error_ratio (pi 0))
    (last_step *! pf ifactor)
    (last_step *! factor)

let mx_value = function Mx_int i -> pi i | Mx_float f -> pf f
let clip hmax v = LN.clip_values ~method_:true ~min:(pf 0.) ~max:(pf hmax) v
let with_highest f = Api.default_matmul_precision "highest" f

let odeint_core func rtol atol mxstep hmax y0 ts args =
  with_highest @@ fun () ->
  let func_ y t = func y t args in
  let scan_fun = function
    | [ y; f; t; dt; last_t; interp_coeff; target_t ] ->
        let cond_fun = function
          | [ i; _; _; t; dt; _; _ ] ->
              U.bitwise_and
                (U.bitwise_and (U.less t target_t) (U.less i (mx_value mxstep)))
                (U.greater dt (pi 0))
          | _ -> assert false
        in
        let body_fun = function
          | [ i; y; f; t; dt; last_t; interp_coeff ] ->
              let next_y, next_f, next_y_error, k =
                runge_kutta_step func_ y f t dt
              in
              let next_t = t +! dt in
              let error_ratio =
                mean_error_ratio next_y_error rtol atol y next_y
              in
              let new_interp_coeff = interp_fit_dopri y next_y k dt in
              let dt = clip hmax (optimal_step_size dt error_ratio) in
              let nw =
                [ i +! pi 1; next_y; next_f; next_t; dt; t; new_interp_coeff ]
              in
              let old = [ i +! pi 1; y; f; t; dt; last_t; interp_coeff ] in
              let c = U.less_equal error_ratio (pf 1.) in
              List.map2 (fun a b -> where c a b) nw old
          | _ -> assert false
        in
        let carry =
          match
            Lax_api.while_loop cond_fun body_fun
              [ pi 0; y; f; t; dt; last_t; interp_coeff ]
          with
          | _ :: carry -> carry
          | [] -> assert false
        in
        let t, last_t, interp_coeff =
          match carry with
          | [ _; _; t; _; last_t; interp_coeff ] -> (t, last_t, interp_coeff)
          | _ -> assert false
        in
        let relative_output_time = (target_t -! last_t) /! (t -! last_t) in
        let y_target =
          Numpy.Polynomial.polyval interp_coeff
            (astype relative_output_time (dtype_of interp_coeff))
        in
        carry @ [ y_target ]
    | _ -> assert false
  in
  let ts0 = take ts [ G.Int 0 ] in
  let f0 = func_ y0 ts0 in
  let ts0 = take ts [ G.Int 0 ] in
  let dt = clip hmax (initial_step_size func_ ts0 y0 4 rtol atol f0) in
  let interp_coeff = array_list [ y0; y0; y0; y0; y0 ] in
  let init_carry =
    [ y0; f0; take ts [ G.Int 0 ]; dt; take ts [ G.Int 0 ]; interp_coeff ]
  in
  let ys =
    match
      Lax_api.scan scan_fun init_carry
        [ take ts [ G.Slice { start = Some 1; stop = None; step = None } ] ]
    with
    | [ _; _; _; _; _; _; ys ] -> ys
    | _ -> assert false
  in
  LN.concatenate [ take y0 [ G.Newaxis ]; ys ]

let tuple l = TU.Tuple (List.map (fun v -> TU.Leaf v) l)

let rec odeint ?(rtol = 1.4e-8) ?(atol = 1.4e-8) ?(mxstep = Mx_float infinity)
    ?(hmax = infinity) func y0 t args =
  if not (Dtypes.issubdtype (dtype_of t) `Floating) then
    Py_exn.type_error
      (Printf.sprintf "t must be an array of floats, but got %s."
         (Numpy.Extras.array_str t));
  let y_leaves, y_def = TU.tree_flatten y0 in
  let ny = List.length y_leaves and nargs = List.length args in
  let t0 = take t [ G.Int 0 ] in
  let out_def = ref TU.Leaf_def in
  let flat_fun vals =
    let ys = List.filteri (fun i _ -> i < ny) vals in
    let tt = List.nth vals ny in
    let rest = List.filteri (fun i _ -> i > ny) vals in
    let out = func (TU.tree_unflatten y_def ys) tt rest in
    let leaves, def = TU.tree_flatten out in
    out_def := def;
    leaves
  in
  let converted, _ =
    Custom_derivatives.closure_convert flat_fun
      (List.map C.get_aval ((y_leaves @ [ t0 ]) @ args))
  in
  let out_def = !out_def in
  let converted y tt rest =
    TU.tree_unflatten out_def (converted ((TU.tree_leaves y @ [ tt ]) @ rest))
  in
  let wrapper vals =
    with_highest @@ fun () ->
    let ys = List.filteri (fun i _ -> i < ny) vals in
    let ts = List.nth vals ny in
    let rest = List.filteri (fun i _ -> i > ny) vals in
    let y0f, unravel = Flatten_util.ravel_pytree (TU.tree_unflatten y_def ys) in
    let func y tt rest =
      fst (Flatten_util.ravel_pytree (converted (unravel y) tt rest))
    in
    let out = odeint_custom func rtol atol mxstep hmax y0f ts rest in
    TU.tree_leaves
      (Api.vmap
         (function [ TU.Leaf v ] -> unravel v | _ -> assert false)
         [ Some 0 ] [ TU.Leaf out ])
  in
  ignore nargs;
  TU.tree_unflatten y_def
    (Api.jit_flat ~name:"_odeint_wrapper" wrapper ((y_leaves @ [ t ]) @ args))

and odeint_custom func rtol atol mxstep hmax y0 ts args =
  let n = List.length args in
  let split = function
    | y0 :: ts :: args -> (y0, ts, args)
    | _ -> assert false
  in
  let primal vals =
    let y0, ts, args = split vals in
    [ odeint_core func rtol atol mxstep hmax y0 ts args ]
  in
  let fwd vals =
    let ys = primal vals in
    let _, ts, args = split vals in
    (ys, (ys @ [ ts ]) @ args)
  in
  let bwd res g =
    match (res, g) with
    | ys :: ts :: args, [ g ] ->
        odeint_rev func rtol atol mxstep hmax ys ts args g
    | _ -> assert false
  in
  ignore n;
  match
    Custom_derivatives.custom_vjp ~name:"_odeint" ~fwd_name:"_odeint_fwd"
      ~bwd_name:"_odeint_rev" primal ~fwd ~bwd (y0 :: ts :: args)
  with
  | [ v ] -> v
  | _ -> assert false

and odeint_rev func rtol atol mxstep hmax ys ts args g =
  with_highest @@ fun () ->
  let nargs = List.length args in
  let aug_dynamics state t args =
    match state with
    | TU.Tuple (TU.Leaf y :: TU.Leaf y_bar :: _) ->
        let y_dot, vjpfun =
          Api.vjp
            (fun l ->
              match
                List.map (function TU.Leaf v -> v | _ -> assert false) l
              with
              | y :: t :: args -> TU.Leaf (func y t args)
              | [] | [ _ ] -> assert false)
            (List.map (fun v -> TU.Leaf v) ((y :: [ U.negative t ]) @ args))
        in
        let y_dot = match y_dot with TU.Leaf v -> v | _ -> assert false in
        let cts =
          List.map
            (function TU.Leaf v -> v | _ -> assert false)
            (vjpfun (TU.Leaf y_bar))
        in
        tuple (U.negative y_dot :: cts)
    | _ -> assert false
  in
  let real v =
    if Dtypes.issubdtype (dtype_of v) `Complex then U.real v else v
  in
  let scan_fun = function
    | y_bar :: t0_bar :: rest -> (
        let args_bar = List.filteri (fun j _ -> j < nargs) rest in
        let i = List.nth rest nargs in
        let ysi = take ys [ G.Arr i ] in
        let tsi = take ts [ G.Arr i ] in
        let gi = take g [ G.Arr i ] in
        let t_bar = real (dot (func ysi tsi args) gi) in
        let t0_bar = t0_bar -! t_bar in
        let state =
          TU.Tuple
            [ TU.Leaf ysi; TU.Leaf y_bar; TU.Leaf t0_bar; tuple args_bar ]
        in
        let tspan =
          array_list
            [ U.negative tsi; U.negative (take ts [ G.Arr (i -! pi 1) ]) ]
        in
        let out =
          odeint ~rtol ~atol ~mxstep ~hmax aug_dynamics state tspan args
        in
        let second v = take v [ G.Int 1 ] in
        match out with
        | TU.Tuple [ _; TU.Leaf y_bar; TU.Leaf t0_bar; TU.Tuple args_bar ] ->
            let y_bar = second y_bar in
            let t0_bar = second t0_bar in
            let args_bar =
              List.map
                (function TU.Leaf v -> second v | _ -> assert false)
                args_bar
            in
            let y_bar = y_bar +! take g [ G.Arr (i -! pi 1) ] in
            (y_bar :: t0_bar :: args_bar) @ [ t_bar ]
        | _ -> assert false)
    | _ -> assert false
  in
  let init_carry =
    take g [ G.Int (-1) ] :: pf 0.
    :: List.map (fun a -> Numpy.Array_creation.zeros_like a) args
  in
  let n = (shape_of ts).(0) in
  let idx =
    LN.arange
      ~start:(float_of_int (n - 1))
      ~step:(-1.)
      ~dtype:(Dtypes.default_int_dtype ())
      0.
  in
  match Lax_api.scan scan_fun init_carry [ idx ] with
  | y_bar :: t0_bar :: rest ->
      let args_bar = List.filteri (fun j _ -> j < nargs) rest in
      let rev_ts_bar = List.nth rest nargs in
      let ts_bar =
        LN.concatenate
          [
            array_list [ t0_bar ];
            take rev_ts_bar
              [ G.Slice { start = None; stop = None; step = Some (-1) } ];
          ]
      in
      y_bar :: ts_bar :: args_bar
  | _ -> assert false
