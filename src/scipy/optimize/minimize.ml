module T = Types

type optimize_results = {
  x : T.value;
  success : bool;
  status : int;
  fun_ : T.value;
  jac : T.value;
  hess_inv : T.value option;
  nfev : int;
  njev : int;
  nit : int;
}

let bfgs_option_names = [ "maxiter"; "norm"; "gtol"; "line_search_maxiter" ]

let minimize f x0 ~method_ ?maxiter ?gtol ?norm ?(tol : float option)
    ?(options = ([] : (string * float) list)) () =
  ignore tol;
  List.iter
    (fun (name, _) ->
      if not (List.mem name bfgs_option_names) then
        raise
          (Py_exn.Type_error
             ("minimize_bfgs() got an unexpected keyword argument '" ^ name
            ^ "'")))
    options;
  let option_int name given =
    match List.assoc_opt name options with
    | Some v -> Some (int_of_float v)
    | None -> given
  in
  let option_float name given =
    match List.assoc_opt name options with Some v -> Some v | None -> given
  in
  let maxiter = option_int "maxiter" maxiter in
  let gtol = option_float "gtol" gtol in
  let norm = option_float "norm" norm in
  match String.lowercase_ascii method_ with
  | "bfgs" ->
      let r = Bfgs.minimize_bfgs f x0 ?maxiter ?norm ?gtol () in
      {
        x = r.Bfgs.x_k;
        success = r.Bfgs.converged && not r.Bfgs.failed;
        status = r.Bfgs.status;
        fun_ = r.Bfgs.f_k;
        jac = r.Bfgs.g_k;
        hess_inv = Some r.Bfgs.h_k;
        nfev = r.Bfgs.nfev;
        njev = r.Bfgs.ngev;
        nit = r.Bfgs.k;
      }
  | "l-bfgs-experimental-do-not-rely-on-this" ->
      let r = Lbfgs.minimize_lbfgs f x0 ?maxiter ?norm ?gtol () in
      {
        x = r.Lbfgs.x_k;
        success = r.Lbfgs.converged && not r.Lbfgs.failed;
        status = r.Lbfgs.status;
        fun_ = r.Lbfgs.f_k;
        jac = r.Lbfgs.g_k;
        hess_inv = None;
        nfev = r.Lbfgs.nfev;
        njev = r.Lbfgs.ngev;
        nit = r.Lbfgs.k;
      }
  | _ -> invalid_arg (Printf.sprintf "Method %s not recognized" method_)

let minimize_args f x0 ?(args = ([] : T.value list)) ~method_ ?maxiter ?gtol
    ?norm ?tol ?options () =
  minimize (fun x -> f x args) x0 ~method_ ?maxiter ?gtol ?norm ?tol ?options ()
