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

let minimize f x0 ~method_ ?maxiter ?gtol ?norm () =
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
