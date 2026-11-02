external available_ : unit -> bool = "ojax_gpu_solver_available"
external handler_ : string -> nativeint = "ojax_gpu_solver_handler"

let available () = available_ ()

let ffi_targets () =
  List.map
    (fun target -> (target, handler_ target))
    [
      "cusolver_getrf_ffi";
      "cusolver_geqrf_ffi";
      "cusolver_orgqr_ffi";
      "cusolver_ormqr_ffi";
      "cusolver_potrf_ffi";
      "cusolver_syevd_ffi";
      "cusolver_syrk_ffi";
      "cusolver_gesvdj_ffi";
      "cusolver_sytrd_ffi";
      "cusolver_geev_ffi";
      "cu_lu_pivots_to_permutation";
      "cu_cholesky_update_ffi";
      "cu_tridiagonal_solve_perturbed";
      "cuhybrid_geqp3";
    ]
