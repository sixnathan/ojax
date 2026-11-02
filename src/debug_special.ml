module D = Dtype

let needs_check () =
  Config.value Config.debug_infs || Config.value Config.debug_nans

let is_inexact = function
  | D.F16 | D.Bf16 | D.F32 | D.F64 | D.Complex64 | D.Complex128 | D.F4_e2m1fn
  | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz
  | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu ->
      true
  | _ -> false

let exists_element pred nd =
  match Ndarray.dtype nd with
  | D.Complex64 | D.Complex128 ->
      let shape = Ndarray.shape nd in
      let n = Array.fold_left ( * ) 1 shape in
      let rank = Array.length shape in
      let found = ref false in
      for flat = 0 to n - 1 do
        let idx = Array.make rank 0 in
        let rest = ref flat in
        for k = rank - 1 downto 0 do
          idx.(k) <- !rest mod shape.(k);
          rest := !rest / shape.(k)
        done;
        let c = Ndarray.get_c nd idx in
        if pred c.Complex.re || pred c.Complex.im then found := true
      done;
      !found
  | _ -> Ndarray.fold (fun acc x -> acc || pred x) false nd

let is_inf x = Float.abs x = Float.infinity

let special_of nd =
  if not (is_inexact (Ndarray.dtype nd)) then None
  else if Config.value Config.debug_nans && exists_element Float.is_nan nd then
    Some "nan"
  else if Config.value Config.debug_infs && exists_element is_inf nd then
    Some "inf"
  else None

let first_special nds = List.find_map special_of nds

let raise_special name ty =
  raise
    (Py_exn.Floating_point_error
       (Printf.sprintf "invalid value (%s) encountered in %s" ty name))

let check name nds =
  if needs_check () then
    match first_special nds with Some ty -> raise_special name ty | None -> ()

let deoptimized_message inner =
  inner
  ^ ". Because jax_config.debug_nans.value and/or config.jax_debug_infs is \
     set, the de-optimized function (i.e., the function as if the `jit` \
     decorator were removed) was called in an attempt to get a more precise \
     error message. However, the de-optimized function did not produce invalid \
     values during its execution. This behavior can result from `jit` \
     optimizations causing the invalid value to be produced. It may also arise \
     from having nan/inf literals as inputs or outputs, like `jax.jit(lambda \
     ...: jax.numpy.nan)(...)`. \n\n\
     It may be possible to avoid the invalid value by removing the `jit` \
     decorator, at the cost of losing optimizations. \n\n\
     If you see this error, consider opening a bug report at \
     https://github.com/jax-ml/jax."

let rerun_notice =
  "Invalid nan value encountered in the output of a jax.jit function. Calling \
   the de-optimized version."

let notice_sink : (string -> unit) ref = ref print_endline

let check_jit ~name ~rerun outs =
  if needs_check () then
    match first_special outs with
    | None -> ()
    | Some ty ->
        !notice_sink rerun_notice;
        (try rerun () with
        | Py_exn.Floating_point_error _ as e -> raise e
        | Py_exn.Zero_division_error _ as e -> raise e);
        raise
          (Py_exn.Floating_point_error
             (deoptimized_message (Printf.sprintf "('jit(%s)', '%s')" name ty)))
