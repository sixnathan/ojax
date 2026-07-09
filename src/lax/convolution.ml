open Types

let take shape perm = Array.map (fun p -> shape.(p)) perm
let dilate_dim d f = if d = 0 then 0 else ((d - 1) * f) + 1
let stride_dim d w s = if d < w then 0 else ((d - w) / s) + 1

let out_spatial l_sp k_sp padding lhs_dilation rhs_dilation window_strides =
  Array.mapi
    (fun i l ->
      let lo, hi = padding.(i) in
      let el = dilate_dim l lhs_dilation.(i) in
      let ek = dilate_dim k_sp.(i) rhs_dilation.(i) in
      stride_dim (lo + hi + el) ek window_strides.(i))
    l_sp

let conv_shape (dn : conv_dims) window_strides padding lhs_dilation rhs_dilation
    _feature_group_count batch_group_count lhs_shape rhs_shape =
  let ndim = Array.length lhs_shape in
  let ns = ndim - 2 in
  let lhs_can = take lhs_shape dn.lhs_spec in
  let rhs_can = take rhs_shape dn.rhs_spec in
  let l_sp = Array.sub lhs_can 2 ns in
  let k_sp = Array.sub rhs_can 2 ns in
  let out_sp =
    out_spatial l_sp k_sp padding lhs_dilation rhs_dilation window_strides
  in
  let out_batch =
    if batch_group_count > 1 then lhs_can.(0) / batch_group_count
    else lhs_can.(0)
  in
  let out_can = Array.append [| out_batch; rhs_can.(0) |] out_sp in
  let inv = Utils.argsort dn.out_spec in
  take out_can inv

let conv_impl (dn : conv_dims) window_strides padding lhs_dilation rhs_dilation
    feature_group_count batch_group_count lhs rhs =
  if batch_group_count <> 1 then
    failwith "lax: conv_general_dilated batch_group_count>1 deferred (M2)";
  let lshape = Ndarray.shape lhs in
  let rshape = Ndarray.shape rhs in
  let ndim = Array.length lshape in
  let ns = ndim - 2 in
  let lhs_can = take lshape dn.lhs_spec in
  let rhs_can = take rshape dn.rhs_spec in
  let bn = lhs_can.(0) in
  let cout = rhs_can.(0) in
  let cin_g = rhs_can.(1) in
  let l_sp = Array.sub lhs_can 2 ns in
  let k_sp = Array.sub rhs_can 2 ns in
  let out_sp =
    out_spatial l_sp k_sp padding lhs_dilation rhs_dilation window_strides
  in
  let out_can = Array.append [| bn; cout |] out_sp in
  let out_shape = take out_can (Utils.argsort dn.out_spec) in
  let out_str = Utils.strides out_shape in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  let cout_g = cout / feature_group_count in
  let n_osp = Utils.prod out_sp in
  let n_ksp = Utils.prod k_sp in
  let lhs_idx = Array.make ndim 0 in
  let rhs_idx = Array.make ndim 0 in
  let out_idx = Array.make ndim 0 in
  for n = 0 to bn - 1 do
    for o = 0 to cout - 1 do
      let g = o / cout_g in
      for of_ = 0 to n_osp - 1 do
        let osp = Utils.decode of_ out_sp in
        let acc = ref 0.0 in
        for j = 0 to cin_g - 1 do
          let cin_idx = (g * cin_g) + j in
          for kf = 0 to n_ksp - 1 do
            let ksp = Utils.decode kf k_sp in
            let valid = ref true in
            let lsp = Array.make ns 0 in
            let i = ref 0 in
            while !valid && !i < ns do
              let lo, _ = padding.(!i) in
              let p =
                (osp.(!i) * window_strides.(!i))
                + (ksp.(!i) * rhs_dilation.(!i))
                - lo
              in
              let dl = lhs_dilation.(!i) in
              if p < 0 || p mod dl <> 0 then valid := false
              else begin
                let li = p / dl in
                if li >= l_sp.(!i) then valid := false else lsp.(!i) <- li
              end;
              incr i
            done;
            if !valid then begin
              lhs_idx.(dn.lhs_spec.(0)) <- n;
              lhs_idx.(dn.lhs_spec.(1)) <- cin_idx;
              for d = 0 to ns - 1 do
                lhs_idx.(dn.lhs_spec.(d + 2)) <- lsp.(d)
              done;
              rhs_idx.(dn.rhs_spec.(0)) <- o;
              rhs_idx.(dn.rhs_spec.(1)) <- j;
              for d = 0 to ns - 1 do
                rhs_idx.(dn.rhs_spec.(d + 2)) <- ksp.(d)
              done;
              acc :=
                !acc +. (Ndarray.get_f lhs lhs_idx *. Ndarray.get_f rhs rhs_idx)
            end
          done
        done;
        out_idx.(dn.out_spec.(0)) <- n;
        out_idx.(dn.out_spec.(1)) <- o;
        for d = 0 to ns - 1 do
          out_idx.(dn.out_spec.(d + 2)) <- osp.(d)
        done;
        let flat = ref 0 in
        for d = 0 to ndim - 1 do
          flat := !flat + (out_idx.(d) * out_str.(d))
        done;
        out.(!flat) <- !acc
      done
    done
  done;
  Ndarray.of_floats (Ndarray.dtype lhs) out_shape out
