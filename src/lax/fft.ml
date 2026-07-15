open Types

let pi = 4.0 *. atan 1.0
let prod a = Array.fold_left ( * ) 1 a

let decode flat shape =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let f = ref flat in
  for d = n - 1 downto 0 do
    idx.(d) <- !f mod shape.(d);
    f := !f / shape.(d)
  done;
  idx

let is_complex = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let complexify = function
  | Dtype.F32 -> Dtype.Complex64
  | Dtype.F64 -> Dtype.Complex128
  | Dtype.Complex64 -> Dtype.Complex64
  | Dtype.Complex128 -> Dtype.Complex128
  | _ -> Dtype.Complex64

let complex_basetype = function
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | d -> d

let to_inexact = function
  | Dtype.F32 -> Dtype.F32
  | Dtype.F64 -> Dtype.F64
  | Dtype.Complex64 -> Dtype.Complex64
  | Dtype.Complex128 -> Dtype.Complex128
  | _ -> Dtypes.default_float_dtype ()

let carray nd =
  let sh = Ndarray.shape nd in
  let n = prod sh in
  let dt = Ndarray.dtype nd in
  if is_complex dt then Array.init n (fun i -> Ndarray.get_c nd (decode i sh))
  else
    Array.init n (fun i ->
        { Complex.re = Ndarray.get_f nd (decode i sh); im = 0.0 })

let next_pow2 n =
  let m = ref 1 in
  while !m < n do
    m := !m * 2
  done;
  !m

let rec radix2 (a : Complex.t array) sign =
  let n = Array.length a in
  if n <= 1 then Array.copy a
  else begin
    let half = n / 2 in
    let ev = Array.init half (fun i -> a.(2 * i)) in
    let od = Array.init half (fun i -> a.((2 * i) + 1)) in
    let fev = radix2 ev sign and fod = radix2 od sign in
    let out = Array.make n Complex.zero in
    for k = 0 to half - 1 do
      let ang = sign *. 2.0 *. pi *. float_of_int k /. float_of_int n in
      let tw = Complex.mul { Complex.re = cos ang; im = sin ang } fod.(k) in
      out.(k) <- Complex.add fev.(k) tw;
      out.(k + half) <- Complex.sub fev.(k) tw
    done;
    out
  end

let bluestein (a : Complex.t array) sign =
  let n = Array.length a in
  let m = next_pow2 ((2 * n) - 1) in
  let w =
    Array.init n (fun k ->
        let ang = sign *. pi *. float_of_int (k * k) /. float_of_int n in
        { Complex.re = cos ang; im = sin ang })
  in
  let av = Array.make m Complex.zero in
  for k = 0 to n - 1 do
    av.(k) <- Complex.mul a.(k) w.(k)
  done;
  let bv = Array.make m Complex.zero in
  bv.(0) <- Complex.conj w.(0);
  for k = 1 to n - 1 do
    let c = Complex.conj w.(k) in
    bv.(k) <- c;
    bv.(m - k) <- c
  done;
  let fa = radix2 av (-1.0) and fb = radix2 bv (-1.0) in
  let cprod = Array.init m (fun i -> Complex.mul fa.(i) fb.(i)) in
  let conj_c = Array.map Complex.conj cprod in
  let ic = radix2 conj_c (-1.0) in
  Array.init n (fun k ->
      let v =
        Complex.div
          (Complex.conj ic.(k))
          { Complex.re = float_of_int m; im = 0.0 }
      in
      Complex.mul v w.(k))

let pocket_pair ca cb = (Complex.add ca cb, Complex.sub ca cb)

let pocket3 (a : Complex.t array) =
  let t0 = a.(0) in
  let t1 = Complex.add a.(1) a.(2) in
  let t2 = Complex.sub a.(1) a.(2) in
  let twr = -0.5 in
  let twi = -0.8660254037844386467637231707529362 in
  let ca = Complex.add t0 (Complex.mul { Complex.re = twr; im = 0.0 } t1) in
  let cb = { Complex.re = -.t2.im *. twi; im = t2.re *. twi } in
  let x1, x2 = pocket_pair ca cb in
  [| Complex.add t0 t1; x1; x2 |]

let pocket5_pair (t0 : Complex.t) (t1 : Complex.t) (t2 : Complex.t)
    (t3 : Complex.t) (t4 : Complex.t) twar twbr twai twbi =
  let ca =
    {
      Complex.re = t0.re +. (twar *. t1.re) +. (twbr *. t2.re);
      im = t0.im +. (twar *. t1.im) +. (twbr *. t2.im);
    }
  in
  let cb =
    {
      Complex.re = -.((twai *. t4.im) +. (twbi *. t3.im));
      im = (twai *. t4.re) +. (twbi *. t3.re);
    }
  in
  pocket_pair ca cb

let pocket5 (a : Complex.t array) =
  let tw1r = 0.3090169943749474241022934171828191 in
  let tw1i = -0.9510565162951535721164393333793821 in
  let tw2r = -0.8090169943749474241022934171828191 in
  let tw2i = -0.5877852522924731291687059546390728 in
  let t0 = a.(0) in
  let t1 = Complex.add a.(1) a.(4) in
  let t4 = Complex.sub a.(1) a.(4) in
  let t2 = Complex.add a.(2) a.(3) in
  let t3 = Complex.sub a.(2) a.(3) in
  let x1, x4 = pocket5_pair t0 t1 t2 t3 t4 tw1r tw2r tw1i tw2i in
  let x2, x3 = pocket5_pair t0 t1 t2 t3 t4 tw2r tw1r tw2i (-.tw1i) in
  let x0 =
    { Complex.re = t0.re +. t1.re +. t2.re; im = t0.im +. t1.im +. t2.im }
  in
  [| x0; x1; x2; x3; x4 |]

let pocket7_pair (t1 : Complex.t) (t2 : Complex.t) (t3 : Complex.t)
    (t4 : Complex.t) (t5 : Complex.t) (t6 : Complex.t) (t7 : Complex.t) x1 x2 x3
    y1 y2 y3 =
  let ca =
    {
      Complex.re = t1.re +. (x1 *. t2.re) +. (x2 *. t3.re) +. (x3 *. t4.re);
      im = t1.im +. (x1 *. t2.im) +. (x2 *. t3.im) +. (x3 *. t4.im);
    }
  in
  let cb =
    {
      Complex.re = -.((y1 *. t7.im) +. (y2 *. t6.im) +. (y3 *. t5.im));
      im = (y1 *. t7.re) +. (y2 *. t6.re) +. (y3 *. t5.re);
    }
  in
  pocket_pair ca cb

let pocket7 (a : Complex.t array) =
  let tw1r = 0.6234898018587335305250048840042398 in
  let tw1i = -0.7818314824680298087084445266740578 in
  let tw2r = -0.2225209339563144042889025644967948 in
  let tw2i = -0.9749279121818236070181316829939312 in
  let tw3r = -0.9009688679024191262361023195074451 in
  let tw3i = -0.433883739117558120475768332848359 in
  let t1 = a.(0) in
  let t2 = Complex.add a.(1) a.(6) in
  let t7 = Complex.sub a.(1) a.(6) in
  let t3 = Complex.add a.(2) a.(5) in
  let t6 = Complex.sub a.(2) a.(5) in
  let t4 = Complex.add a.(3) a.(4) in
  let t5 = Complex.sub a.(3) a.(4) in
  let x1, x6 =
    pocket7_pair t1 t2 t3 t4 t5 t6 t7 tw1r tw2r tw3r tw1i tw2i tw3i
  in
  let x2, x5 =
    pocket7_pair t1 t2 t3 t4 t5 t6 t7 tw2r tw3r tw1r tw2i (-.tw3i) (-.tw1i)
  in
  let x3, x4 =
    pocket7_pair t1 t2 t3 t4 t5 t6 t7 tw3r tw1r tw2r tw3i (-.tw1i) tw2i
  in
  let x0 =
    {
      Complex.re = t1.re +. t2.re +. t3.re +. t4.re;
      im = t1.im +. t2.im +. t3.im +. t4.im;
    }
  in
  [| x0; x1; x2; x3; x4; x5; x6 |]

let dft1 a sign =
  let n = Array.length a in
  let forward x =
    match n with
    | 3 -> pocket3 x
    | 5 -> pocket5 x
    | 7 -> pocket7 x
    | _ -> if n land (n - 1) = 0 then radix2 x (-1.0) else bluestein x (-1.0)
  in
  if sign < 0.0 then forward a
  else Array.map Complex.conj (forward (Array.map Complex.conj a))

let dft_axis (data : Complex.t array) shape axis inv =
  let ndim = Array.length shape in
  let n = shape.(axis) in
  let stride = ref 1 in
  for d = axis + 1 to ndim - 1 do
    stride := !stride * shape.(d)
  done;
  let stride = !stride in
  let total = Array.length data in
  let sign = if inv then 1.0 else -1.0 in
  let out = Array.copy data in
  for base = 0 to total - 1 do
    if base / stride mod n = 0 then begin
      let line = Array.init n (fun k -> data.(base + (k * stride))) in
      let r = dft1 line sign in
      for k = 0 to n - 1 do
        out.(base + (k * stride)) <- r.(k)
      done
    end
  done;
  out

let fft_impl fft_type fft_lengths inputs =
  match inputs with
  | [ x ] -> (
      let sh = Ndarray.shape x in
      let ndim = Array.length sh in
      let dt = Ndarray.dtype x in
      let k = Array.length fft_lengths in
      let axes = Array.init k (fun i -> ndim - k + i) in
      let data = ref (carray x) in
      let shape = ref (Array.copy sh) in
      let full_fft inv =
        for i = Array.length axes - 1 downto 0 do
          data := dft_axis !data !shape axes.(i) inv
        done
      in
      match fft_type with
      | Fft_fft ->
          full_fft false;
          [ Ndarray.of_complex dt sh !data ]
      | Fft_ifft ->
          full_fft true;
          let nf = float_of_int (prod fft_lengths) in
          let scaled =
            Array.map
              (fun c -> Complex.div c { Complex.re = nf; im = 0.0 })
              !data
          in
          [ Ndarray.of_complex dt sh scaled ]
      | Fft_rfft ->
          full_fft false;
          let nlast = fft_lengths.(k - 1) in
          let m = (nlast / 2) + 1 in
          let osh = Array.copy sh in
          osh.(ndim - 1) <- m;
          let outer = prod sh / sh.(ndim - 1) in
          let last = sh.(ndim - 1) in
          let out = Array.make (prod osh) Complex.zero in
          for o = 0 to outer - 1 do
            for j = 0 to m - 1 do
              out.((o * m) + j) <- !data.((o * last) + j)
            done
          done;
          [ Ndarray.of_complex (complexify dt) osh out ]
      | Fft_irfft ->
          let nlast = fft_lengths.(k - 1) in
          let mlast = sh.(ndim - 1) in
          for i = 0 to k - 2 do
            data := dft_axis !data !shape axes.(i) true
          done;
          let ysh = Array.copy sh in
          ysh.(ndim - 1) <- nlast;
          let y = Array.make (prod ysh) Complex.zero in
          let outer = prod sh / mlast in
          for o = 0 to outer - 1 do
            let bd = o * mlast and by = o * nlast in
            for j = 0 to nlast - 1 do
              y.(by + j) <-
                (if j < mlast then !data.(bd + j)
                 else Complex.conj !data.(bd + (nlast - j)))
            done
          done;
          let y = dft_axis y ysh (ndim - 1) true in
          let nf = float_of_int (prod fft_lengths) in
          let out = Array.map (fun c -> c.Complex.re /. nf) y in
          [ Ndarray.of_floats (complex_basetype dt) ysh out ])
  | _ -> failwith "fft: fft_p expects one operand"

let fft_aval fft_type fft_lengths avals =
  match avals with
  | [ a ] ->
      let sh = a.shape in
      let ndim = Array.length sh in
      let k = Array.length fft_lengths in
      let shape, dtype =
        match fft_type with
        | Fft_fft | Fft_ifft -> (sh, a.dtype)
        | Fft_rfft ->
            let nlast = fft_lengths.(k - 1) in
            let osh = Array.copy sh in
            for i = 0 to k - 2 do
              osh.(ndim - k + i) <- fft_lengths.(i)
            done;
            osh.(ndim - 1) <- (nlast / 2) + 1;
            (osh, complexify a.dtype)
        | Fft_irfft ->
            let osh = Array.copy sh in
            for i = 0 to k - 1 do
              osh.(ndim - k + i) <- fft_lengths.(i)
            done;
            (osh, complex_basetype a.dtype)
      in
      [ { shape; dtype; weak_type = false } ]
  | _ -> failwith "fft: fft_aval expects one aval"

let fft x fft_type fft_lengths =
  let av = Core.get_aval x in
  let target =
    match fft_type with
    | Fft_rfft -> to_inexact av.dtype
    | _ -> complexify (to_inexact av.dtype)
  in
  let x = Core.bind1 (Convert_element_type target) [ x ] in
  if Array.length fft_lengths = 0 then x
  else Core.bind1 (Fft { fft_type; fft_lengths }) [ x ]
