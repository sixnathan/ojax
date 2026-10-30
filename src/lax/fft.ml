open Types

let two_pi = 8.0 *. Float.atan 1.0
let rec is_power_of_two n = n = 1 || (n mod 2 = 0 && is_power_of_two (n / 2))

let radix2 (a : Complex.t array) sign =
  let n = Array.length a in
  let j = ref 0 in
  for i = 0 to n - 2 do
    if i < !j then begin
      let tmp = a.(i) in
      a.(i) <- a.(!j);
      a.(!j) <- tmp
    end;
    let m = ref (n / 2) in
    while !m >= 1 && !j land !m <> 0 do
      j := !j lxor !m;
      m := !m / 2
    done;
    j := !j lor !m
  done;
  let len = ref 2 in
  while !len <= n do
    let half = !len / 2 in
    for start = 0 to (n / !len) - 1 do
      let base = start * !len in
      for k = 0 to half - 1 do
        let angle = sign *. two_pi *. float_of_int k /. float_of_int !len in
        let w = { Complex.re = cos angle; im = sin angle } in
        let u = a.(base + k) in
        let v = Complex.mul a.(base + k + half) w in
        a.(base + k) <- Complex.add u v;
        a.(base + k + half) <- Complex.sub u v
      done
    done;
    len := !len * 2
  done

let bluestein (a : Complex.t array) sign =
  let n = Array.length a in
  let m =
    let rec grow m = if m >= (2 * n) - 1 then m else grow (m * 2) in
    grow 1
  in
  let chirp =
    Array.init n (fun k ->
        let k2 = k * k mod (2 * n) in
        let angle = sign *. two_pi *. float_of_int k2 /. float_of_int (2 * n) in
        { Complex.re = cos angle; im = sin angle })
  in
  let x = Array.make m Complex.zero in
  let y = Array.make m Complex.zero in
  for k = 0 to n - 1 do
    x.(k) <- Complex.mul a.(k) chirp.(k);
    y.(k) <- Complex.conj chirp.(k);
    if k > 0 then y.(m - k) <- Complex.conj chirp.(k)
  done;
  radix2 x (-1.0);
  radix2 y (-1.0);
  let z = Array.map2 Complex.mul x y in
  radix2 z 1.0;
  let scale = { Complex.re = 1.0 /. float_of_int m; im = 0.0 } in
  for k = 0 to n - 1 do
    a.(k) <- Complex.mul (Complex.mul z.(k) scale) chirp.(k)
  done

let transform_line (a : Complex.t array) ~inverse =
  let n = Array.length a in
  let sign = if inverse then 1.0 else -1.0 in
  if n > 1 then if is_power_of_two n then radix2 a sign else bluestein a sign;
  if inverse then begin
    let scale = { Complex.re = 1.0 /. float_of_int n; im = 0.0 } in
    Array.iteri (fun i v -> a.(i) <- Complex.mul v scale) a
  end

let strides_of shape =
  let n = Array.length shape in
  let s = Array.make n 1 in
  for d = n - 2 downto 0 do
    s.(d) <- s.(d + 1) * shape.(d + 1)
  done;
  s

let total shape = Array.fold_left ( * ) 1 shape

let transform_axis (data : Complex.t array) shape axis ~inverse =
  let n = shape.(axis) in
  let stride = (strides_of shape).(axis) in
  let line = Array.make n Complex.zero in
  let block = stride * n in
  let count = Array.length data in
  if n > 0 && count > 0 then
    let outer = ref 0 in
    while !outer < count do
      for inner = 0 to stride - 1 do
        let base = !outer + inner in
        for k = 0 to n - 1 do
          line.(k) <- data.(base + (k * stride))
        done;
        transform_line line ~inverse;
        for k = 0 to n - 1 do
          data.(base + (k * stride)) <- line.(k)
        done
      done;
      outer := !outer + block
    done

let read_complex nd =
  let shape = Ndarray.shape nd in
  let n = total shape in
  let rank = Array.length shape in
  let strides = strides_of shape in
  let idx = Array.make rank 0 in
  let is_complex =
    match Ndarray.dtype nd with
    | Dtype.Complex64 | Dtype.Complex128 -> true
    | _ -> false
  in
  Array.init n (fun i ->
      for d = 0 to rank - 1 do
        idx.(d) <- i / strides.(d) mod shape.(d)
      done;
      if is_complex then Ndarray.get_c nd idx
      else { Complex.re = Ndarray.get_f nd idx; im = 0.0 })

let complex_of = function
  | Dtype.F32 | Dtype.Complex64 -> Dtype.Complex64
  | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
  | _ -> invalid_arg "fft requires a floating or complex operand"

let real_of = function
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | _ -> invalid_arg "irfft requires a complex operand"

let last_axes shape lengths =
  let rank = Array.length shape and k = Array.length lengths in
  List.init k (fun i -> rank - k + i)

let fft_impl fft_type lengths x =
  let shape = Ndarray.shape x in
  let dt = Ndarray.dtype x in
  match fft_type with
  | Fft_forward | Fft_inverse ->
      let data = read_complex x in
      let inverse = fft_type = Fft_inverse in
      List.iter
        (fun axis -> transform_axis data shape axis ~inverse)
        (last_axes shape lengths);
      Ndarray.of_complex (complex_of dt) shape data
  | Fft_real ->
      let data = read_complex x in
      List.iter
        (fun axis -> transform_axis data shape axis ~inverse:false)
        (last_axes shape lengths);
      let rank = Array.length shape in
      let n = shape.(rank - 1) in
      let kept = (n / 2) + 1 in
      let out_shape = Array.copy shape in
      out_shape.(rank - 1) <- kept;
      let rows = total shape / max n 1 in
      let out = Array.make (rows * kept) Complex.zero in
      for r = 0 to rows - 1 do
        for k = 0 to kept - 1 do
          out.((r * kept) + k) <- data.((r * n) + k)
        done
      done;
      Ndarray.of_complex (complex_of dt) out_shape out
  | Fft_inverse_real ->
      let data = read_complex x in
      let rank = Array.length shape in
      let k = Array.length lengths in
      let head = last_axes shape lengths in
      List.iteri
        (fun i axis ->
          if i < k - 1 then transform_axis data shape axis ~inverse:true)
        head;
      let kept = shape.(rank - 1) in
      let n = lengths.(k - 1) in
      let out_shape = Array.copy shape in
      out_shape.(rank - 1) <- n;
      let rows = total shape / max kept 1 in
      let out = Array.make (rows * n) 0.0 in
      let line = Array.make n Complex.zero in
      for r = 0 to rows - 1 do
        for j = 0 to n - 1 do
          line.(j) <-
            (if j < kept then data.((r * kept) + j)
             else Complex.conj data.((r * kept) + (n - j)))
        done;
        line.(0) <- { (line.(0)) with im = 0.0 };
        transform_line line ~inverse:true;
        for j = 0 to n - 1 do
          out.((r * n) + j) <- line.(j).Complex.re
        done
      done;
      Ndarray.of_floats (real_of dt) out_shape out

let fft_aval fft_type lengths (a : aval) =
  let rank = Array.length a.shape in
  match fft_type with
  | Fft_forward | Fft_inverse ->
      { a with dtype = complex_of a.dtype; weak_type = false }
  | Fft_real ->
      let shape = Array.copy a.shape in
      shape.(rank - 1) <- (shape.(rank - 1) / 2) + 1;
      { shape; dtype = complex_of a.dtype; weak_type = false }
  | Fft_inverse_real ->
      let shape = Array.copy a.shape in
      shape.(rank - 1) <- lengths.(Array.length lengths - 1);
      { shape; dtype = real_of a.dtype; weak_type = false }
