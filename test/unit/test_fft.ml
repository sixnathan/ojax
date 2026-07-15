module T = Ojax.Types
module C = Ojax.Core
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module LF = Ojax.Lax.Fft
module NF = Ojax.Numpy.Fft

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

let carray v =
  match v with
  | T.Concrete nd ->
      let sh = Nd.shape nd in
      let n = Array.fold_left ( * ) 1 sh in
      let dec i =
        let idx = Array.make (Array.length sh) 0 in
        let f = ref i in
        for d = Array.length sh - 1 downto 0 do
          idx.(d) <- !f mod sh.(d);
          f := !f / sh.(d)
        done;
        idx
      in
      Array.init n (fun i -> Nd.get_c nd (dec i))
  | _ -> failwith "expected concrete"

let farray v =
  match v with
  | T.Concrete nd ->
      let sh = Nd.shape nd in
      let n = Array.fold_left ( * ) 1 sh in
      let a = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            a.(i) <- x;
            i + 1)
          0 nd
      in
      a
  | _ -> failwith "expected concrete"

let approx_c name got want =
  Alcotest.(check int) (name ^ " len") (Array.length want) (Array.length got);
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      if
        Float.abs (g.Complex.re -. w.Complex.re) > 1e-4
        || Float.abs (g.Complex.im -. w.Complex.im) > 1e-4
      then
        Alcotest.failf "%s[%d] = %f%+fj want %f%+fj" name i g.Complex.re
          g.Complex.im w.Complex.re w.Complex.im)
    want

let approx_f name got want =
  Alcotest.(check int) (name ^ " len") (Array.length want) (Array.length got);
  Array.iteri
    (fun i w ->
      if Float.abs (got.(i) -. w) > 1e-4 then
        Alcotest.failf "%s[%d] = %f want %f" name i got.(i) w)
    want

let c re im = { Complex.re; im }

let test_fft_1d () =
  let x = f32 [| 4 |] [| 1.; 2.; 4.; 7. |] in
  let y = LF.fft x T.Fft_fft [| 4 |] in
  approx_c "lax fft" (carray y)
    [| c 14. 0.; c (-3.) 5.; c (-4.) 0.; c (-3.) (-5.) |];
  let y2 = NF.fft x in
  approx_c "numpy fft" (carray y2)
    [| c 14. 0.; c (-3.) 5.; c (-4.) 0.; c (-3.) (-5.) |]

let test_rfft () =
  let x = f32 [| 3 |] [| 1.; 3.; 5. |] in
  let y = NF.rfft x in
  approx_c "rfft" (carray y) [| c 9. 0.; c (-3.) 1.7320508 |]

let test_roundtrip () =
  let x = f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = NF.ifft (NF.fft x) in
  approx_c "ifft(fft)" (carray y)
    [| c 1. 0.; c 2. 0.; c 3. 0.; c 4. 0.; c 5. 0.; c 6. 0. |];
  let z = NF.irfft ~n:6 (NF.rfft x) in
  approx_f "irfft(rfft)" (farray z) [| 1.; 2.; 3.; 4.; 5.; 6. |]

let test_fft2 () =
  let x = f32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let y = NF.fft2 x in
  approx_c "fft2" (carray y) [| c 10. 0.; c (-4.) 0.; c (-2.) 0.; c 0. 0. |]

let test_fftfreq () =
  approx_f "fftfreq" (farray (NF.fftfreq 5)) [| 0.; 0.2; 0.4; -0.4; -0.2 |];
  approx_f "fftshift"
    (farray (NF.fftshift (NF.fftfreq 5)))
    [| -0.4; -0.2; 0.; 0.2; 0.4 |];
  approx_f "rfftfreq" (farray (NF.rfftfreq 5)) [| 0.; 0.2; 0.4 |]

let () =
  Alcotest.run "fft"
    [
      ( "fft",
        [
          Alcotest.test_case "fft_1d" `Quick test_fft_1d;
          Alcotest.test_case "rfft" `Quick test_rfft;
          Alcotest.test_case "roundtrip" `Quick test_roundtrip;
          Alcotest.test_case "fft2" `Quick test_fft2;
          Alcotest.test_case "freq" `Quick test_fftfreq;
        ] );
    ]
