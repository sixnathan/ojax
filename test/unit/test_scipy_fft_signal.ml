module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module F = Ojax.Scipy.Fft
module S = Ojax.Scipy.Signal

let () = Ojax.Lax.install ()
let value shape data = T.Concrete (Nd.of_floats D.F64 shape data)

let concrete = function
  | T.Concrete nd -> nd
  | _ -> failwith "expected concrete"

let floats v =
  let out = ref [] in
  Nd.fold
    (fun () x ->
      out := x :: !out;
      ())
    () (concrete v);
  Array.of_list (List.rev !out)

let approx_array name tolerance expected actual =
  Alcotest.(check int)
    (name ^ " length") (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i want ->
      let got = actual.(i) in
      if abs_float (got -. want) > tolerance then
        Alcotest.failf "%s[%d] = %.17g, expected %.17g" name i got want)
    expected

let dct_roundtrip () =
  let x = value [| 5 |] [| 0.25; -1.5; 2.0; 0.75; -0.5 |] in
  approx_array "dct/idct" 1e-12 (floats x)
    (floats (F.idct ~norm:"ortho" (F.dct ~norm:"ortho" x)))

let dctn_roundtrip () =
  let x = value [| 2; 3 |] [| 1.0; -2.0; 0.5; 3.0; 0.25; -1.0 |] in
  let transformed = F.dctn ~axes:[| 1; 0 |] ~norm:"ortho" x in
  approx_array "dctn/idctn" 1e-12 (floats x)
    (floats (F.idctn ~axes:[| 1; 0 |] ~norm:"ortho" transformed))

let dct_type_error () =
  Alcotest.check_raises "type 3" (Failure "Only DCT type 2 is implemented.")
    (fun () -> ignore (F.dct ~type_:3 (value [| 2 |] [| 1.0; 2.0 |])))

let fftconvolve_modes () =
  let x = value [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let y = value [| 2 |] [| 1.0; 1.0 |] in
  approx_array "full" 1e-12 [| 1.0; 3.0; 5.0; 3.0 |]
    (floats (S.fftconvolve x y));
  approx_array "same" 1e-12 [| 1.0; 3.0; 5.0 |]
    (floats (S.fftconvolve ~mode:"same" x y));
  approx_array "valid" 1e-12 [| 3.0; 5.0 |]
    (floats (S.fftconvolve ~mode:"valid" x y))

let stft_roundtrip () =
  let x = value [| 8 |] [| 1.0; -0.5; 2.0; 0.25; -1.0; 0.75; 3.0; -2.0 |] in
  let _, _, z = S.stft ~nperseg:4 ~noverlap:2 x in
  let time, reconstructed = S.istft ~nperseg:4 ~noverlap:2 z in
  Alcotest.(check (array int)) "time shape" [| 8 |] (Nd.shape (concrete time));
  approx_array "stft/istft" 1e-12 (floats x) (floats reconstructed)

let welch_matches_csd () =
  let x = value [| 8 |] [| 0.5; -1.0; 2.0; 0.25; 1.5; -0.75; 0.0; 1.0 |] in
  let fw, pw = S.welch ~nperseg:4 ~noverlap:2 x in
  let fc, pc = S.csd ~nperseg:4 ~noverlap:2 x None in
  approx_array "frequencies" 1e-15 (floats fw) (floats fc);
  approx_array "periodogram" 1e-12 (floats pw) (floats pc)

let () =
  Alcotest.run "scipy_fft_signal"
    [
      ( "fft",
        [
          Alcotest.test_case "dct_roundtrip" `Quick dct_roundtrip;
          Alcotest.test_case "dctn_roundtrip" `Quick dctn_roundtrip;
          Alcotest.test_case "dct_type_error" `Quick dct_type_error;
        ] );
      ( "signal",
        [
          Alcotest.test_case "fftconvolve_modes" `Quick fftconvolve_modes;
          Alcotest.test_case "stft_roundtrip" `Quick stft_roundtrip;
          Alcotest.test_case "welch_matches_csd" `Quick welch_matches_csd;
        ] );
    ]
