open Ojax
module T = Types
module D = Dtype
module Nd = Ndarray
module IO = Numpy.Io
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module S = Scipy.Special
module St = Scipy.Stats
module N = Nn.Functions
module LL = Lax.Linalg
module Ad = Interpreters.Ad

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ()

let f16 v = T.Concrete (Nd.of_floats D.F16 [| Array.length v |] v)
let f32 v = T.Concrete (Nd.of_floats D.F32 [| Array.length v |] v)
let f64 v = T.Concrete (Nd.of_floats D.F64 [| Array.length v |] v)
let bools v = T.Concrete (Nd.of_floats D.Bool [| Array.length v |] v)

let c64 v =
  T.Concrete
    (Nd.of_complex D.Complex64
       [| Array.length v |]
       (Array.map (fun (re, im) -> { Complex.re; im }) v))

let host = function
  | T.Concrete n -> n
  | T.Device b -> Pjrt.Buffer.to_host b
  | T.Tracer _ -> Alcotest.fail "tracer"

let matches tol a b =
  (Float.is_nan a && Float.is_nan b)
  || a = b
  || Float.abs (a -. b) <= tol *. Float.max 1.0 (Float.abs b)

let reals name ?(tol = 1e-4) dt expected v =
  let n = host v in
  Alcotest.(check string)
    (name ^ " dtype") (D.short_name dt)
    (D.short_name (Nd.dtype n));
  Array.iteri
    (fun i e ->
      let got = Nd.get_f n [| i |] in
      if not (matches tol got e) then
        Alcotest.failf "%s[%d]: expected %.10g, got %.10g" name i e got)
    expected

let complexes name ?(tol = 1e-4) dt expected v =
  let n = host v in
  Alcotest.(check string)
    (name ^ " dtype") (D.short_name dt)
    (D.short_name (Nd.dtype n));
  Array.iteri
    (fun i (re, im) ->
      let z = Nd.get_c n [| i |] in
      if not (matches tol z.Complex.re re && matches tol z.Complex.im im) then
        Alcotest.failf "%s[%d]: expected %g%+gi, got %g%+gi" name i re im
          z.Complex.re z.Complex.im)
    expected

let raises name f =
  match f () with
  | _ -> Alcotest.failf "%s should raise" name
  | exception
      ( Failure _ | Invalid_argument _ | Py_exn.Type_error _
      | Py_exn.Value_error _ | Py_exn.Index_error _ | Py_exn.Overflow_error _
      | Py_exn.Not_implemented_error _ ) ->
      ()

let unhex s =
  String.init
    (String.length s / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub s (2 * i) 2)))

let temp_with suffix data =
  let path = Filename.temp_file "ojax_hardening" suffix in
  let oc = open_out_bin path in
  output_string oc data;
  close_out oc;
  path

let compressed_npz =
  "504b03042d0000000800000021004fe9002affffffffffffffff05001400612e6e707901001000980000000000000055000000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a06e9366a2aea3a09e965f54529498179f5f94920a12774bcc294e058a17672416a402f91a463a0ac69a3a0ab50a64032e063068b007120e4004c40d40bcc00100504b03042d0000000800000021008cd27e8dffffffffffffffff05001400622e6e70790100100088000000000000004c000000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a06e9366a4aea3a09e965f54529498179f5f94920a12774bcc294e058a17672416a402f91a263a9a3a0ab50a14002e063b86030c35ff9901504b03042d00000008000000210098e3a8b6ffffffffffffffff05001400632e6e707901001000300500000000000063000000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a06e9369a2aea3a09e965f54529498179f5f94920a12774bcc294e058a17672416a402f91ac606063a9a3a0ab50a64032e06206004622620660662162066056236060818951b951b951b951b2c7200504b01022d032d0000000800000021004fe9002a5500000098000000050000000000000000000000800100000000612e6e7079504b01022d032d0000000800000021008cd27e8d4c0000008800000005000000000000000000000080018c000000622e6e7079504b01022d032d00000008000000210098e3a8b6630000003005000005000000000000000000000080010f010000632e6e7079504b0506000000000300030099000000a90100000000"

let half_npy =
  "934e554d5059010076007b276465736372273a20273c6632272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028352c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a003e00c0ff03ff7b007c"

let bfloat_npy =
  "934e554d5059010076007b276465736372273a20273c5632272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200ac03f00c09643"

let big_endian_npy =
  "934e554d5059010076007b276465736372273a20273e6638272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a3ff0000000000000c004000000000000"

let big_endian_complex_npy =
  "934e554d5059010076007b276465736372273a20273e6338272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028312c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a3f80000040000000"

let npz_compressed () =
  let members = IO.load_npz (temp_with ".npz" (unhex compressed_npz)) in
  Alcotest.(check (list string))
    "names" [ "a"; "b"; "c" ] (List.map fst members);
  let a = host (List.assoc "a" members) in
  Alcotest.(check (array int)) "a shape" [| 2; 3 |] (Nd.shape a);
  Alcotest.(check (float 0.0)) "a cell" 5.0 (Nd.get_f a [| 1; 2 |]);
  reals "b" ~tol:0.0 D.F16
    [| 1.5; -2.0; Float.infinity; 6.097555160522461e-05 |]
    (List.assoc "b" members);
  let c = host (List.assoc "c" members) in
  Alcotest.(check (array int)) "c shape" [| 300 |] (Nd.shape c);
  Alcotest.(check (float 0.0))
    "c cell"
    (float_of_int (299 mod 7))
    (Nd.get_f c [| 299 |])

let npz_bad_checksum () =
  let data = Bytes.of_string (unhex compressed_npz) in
  let crc_at =
    let s = Bytes.to_string data in
    let rec find i =
      if String.get_int32_le s i = 0x02014b50l then i + 16 else find (i + 1)
    in
    find 0
  in
  Bytes.set data crc_at (Char.chr (Char.code (Bytes.get data crc_at) lxor 1));
  raises "corrupt member" (fun () ->
      IO.load_npz (temp_with ".npz" (Bytes.to_string data)))

let npy_narrow () =
  let v = IO.load (temp_with ".npy" (unhex half_npy)) in
  reals "f16 read" ~tol:0.0 D.F16
    [| 1.5; -2.0; 6.097555160522461e-05; 65504.0; Float.infinity |]
    v;
  let out = Filename.temp_file "ojax_hardening" ".npy" in
  IO.save out v;
  let ic = open_in_bin out in
  let written = really_input_string ic (in_channel_length ic) in
  close_in ic;
  Alcotest.(check string) "f16 bytes" (unhex half_npy) written;
  let b = IO.load (temp_with ".npy" (unhex bfloat_npy)) in
  reals "bfloat read" ~tol:0.0 D.Bf16 [| 1.5; -2.0; 300.0 |] b;
  let out = Filename.temp_file "ojax_hardening" ".npy" in
  IO.save out b;
  let ic = open_in_bin out in
  let written = really_input_string ic (in_channel_length ic) in
  close_in ic;
  Alcotest.(check string) "bfloat bytes" (unhex bfloat_npy) written

let npy_big_endian_and_canonical () =
  let path = temp_with ".npy" (unhex big_endian_npy) in
  reals "x64 off narrows" D.F32 [| 1.0; -2.5 |] (IO.load path);
  Config.with_value Config.enable_x64 true (fun () ->
      reals "x64 on keeps" D.F64 [| 1.0; -2.5 |] (IO.load path));
  complexes "big endian complex" D.Complex64
    [| (1.0, 2.0) |]
    (IO.load (temp_with ".npy" (unhex big_endian_complex_npy)))

let no_execution () =
  let x = f32 [| 1.0; 2.0 |] in
  let blocked f =
    Config.with_value Config.no_execution true (fun () -> raises "blocked" f)
  in
  blocked (fun () -> UF.add x x);
  blocked (fun () -> Lax_api.add x x);
  blocked (fun () -> UF.sin x);
  blocked (fun () ->
      Ad.jvp (fun a -> [ UF.multiply (List.hd a) (List.hd a) ]) [ x ] [ x ]);
  let shape_only =
    Config.with_value Config.no_execution true (fun () ->
        Jaxpr.make_jaxpr
          [ Core.get_aval x ]
          (fun a -> [ UF.add (List.hd a) (List.hd a) ]))
  in
  ignore shape_only;
  reals "after the block" ~tol:0.0 D.F32 [| 2.0; 4.0 |] (UF.add x x)

let eig_derivative_flag () =
  let a = T.Concrete (Nd.of_floats D.F32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]) in
  let da = T.Concrete (Nd.of_floats D.F32 [| 2; 2 |] [| 1.; 0.; 0.; 1. |]) in
  let vectors flag args =
    [
      List.nth
        (LL.eig ~compute_left_eigenvectors:false ?enable_eigvec_derivs:flag
           (List.hd args))
        1;
    ]
  in
  raises "eigenvector jvp needs the flag" (fun () ->
      Ad.jvp (vectors None) [ a ] [ da ]);
  raises "numpy eig has no flag" (fun () ->
      Ad.jvp
        (fun args ->
          let _, v = Numpy.Linalg.eig (List.hd args) in
          [ v ])
        [ a ] [ da ]);
  let _, tangent = Ad.jvp (vectors (Some true)) [ a ] [ da ] in
  let t = host (List.hd tangent) in
  for i = 0 to 1 do
    for j = 0 to 1 do
      Alcotest.(check (float 1e-5))
        "shifting by the identity leaves the vectors alone" 0.0
        (Nd.get_c t [| i; j |]).Complex.re
    done
  done;
  let values args =
    [
      List.hd
        (LL.eig ~compute_left_eigenvectors:false
           ~compute_right_eigenvectors:false (List.hd args));
    ]
  in
  let _, tangent = Ad.jvp values [ a ] [ da ] in
  complexes "eigenvalues need no flag" D.Complex64
    [| (1.0, 0.0); (0.99999994, 0.0) |]
    (List.hd tangent)

let zs = [| (0.5, 0.25); (-1.5, 0.75); (-3.5, 0.1); (2.0, 3.0) |]
let zr = [| (2.0, 3.0); (-3.5, 0.1); (-1.5, 0.75); (0.5, 0.25) |]

let complex_special () =
  let z = c64 zs in
  complexes "gamma" D.Complex64
    [|
      (1.3851134, -0.6731814);
      (0.43501374, 0.26276758);
      (0.2551255, 0.035665873);
      (-0.08239532, 0.09177426);
    |]
    (S.gamma z);
  complexes "loggamma" D.Complex64
    [|
      (0.4318061, -0.45239455);
      (-0.6768471, -5.7397976);
      (-1.3563206, -12.427472);
      (-2.0928516, 2.302397);
    |]
    (S.loggamma z);
  complexes "erfcx" ~tol:1e-3 D.Complex64
    [|
      (5.9370887e-01, -1.2476491e-01);
      (-7.0744176, -8.5224094);
      (3.1649438e+05, -2.6657966e+05);
      (9.2710771e-02, -1.2831697e-01);
    |]
    (S.erfcx z);
  complexes "owens_t" ~tol:1e-5 D.Complex64
    [|
      (7.94945285e-02, 1.1924180e-01);
      (-2.33735889e-02, 6.6781684e-04);
      (-1.03478946e-04, 5.1739473e-05);
      (6.73797040e-05, 3.3689852e-05);
    |]
    (S.owens_t z (c64 zr));
  raises "erf rejects complex" (fun () -> S.erf z);
  raises "gammaln rejects complex" (fun () -> S.gammaln z);
  raises "dawsn rejects complex" (fun () -> S.dawsn z)

let complex_nn () =
  let z = c64 zs in
  complexes "softplus" D.Complex64
    [|
      (0.96671736, 1.5576553e-01);
      (0.15970328, 1.3001060e-01);
      (0.029608252, 2.9268265e-03);
      (1.8563948, 2.9779503);
    |]
    (N.softplus z);
  complexes "logaddexp" D.Complex64
    [|
      (1.7747486, 2.893122);
      (-1.3949537, 0.67619705);
      (-1.3949537, 0.67619705);
      (1.7747486, 2.893122);
    |]
    (UF.logaddexp z (c64 zr));
  complexes "logaddexp2" D.Complex64
    [|
      (1.9177889, 2.4788399);
      (-1.2014825, 0.6221309);
      (-1.2014825, 0.6221309);
      (1.9177889, 2.4788399);
    |]
    (UF.logaddexp2 z (c64 zr));
  complexes "mish" D.Complex64
    [|
      (0.36054114, 0.22302473);
      (-0.33705705, -0.0703739);
      (-0.10389192, -0.007274961);
      (1.9543798, 2.8341224);
    |]
    (N.mish z);
  complexes "log_sigmoid" D.Complex64
    [|
      (-0.4667174, 0.09423447);
      (-1.6597033, 0.61998963);
      (-3.5296082, 0.09707332);
      (0.14360522, 0.022049665);
    |]
    (N.log_sigmoid z);
  raises "sparse_sigmoid clips" (fun () -> N.sparse_sigmoid z);
  raises "standardize clips" (fun () -> N.standardize z)

let complex_stats () =
  let z = c64 zs in
  complexes "logistic.logpdf" D.Complex64
    [|
      (-1.4334347, -0.061531067);
      (-1.8194065, 0.4899788);
      (-3.5592165, 0.09414625);
      (-1.7127895, -2.9559007);
    |]
    (St.Logistic.logpdf z);
  complexes "norm.pdf" D.Complex64
    [|
      (3.6040697e-01, -4.5286983e-02);
      (7.3982529e-02, 1.5481372e-01);
      (8.2388346e-04, 3.0074094e-04);
      (4.6665354, 1.3579906);
    |]
    (St.Norm.pdf z);
  complexes "cauchy.cdf" D.Complex64
    [|
      (0.654143, 0.063848704);
      (0.16350284, 0.06618723);
      (0.088522136, 0.002400709);
      (0.9487918, 0.0729161);
    |]
    (St.Cauchy.cdf z);
  complexes "expon.ppf" D.Complex64
    [|
      (0.5815754, 0.4636476);
      (Float.nan, 0.0);
      (Float.nan, 0.0);
      (Float.nan, 0.0);
    |]
    (St.Expon.ppf z);
  complexes "uniform.ppf" D.Complex64
    [| (0.5, 0.25); (Float.nan, 0.0); (Float.nan, 0.0); (Float.nan, 0.0) |]
    (St.Uniform.ppf z);
  complexes "pareto.ppf" D.Complex64
    [|
      (3.5349312, 1.0033926);
      (Float.nan, 0.0);
      (Float.nan, 0.0);
      (Float.nan, 0.0);
    |]
    (St.Pareto.ppf z z);
  raises "norm.cdf rejects complex" (fun () -> St.Norm.cdf z);
  raises "expon.pdf rejects complex" (fun () -> St.Expon.pdf z);
  raises "gamma.logpdf rejects complex" (fun () -> St.Gamma.logpdf z z)

let narrow_stats () =
  let h = f16 [| 0.5; 1.5; 3.0 |] in
  let tol = 1e-3 in
  reals "norm.logpdf" ~tol D.F16 [| -1.044; -2.043; -5.418 |] (St.Norm.logpdf h);
  reals "gamma.logpdf" ~tol D.F16
    [| -0.7256; -1.176; -1.496 |]
    (St.Gamma.logpdf h h);
  reals "cauchy.cdf" ~tol D.F16 [| 0.6475; 0.813; 0.8975 |] (St.Cauchy.cdf h);
  reals "uniform.pdf" ~tol D.F16 [| 1.0; 0.0; 0.0 |] (St.Uniform.pdf h);
  reals "t.logpdf" ~tol D.F16 [| -1.615; -2.223; -3.773 |] (St.T.logpdf h h);
  reals "expon.logpdf" ~tol D.F16 [| -0.5; -1.5; -3.0 |] (St.Expon.logpdf h);
  raises "norm.cdf rejects f16" (fun () -> St.Norm.cdf h);
  raises "truncnorm rejects f16" (fun () -> St.Truncnorm.pdf h h h);
  let b = bools [| 1.0; 0.0; 1.0 |] in
  reals "bool norm.pdf" D.F32
    [| 0.24197073; 0.3989423; 0.24197073 |]
    (St.Norm.pdf b);
  reals "bool expon.cdf" D.F32
    [| 0.63212055; 0.0; 0.63212055 |]
    (St.Expon.cdf b);
  Config.with_value Config.enable_x64 true (fun () ->
      reals "bool under x64 uses the default float" D.F64
        [| 0.24197072451914337; 0.39894228040143265; 0.24197072451914337 |]
        (St.Norm.pdf b))

let scalar_bugs () =
  reals "erfinv at the ends" D.F32
    [| Float.infinity; Float.neg_infinity; 0.47693628; Float.nan |]
    (S.erfinv (f32 [| 1.0; -1.0; 0.5; 2.0 |]));
  reals "gammainc at a=0" D.F32
    [| 1.0; Float.nan; 0.6321202 |]
    (S.gammainc (f32 [| 0.0; 0.0; 1.0 |]) (f32 [| 1.0; 0.0; 1.0 |]));
  reals "gammaincc at a=0" D.F32
    [| 0.0; Float.nan; 0.36787927 |]
    (S.gammaincc (f32 [| 0.0; 0.0; 1.0 |]) (f32 [| 1.0; 0.0; 1.0 |]));
  let b = bools [| 1.0; 0.0; 1.0 |] in
  reals "hard_swish" D.F32 [| 0.6666667; 0.0; 0.6666667 |] (N.hard_swish b);
  reals "tanh" D.F32 [| 0.7615942; 0.0; 0.7615942 |] (N.tanh b);
  let s, c = S.fresnel b in
  reals "fresnel s" D.F32 [| 0.43825915; 0.0; 0.43825915 |] s;
  reals "fresnel c" D.F32 [| 0.7798934; 0.0; 0.7798934 |] c;
  reals "owens_t f32" D.F32 [| 0.064488605 |]
    (S.owens_t (f32 [| 0.5 |]) (f32 [| 0.5 |]));
  reals "owens_t f16 widens" D.F32 [| 0.064488605 |]
    (S.owens_t (f16 [| 0.5 |]) (f16 [| 0.5 |]));
  raises "ndtr rejects integers" (fun () ->
      S.ndtr (T.Concrete (Nd.of_floats D.I32 [| 1 |] [| 1.0 |])));
  raises "sigmoid rejects bool" (fun () -> N.sigmoid b);
  raises "softmax rejects bool" (fun () -> S.softmax b)

let max_min_ties () =
  let grad op x =
    host
      (List.hd
         (snd
            (Ad.jvp
               (fun a -> [ op (List.hd a) (f32 [| 1.0 |]) ])
               [ f32 [| x |] ]
               [ f32 [| 1.0 |] ])))
  in
  let get op x = Nd.get_f (grad op x) [| 0 |] in
  Alcotest.(check (float 0.0)) "max tie" 0.5 (get UF.maximum 1.0);
  Alcotest.(check (float 0.0)) "min tie" 0.5 (get UF.minimum 1.0);
  Alcotest.(check (float 0.0)) "max above" 1.0 (get UF.maximum 2.0);
  Alcotest.(check (float 0.0)) "max below" 0.0 (get UF.maximum 0.5)

let primitive_dtypes () =
  let z = c64 [| (1.0, 1.0) |] in
  let i = T.Concrete (Nd.of_floats D.I32 [| 1 |] [| 1.0 |]) in
  let b = bools [| 1.0 |] in
  raises "lax.lt complex" (fun () -> Lax_api.lt z z);
  raises "lax.floor complex" (fun () -> Lax_api.floor z);
  raises "lax.exp int" (fun () -> Lax_api.exp i);
  raises "lax.neg bool" (fun () -> Lax_api.neg b);
  reals "jnp.less keeps its complex order" D.Bool [| 1.0 |]
    (UF.less z (c64 [| (1.0, 2.0) |]))

let scalar dt v = T.Concrete (Nd.of_floats dt [||] [| v |])

let one name dt expected v =
  let n = host v in
  Alcotest.(check string)
    (name ^ " dtype") (D.short_name dt)
    (D.short_name (Nd.dtype n));
  Alcotest.(check (float 1e-3)) name expected (Nd.get_f n [||])

let integer_primitives () =
  let i32 = scalar D.I32 and u8 = scalar D.Uint8 and i8 = scalar D.I8 in
  one "shift out" D.I32 0.0 (UF.left_shift (i32 (-2.0)) (i32 100.0));
  one "sra unsigned" D.Uint8 255.0
    (Lax_api.shift_right_arithmetic (u8 128.0) (u8 9.0));
  one "jnp shift unsigned" D.Uint8 100.0 (UF.right_shift (u8 200.0) (u8 1.0));
  one "srl signed" D.I8 64.0
    (Lax_api.shift_right_logical (i8 (-128.0)) (i8 1.0));
  one "shl wraps" D.I32 (-2147483648.0) (UF.left_shift (i32 3.0) (i32 31.0));
  one "i32 mul wraps exactly" D.I32 (-67153019.0)
    (UF.multiply (i32 123456789.0) (i32 987654321.0));
  one "u16 mul" D.Uint16 1.0
    (UF.multiply (scalar D.Uint16 65535.0) (scalar D.Uint16 65535.0));
  one "i8 mul" D.I8 16.0 (UF.multiply (i8 100.0) (i8 100.0));
  one "int power" D.I32 532643583.0 (UF.power (i32 255.0) (i32 127.0));
  one "unsigned floor_divide by zero" D.Uint8 255.0
    (UF.floor_divide (u8 3.0) (u8 0.0));
  one "signed floor_divide by zero" D.I8 (-2.0)
    (UF.floor_divide (i8 3.0) (i8 0.0));
  one "clz i8" D.I8 0.0 (Lax_api.clz (i8 (-1.0)));
  one "clz u8" D.Uint8 7.0 (Lax_api.clz (u8 1.0));
  one "clz i16" D.I16 15.0 (Lax_api.clz (scalar D.I16 1.0));
  one "popcount i8" D.I8 8.0 (Lax_api.population_count (i8 (-1.0)));
  one "popcount u16" D.Uint16 2.0
    (Lax_api.population_count (scalar D.Uint16 257.0));
  Alcotest.(check string)
    "floor keeps ints" "i8"
    (D.short_name (Nd.dtype (host (UF.floor (i8 3.0)))));
  Alcotest.(check string)
    "ceil keeps ints" "u16"
    (D.short_name (Nd.dtype (host (UF.ceil (scalar D.Uint16 3.0)))));
  Alcotest.(check string)
    "x64 off narrows u32 with i32" "i32"
    (D.short_name
       (Nd.dtype (host (UF.add (scalar D.Uint32 1.0) (scalar D.I32 1.0)))));
  raises "lax.abs unsigned" (fun () -> Lax_api.abs (u8 1.0));
  one "jnp.abs unsigned" D.Uint8 1.0 (UF.abs (u8 1.0));
  raises "lax.add bool" (fun () ->
      Lax_api.add (scalar D.Bool 1.0) (scalar D.Bool 1.0));
  raises "lax.real of a float" (fun () -> Lax_api.real (scalar D.F32 1.0));
  raises "lax.round of an int" (fun () -> Lax_api.round (i32 1.0));
  raises "lax.reciprocal of an int" (fun () -> Lax_api.reciprocal (i32 1.0))

let conversions () =
  let z = c64 [| (0.0, 1.0); (0.0, 0.0); (-1.5, 0.0); (300.0, 1.0) |] in
  reals "complex to bool" ~tol:0.0 D.Bool [| 1.0; 0.0; 1.0; 1.0 |]
    (NL.astype z D.Bool);
  reals "complex to u8 saturates" ~tol:0.0 D.Uint8 [| 0.0; 255.0 |]
    (NL.astype (c64 [| (-1.5, 1.0); (300.0, 1.0) |]) D.Uint8);
  reals "complex to i8 saturates" ~tol:0.0 D.I8 [| -128.0; 127.0 |]
    (NL.astype (c64 [| (-200.0, 0.0); (200.0, 1.0) |]) D.I8);
  Config.with_value Config.enable_x64 true (fun () ->
      let wide =
        T.Concrete
          (Nd.of_int64s D.I64 [| 3 |]
             [| 1099511627776L; 2147483653L; -8589934593L |])
      in
      reals "i64 to i32 wraps" ~tol:0.0 D.I32
        [| 0.0; -2147483643.0; -1.0 |]
        (NL.astype wide D.I32);
      reals "i64 to i8 wraps" ~tol:0.0 D.I8 [| 0.0; 5.0; -1.0 |]
        (NL.astype wide D.I8);
      reals "i64 to u32 wraps" ~tol:0.0 D.Uint32
        [| 0.0; 2147483653.0; 4294967295.0 |]
        (NL.astype wide D.Uint32);
      let get_i64 v i = Nd.get_i64 (host v) [| i |] in
      let f = f64 [| 1e20; -1e20; Float.infinity; Float.nan |] in
      let as_i64 = NL.astype f D.I64 in
      Alcotest.(check int64) "saturates up" Int64.max_int (get_i64 as_i64 0);
      Alcotest.(check int64) "saturates down" Int64.min_int (get_i64 as_i64 1);
      Alcotest.(check int64) "inf" Int64.max_int (get_i64 as_i64 2);
      Alcotest.(check int64) "nan" 0L (get_i64 as_i64 3);
      let as_u64 = NL.astype f D.Uint64 in
      Alcotest.(check int64) "u64 top" (-1L) (get_i64 as_u64 0);
      Alcotest.(check int64) "u64 floor" 0L (get_i64 as_u64 1);
      let top = T.Concrete (Nd.of_int64s D.Uint64 [||] [| Int64.min_int |]) in
      let shifted v =
        Nd.get_i64
          (host
             (Lax_api.shift_right_arithmetic top
                (T.Concrete (Nd.of_int64s D.Uint64 [||] [| v |]))))
          [||]
      in
      Alcotest.(check int64) "u64 sra out of range" (-1L) (shifted 70L);
      Alcotest.(check int64) "u64 sra" (-4611686018427387904L) (shifted 1L))

let narrow_floats () =
  let f = scalar D.F16 and tol = 1e-4 in
  one "nextafter up" D.F16 1.0009765625 (UF.nextafter (f 1.0) (f 2.0));
  one "nextafter down" D.F16 0.99951171875 (UF.nextafter (f 1.0) (f 0.0));
  Alcotest.(check (float 1e-9))
    "nextafter from zero" 5.960464477539063e-08
    (Nd.get_f (host (UF.nextafter (f 0.0) (f 1.0))) [||]);
  one "bfloat nextafter" D.Bf16 1.0078125
    (UF.nextafter (scalar D.Bf16 1.0) (scalar D.Bf16 2.0));
  one "spacing" D.F16 0.0009765625 (UF.spacing (f 1.0));
  one "spacing of -inf" D.F16 Float.infinity (UF.spacing (f Float.neg_infinity));
  let h = f16 [| 1.5; Float.neg_infinity; Float.infinity; -0.0; Float.nan |] in
  reals "isfinite" ~tol:0.0 D.Bool [| 1.; 0.; 0.; 1.; 0. |] (UF.isfinite h);
  reals "isinf" ~tol:0.0 D.Bool [| 0.; 1.; 1.; 0.; 0. |] (UF.isinf h);
  reals "isposinf" ~tol:0.0 D.Bool [| 0.; 0.; 1.; 0.; 0. |] (UF.isposinf h);
  reals "isneginf" ~tol:0.0 D.Bool [| 0.; 1.; 0.; 0.; 0. |] (UF.isneginf h);
  reals "signbit" ~tol:0.0 D.Bool [| 0.; 1.; 0.; 1.; 0. |] (UF.signbit h);
  one "copysign" D.F16 (-2.0) (UF.copysign (f 2.0) (f (-0.0)));
  reals "hypot" ~tol D.F16
    [| 2.121; Float.infinity; Float.infinity; 0.0; Float.nan |]
    (UF.hypot h h)

let rounding () =
  let x = f32 [| -0.5; 0.5; 1.5; 2.5; -1.5 |] in
  reals "rint" ~tol:0.0 D.F32 [| -0.0; 0.0; 2.0; 2.0; -2.0 |] (UF.rint x);
  reals "round" ~tol:0.0 D.F32 [| -0.0; 0.0; 2.0; 2.0; -2.0 |] (NL.round x);
  reals "lax.round even" ~tol:0.0 D.F32
    [| -0.0; 0.0; 2.0; 2.0; -2.0 |]
    (Lax_api.round ~rounding_method:Lax_api.To_nearest_even x);
  reals "lax.round default" ~tol:0.0 D.F32
    [| -1.0; 1.0; 2.0; 3.0; -2.0 |]
    (Lax_api.round x);
  reals "rint under jit" ~tol:0.0 D.F32
    [| -0.0; 0.0; 2.0; 2.0; -2.0 |]
    (List.hd (Api.jit_flat (fun a -> [ UF.rint (List.hd a) ]) [ x ]));
  complexes "rint complex" ~tol:0.0 D.Complex64
    [| (2.0, 2.0) |]
    (UF.rint (c64 [| (1.5, 2.5) |]));
  Alcotest.(check string)
    "rint of ints" "f32"
    (D.short_name
       (Nd.dtype
          (host (UF.rint (T.Concrete (Nd.of_floats D.I8 [| 1 |] [| 1.0 |]))))))

let complex_primitives () =
  let inf = Float.infinity and nan = Float.nan in
  complexes "divide by zero" D.Complex64
    [| (inf, nan); (inf, nan); (nan, nan); (-.inf, nan) |]
    (UF.divide
       (c64 [| (0.5, 0.0); (1.0, 0.0); (0.0, 0.0); (-1.0, 0.0) |])
       (c64 [| (0.0, 0.0); (0.0, 0.0); (0.0, 0.0); (0.0, 0.0) |]));
  complexes "reciprocal" D.Complex64
    [| (inf, nan); (0.0, -1.0) |]
    (UF.reciprocal (c64 [| (0.0, 0.0); (0.0, 1.0) |]));
  complexes "log2" D.Complex64
    [| (Float.neg_infinity, 0.0); (0.0, 2.26618) |]
    (UF.log2 (c64 [| (0.0, 0.0); (0.0, 1.0) |]));
  complexes "log10" D.Complex64
    [| (Float.neg_infinity, 0.0); (0.0, 0.6821882) |]
    (UF.log10 (c64 [| (0.0, 0.0); (0.0, 1.0) |]));
  complexes "exp2" D.Complex64
    [| (0.33253184, 0.12009399) |]
    (UF.exp2 (c64 [| (-1.5, 0.5) |]));
  complexes "atan2" D.Complex64
    [| (-1.80262, -0.40235952) |]
    (Lax_api.atan2 (c64 [| (-1.5, 0.5) |]) (c64 [| (-0.5, -0.5) |]));
  complexes "atan pole" D.Complex64
    [| (0.0, inf); (0.0, Float.neg_infinity) |]
    (UF.arctan (c64 [| (0.0, 1.0); (0.0, -1.0) |]));
  complexes "atanh pole" D.Complex64
    [| (inf, 0.0); (Float.neg_infinity, 0.0) |]
    (UF.arctanh (c64 [| (1.0, 0.0); (-1.0, 0.0) |]));
  let c = c64 [| (inf, 1.0); (1.0, nan); (1.0, 1.0) |] in
  reals "complex isfinite" ~tol:0.0 D.Bool [| 0.; 0.; 1. |] (UF.isfinite c);
  reals "complex isinf" ~tol:0.0 D.Bool [| 1.; 0.; 0. |] (UF.isinf c);
  raises "hypot complex" (fun () -> UF.hypot c c);
  raises "fabs complex" (fun () -> UF.fabs c);
  raises "isposinf complex" (fun () -> UF.isposinf c)

let special_edges () =
  let f = f32 in
  let inf = Float.infinity and nan = Float.nan in
  reals "igamma" ~tol:1e-5 D.F32 [| nan; 1.0; 1.0; 1.0 |]
    (S.gammainc (f [| 0.0; 0.0; 0.5; 10.0 |]) (f [| 0.5; 1.0; 1e4; 1e4 |]));
  reals "zeta domain" ~tol:1e-5 D.F32
    [| inf; inf; inf; nan; nan; nan; nan; 1.644934 |]
    (S.zeta
       ~q:(f [| nan; 0.0; -1.0; -1.0; 0.0; inf; 2.0; 1.0 |])
       (f [| 1.0; 0.0; 2.0; 3.0; 1.5; 1.5; -1.0; 2.0 |]));
  reals "polygamma at inf" ~tol:0.0 D.F32 [| nan |]
    (S.polygamma
       (T.Concrete (Nd.of_floats D.I32 [| 1 |] [| 1.0 |]))
       (f [| inf |]));
  reals "expn float16" ~tol:1e-3 D.F16 [| 0.05704 |]
    (S.expn (f16 [| 0.5 |]) (f16 [| 2.0 |]))

let compiled_paths () =
  let jit1 g x = List.hd (Api.jit_flat (fun a -> [ g (List.hd a) ]) [ x ]) in
  let nan = Float.nan in
  reals "jit max keeps nan" ~tol:0.0 D.F32 [| nan |]
    (List.hd
       (Api.jit_flat
          (fun a -> [ Numpy.Reductions.max (List.hd a) ])
          [ f32 [| 1.0; nan |] ]));
  reals "jit minimum keeps nan" ~tol:0.0 D.F32 [| 0.0; nan |]
    (jit1 (fun x -> UF.minimum x (f32 [| 0.0; 0.0 |])) (f32 [| 1.0; nan |]));
  let c = c64 [| (1.0, 1.0); (2.0, -1.0); (0.0, 0.5); (0.0, 0.0) |] in
  complexes "jit complex minimum with a constant" D.Complex64
    [| (1.0, 1.0); (2.0, -1.0); (0.0, 0.5); (0.0, 0.0) |]
    (jit1
       (fun z ->
         UF.minimum z (T.Concrete (Nd.of_floats D.Complex64 [||] [| 6.0 |])))
       c);
  complexes "jit complex maximum" D.Complex64
    [| (1.0, 1.0); (2.0, -1.0); (1.0, 0.0); (1.0, 0.0) |]
    (jit1
       (fun z ->
         UF.maximum z (T.Concrete (Nd.of_floats D.Complex64 [||] [| 1.0 |])))
       c);
  reals "jit all reads real parts" ~tol:0.0 D.Bool [| 0.0 |]
    (jit1 (fun z -> Numpy.Reductions.all z) c);
  reals "jit sort float16" ~tol:0.0 D.F16 [| 1.0; 2.0; nan |]
    (jit1 (fun x -> Numpy.Sorting.sort x) (f16 [| 2.0; nan; 1.0 |]));
  Config.with_value Config.enable_x64 true (fun () ->
      let u = T.Concrete (Nd.of_int64s D.Uint64 [| 3 |] [| 3L; 1L; 5L |]) in
      Alcotest.(check int64)
        "jit u64 min" 1L
        (Nd.get_i64 (host (jit1 (fun x -> Numpy.Reductions.min x) u)) [||]);
      Alcotest.(check int64)
        "jit u64 ptp" 4L
        (Nd.get_i64 (host (jit1 (fun x -> Numpy.Reductions.ptp x) u)) [||]))

let numpy_bugs () =
  let inf = Float.infinity and nan = Float.nan in
  reals "sort puts nan last" ~tol:0.0 D.F32
    [| Float.neg_infinity; 1.0; 2.0; nan |]
    (Numpy.Sorting.sort (f32 [| 2.0; nan; 1.0; Float.neg_infinity |]));
  complexes "sort_complex puts nan last" ~tol:0.0 D.Complex64
    [| (-1.0, 0.0); (1.0, 0.0); (nan, 0.0) |]
    (Numpy.Extras.sort_complex (f32 [| nan; 1.0; -1.0 |]));
  let c = c64 [| (1.0, 1.0); (2.0, -1.0); (0.0, 0.5); (0.0, 0.0) |] in
  (match Numpy.Extras.nonzero c with
  | [ i ] ->
      reals "nonzero sees imaginary parts" ~tol:0.0 D.I32 [| 0.0; 1.0; 2.0 |] i
  | _ -> Alcotest.fail "nonzero arity");
  reals "all on complex" ~tol:0.0 D.Bool [| 0.0 |] (Numpy.Reductions.all c);
  reals "any on complex" ~tol:0.0 D.Bool [| 1.0 |] (Numpy.Reductions.any c);
  (match
     Numpy.Extras.unique
       (c64 [| (1.0, 1.0); (1.0, -1.0); (1.0, 1.0); (0.0, 0.0) |])
   with
  | [ u ] ->
      complexes "unique on complex" ~tol:0.0 D.Complex64
        [| (0.0, 0.0); (1.0, -1.0); (1.0, 1.0) |]
        u
  | _ -> Alcotest.fail "unique arity");
  reals "clip widens bool" ~tol:0.0 D.I32 [| 1.0; 0.0 |]
    (NL.clip ~min:0.0 ~max:2.0 (bools [| 1.0; 0.0 |]));
  reals "nan_to_num float16" ~tol:0.0 D.F16 [| 65504.0; -65504.0 |]
    (NL.nan_to_num (f16 [| inf; Float.neg_infinity |]));
  reals "isclose on complex" ~tol:0.0 D.Bool [| 1.0; 1.0; 1.0; 1.0 |]
    (NL.isclose c c);
  raises "argmax complex" (fun () -> NL.argmax c);
  raises "round bool" (fun () -> NL.round (bools [| 1.0 |]));
  raises "cholesky float16" (fun () ->
      Numpy.Linalg.cholesky
        (T.Concrete (Nd.of_floats D.F16 [| 1; 1 |] [| 1.0 |])))

let jit_conj_of_real () =
  complexes "jit conj of a float" ~tol:0.0 D.Complex64
    [| (1.5, 0.0); (-2.0, 0.0) |]
    (List.hd
       (Api.jit_flat
          (fun a -> [ Lax_api.conj (List.hd a) ])
          [ f32 [| 1.5; -2.0 |] ]))

let closed_form_det () =
  let m sh v = T.Concrete (Nd.of_floats D.F32 sh v) in
  Alcotest.(check (float 0.0))
    "det 2x2 is exact on singular input" 0.0
    (Nd.get_f
       (host (Numpy.Linalg.det (m [| 2; 2 |] [| 1.; 2.; 3.; 6. |])))
       [||]);
  Alcotest.(check (float 0.0))
    "det 3x3 is exact on singular input" 0.0
    (Nd.get_f
       (host
          (Numpy.Linalg.det
             (m [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |])))
       [||]);
  let s, l =
    Numpy.Linalg.slogdet
      (T.Concrete (Nd.of_floats D.F16 [| 2; 2 |] [| 4.; 1.; 2.; 3. |]))
  in
  Alcotest.(check (float 0.0)) "slogdet sign" 1.0 (Nd.get_f (host s) [||]);
  Alcotest.(check (float 1e-3))
    "slogdet log" (Float.log 10.0)
    (Nd.get_f (host l) [||])

let () =
  Alcotest.run "hardening"
    [
      ( "files",
        [
          Alcotest.test_case "compressed npz" `Quick npz_compressed;
          Alcotest.test_case "npz checksum" `Quick npz_bad_checksum;
          Alcotest.test_case "narrow npy" `Quick npy_narrow;
          Alcotest.test_case "big endian, canonical dtype" `Quick
            npy_big_endian_and_canonical;
        ] );
      ("config", [ Alcotest.test_case "no_execution" `Quick no_execution ]);
      ( "linalg",
        [ Alcotest.test_case "eig derivative flag" `Quick eig_derivative_flag ]
      );
      ( "sweep",
        [
          Alcotest.test_case "complex special" `Quick complex_special;
          Alcotest.test_case "complex nn" `Quick complex_nn;
          Alcotest.test_case "complex stats" `Quick complex_stats;
          Alcotest.test_case "narrow stats" `Quick narrow_stats;
          Alcotest.test_case "scalar bugs" `Quick scalar_bugs;
          Alcotest.test_case "max and min ties" `Quick max_min_ties;
          Alcotest.test_case "primitive dtypes" `Quick primitive_dtypes;
          Alcotest.test_case "integer primitives" `Quick integer_primitives;
          Alcotest.test_case "conversions" `Quick conversions;
          Alcotest.test_case "narrow floats" `Quick narrow_floats;
          Alcotest.test_case "rounding" `Quick rounding;
          Alcotest.test_case "complex primitives" `Quick complex_primitives;
          Alcotest.test_case "special edges" `Quick special_edges;
          Alcotest.test_case "compiled paths" `Quick compiled_paths;
          Alcotest.test_case "numpy bugs" `Quick numpy_bugs;
          Alcotest.test_case "closed form det" `Quick closed_form_det;
          Alcotest.test_case "jit conj of a float" `Quick jit_conj_of_real;
        ] );
    ]
