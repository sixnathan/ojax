open Ojax
open Ojax.Types
module Ad = Ojax.Interpreters.Ad

let () = Lax.install ()
let b1 = Core.bind1
let f64 shape xs = Concrete (Ndarray.of_floats Dtype.F64 shape xs)

let to_nd = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected a concrete value"

let flat nd =
  List.rev (Ndarray.fold (fun acc x -> x :: acc) [] nd) |> Array.of_list

let sum_squares v =
  let ndim = Array.length (Core.get_aval v).shape in
  b1 (Reduce_sum (Array.init ndim Fun.id)) [ b1 Mul [ v; v ] ]

let fill n = Array.init n (fun i -> 0.3 +. (0.7 *. sin (float_of_int (i + 1))))

let check_grad name f shape =
  let n = Array.fold_left ( * ) 1 shape in
  let x0 = fill n in
  let objective a = sum_squares (f a) in
  let g =
    flat
      (to_nd
         (List.hd (Ad.grad (fun a -> objective (List.hd a)) [ f64 shape x0 ])))
  in
  let eval x = Ndarray.get_f (to_nd (objective (f64 shape x))) [||] in
  let eps = 1e-6 in
  Array.iteri
    (fun i gi ->
      let up = Array.copy x0 and down = Array.copy x0 in
      up.(i) <- up.(i) +. eps;
      down.(i) <- down.(i) -. eps;
      let numeric = (eval up -. eval down) /. (2.0 *. eps) in
      Alcotest.(check (float 1e-4)) (name ^ string_of_int i) numeric gi)
    g

let matrix = f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let dot_lhs () =
  let dims =
    {
      lhs_contract = [| 1 |];
      rhs_contract = [| 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  check_grad "lhs" (fun a -> b1 (Dot_general dims) [ a; matrix ]) [| 2; 2 |];
  let rhs_dims =
    {
      lhs_contract = [| 1 |];
      rhs_contract = [| 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  check_grad "rhs" (fun a -> b1 (Dot_general rhs_dims) [ matrix; a ]) [| 3; 2 |]

let dot_batched () =
  let dims =
    {
      lhs_contract = [| 2 |];
      rhs_contract = [| 1 |];
      lhs_batch = [| 0 |];
      rhs_batch = [| 0 |];
    }
  in
  let other = f64 [| 2; 3; 2 |] (Array.init 12 float_of_int) in
  check_grad "batched"
    (fun a -> b1 (Dot_general dims) [ a; other ])
    [| 2; 4; 3 |];
  let contract_two =
    {
      lhs_contract = [| 0; 1 |];
      rhs_contract = [| 1; 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  let w = f64 [| 3; 2; 2 |] (Array.init 12 float_of_int) in
  check_grad "two contract"
    (fun a -> b1 (Dot_general contract_two) [ a; w ])
    [| 2; 3 |]

let dims1d =
  {
    lhs_spec = [| 0; 1; 2 |];
    rhs_spec = [| 0; 1; 2 |];
    out_spec = [| 0; 1; 2 |];
  }

let conv ?(strides = [| 1 |]) ?(padding = [| (0, 0) |]) ?(ld = [| 1 |])
    ?(rd = [| 1 |]) ?(fg = 1) ?(bg = 1) lhs rhs =
  b1
    (Conv_general_dilated
       {
         window_strides = strides;
         padding;
         lhs_dilation = ld;
         rhs_dilation = rd;
         dimension_numbers = dims1d;
         feature_group_count = fg;
         batch_group_count = bg;
       })
    [ lhs; rhs ]

let conv_lhs_rhs () =
  let kernel = f64 [| 1; 1; 3 |] [| 1.; -2.; 0.5 |] in
  check_grad "lhs" (fun a -> conv a kernel) [| 1; 1; 6 |];
  check_grad "lhs stride"
    (fun a -> conv ~strides:[| 2 |] ~padding:[| (1, 1) |] a kernel)
    [| 1; 1; 6 |];
  let input = f64 [| 1; 1; 6 |] (fill 6) in
  check_grad "rhs" (fun a -> conv input a) [| 1; 1; 3 |];
  check_grad "rhs stride dilated"
    (fun a -> conv ~strides:[| 2 |] ~rd:[| 2 |] ~padding:[| (2, 1) |] input a)
    [| 1; 1; 3 |]

let conv_groups () =
  let kernel = f64 [| 2; 1; 2 |] [| 1.; 2.; 3.; -1. |] in
  check_grad "feature group lhs" (fun a -> conv ~fg:2 a kernel) [| 1; 2; 4 |];
  let input = f64 [| 1; 2; 4 |] (fill 8) in
  check_grad "feature group rhs" (fun a -> conv ~fg:2 input a) [| 2; 1; 2 |];
  let batched = f64 [| 2; 1; 4 |] (fill 8) in
  check_grad "batch group lhs" (fun a -> conv ~bg:2 a kernel) [| 2; 1; 4 |];
  check_grad "batch group rhs" (fun a -> conv ~bg:2 batched a) [| 2; 1; 2 |]

let pad_rule () =
  let zero = f64 [||] [| 0.5 |] in
  check_grad "pad interior"
    (fun a -> b1 (Pad [| (1, 2, 1) |]) [ a; zero ])
    [| 3 |];
  check_grad "pad negative"
    (fun a -> b1 (Pad [| (-1, 1, 0); (1, 0, 1) |]) [ a; zero ])
    [| 3; 2 |]

let window ?(strides = [| 1; 2 |]) ?(padding = [| (0, 0); (0, 0) |]) () =
  {
    window_dimensions = [| 1; 2 |];
    window_strides = strides;
    w_padding = padding;
    base_dilation = [| 1; 1 |];
    window_dilation = [| 1; 1 |];
  }

let windows () =
  check_grad "sum"
    (fun a -> b1 (Reduce_window_sum (window ())) [ a ])
    [| 2; 4 |];
  check_grad "sum padded"
    (fun a ->
      b1 (Reduce_window_sum (window ~padding:[| (0, 0); (1, 1) |] ())) [ a ])
    [| 2; 4 |];
  check_grad "max"
    (fun a -> b1 (Reduce_window_max (window ())) [ a ])
    [| 2; 4 |];
  check_grad "min"
    (fun a -> b1 (Reduce_window_min (window ~strides:[| 1; 1 |] ())) [ a ])
    [| 2; 4 |]

let spd_of a =
  let dims =
    {
      lhs_contract = [| 1 |];
      rhs_contract = [| 1 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  let n = (Core.get_aval a).shape.(0) in
  let eye =
    f64 [| n; n |]
      (Array.init (n * n) (fun i -> if i mod (n + 1) = 0 then 4.0 else 0.0))
  in
  b1 Add [ b1 (Dot_general dims) [ a; a ]; eye ]

let ramp n =
  let total = n * n in
  f64 [| n; n |] (Array.init total (fun i -> 1.0 +. (0.5 *. float_of_int i)))

let cholesky_rule () =
  check_grad "cholesky" (fun a -> b1 Cholesky [ spd_of a ]) [| 3; 3 |]

let triangular_rules () =
  let solve ~left a b =
    b1
      (Triangular_solve
         {
           left_side = left;
           lower = true;
           transpose_a = false;
           conjugate_a = false;
           unit_diagonal = false;
         })
      [ a; b ]
  in
  let lower_a = f64 [| 2; 2 |] [| 2.; 0.; 1.; 3. |] in
  let rhs = f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  check_grad "b left" (fun b -> solve ~left:true lower_a b) [| 2; 2 |];
  check_grad "b right" (fun b -> solve ~left:false lower_a b) [| 2; 2 |];
  check_grad "a left"
    (fun a -> solve ~left:true (b1 Add [ a; ramp 2 ]) rhs)
    [| 2; 2 |];
  check_grad "a right"
    (fun a -> solve ~left:false (b1 Add [ a; ramp 2 ]) rhs)
    [| 2; 2 |]

let tridiagonal_rule () =
  let dl = f64 [| 3 |] [| 0.; 1.; 2. |] in
  let d = f64 [| 3 |] [| 5.; 6.; 7. |] in
  let du = f64 [| 3 |] [| 1.; 2.; 0. |] in
  let b = f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_grad "b" (fun x -> b1 Tridiagonal_solve [ dl; d; du; x ]) [| 3; 2 |];
  check_grad "d"
    (fun x -> b1 Tridiagonal_solve [ dl; b1 Add [ d; x ]; du; b ])
    [| 3 |]

let eigh_rule () =
  let sym a = b1 Add [ a; b1 (Transpose [| 1; 0 |]) [ a ] ] in
  let spread = f64 [| 3; 3 |] [| 1.; 0.; 0.; 0.; 3.; 0.; 0.; 0.; 6. |] in
  let run a = Core.bind (Eigh { lower = true }) [ b1 Add [ sym a; spread ] ] in
  check_grad "values" (fun a -> List.nth (run a) 1) [| 3; 3 |];
  check_grad "vectors"
    (fun a -> b1 Mul [ List.nth (run a) 0; ramp 3 ])
    [| 3; 3 |]

let qr_rule () =
  check_grad "q"
    (fun a -> b1 Mul [ List.nth (Core.bind (Qr false) [ a ]) 0; ramp 3 ])
    [| 3; 3 |];
  check_grad "r" (fun a -> List.nth (Core.bind (Qr false) [ a ]) 1) [| 3; 3 |];
  check_grad "tall"
    (fun a -> List.nth (Core.bind (Qr false) [ a ]) 1)
    [| 3; 2 |]

let lu_rule () =
  check_grad "square" (fun a -> List.hd (Core.bind Lu [ a ])) [| 3; 3 |];
  check_grad "tall" (fun a -> List.hd (Core.bind Lu [ a ])) [| 3; 2 |]

let svd_rule () =
  let run a =
    Core.bind (Svd { full_matrices = false; compute_uv = true }) [ a ]
  in
  List.iter
    (fun shape ->
      let name = Printf.sprintf "%dx%d " shape.(0) shape.(1) in
      check_grad (name ^ "s") (fun a -> List.nth (run a) 1) shape;
      check_grad (name ^ "u")
        (fun a ->
          let u = List.nth (run a) 0 in
          b1 Mul
            [
              u;
              f64 (Core.get_aval u).shape
                (fill (Array.fold_left ( * ) 1 (Core.get_aval u).shape));
            ])
        shape;
      check_grad (name ^ "vt")
        (fun a ->
          let vt = List.nth (run a) 2 in
          b1 Mul
            [
              vt;
              f64 (Core.get_aval vt).shape
                (fill (Array.fold_left ( * ) 1 (Core.get_aval vt).shape));
            ])
        shape)
    [ [| 3; 3 |]; [| 3; 2 |]; [| 2; 3 |] ]

let constant_output () =
  let f a =
    match a with
    | [ x ] -> [ b1 Mul [ x; x ]; f64 [||] [| 7.0 |] ]
    | _ -> Alcotest.fail "arity"
  in
  let x = f64 [||] [| 3.0 |] in
  let outs, pull = Ad.vjp f [ x ] in
  Alcotest.(check int) "outputs" 2 (List.length outs);
  (match pull [ f64 [||] [| 1.0 |]; f64 [||] [| 1.0 |] ] with
  | [ g ] ->
      Alcotest.(check (float 1e-9))
        "only the dependent output contributes" 6.0
        (Ndarray.get_f (to_nd g) [||])
  | _ -> Alcotest.fail "arity");
  let _, lin = Ad.linearize f [ x ] in
  match lin [ f64 [||] [| 1.0 |] ] with
  | [ t0; t1 ] ->
      Alcotest.(check (float 1e-9))
        "tangent" 6.0
        (Ndarray.get_f (to_nd t0) [||]);
      Alcotest.(check (float 1e-9)) "zero" 0.0 (Ndarray.get_f (to_nd t1) [||])
  | _ -> Alcotest.fail "arity"

let fft_rules () =
  let module N = Ojax.Numpy.Fft in
  check_grad "rfft" (fun a -> b1 Real [ N.rfft a ]) [| 6 |];
  check_grad "rfft odd" (fun a -> b1 Imag [ N.rfft a ]) [| 7 |];
  check_grad "fft"
    (fun a ->
      b1 Real [ N.fft (b1 (Convert_element_type Dtype.Complex128) [ a ]) ])
    [| 5 |];
  check_grad "irfft"
    (fun a -> N.irfft (b1 (Convert_element_type Dtype.Complex128) [ a ]))
    [| 5 |];
  check_grad "rfft2" (fun a -> b1 Real [ N.rfft2 a ]) [| 3; 4 |]

let () =
  Alcotest.run "grad_checks"
    [
      ("fft", [ Alcotest.test_case "transposes" `Quick fft_rules ]);
      ( "outputs",
        [ Alcotest.test_case "independent of input" `Quick constant_output ] );
      ( "linalg",
        [
          Alcotest.test_case "cholesky" `Quick cholesky_rule;
          Alcotest.test_case "triangular_solve" `Quick triangular_rules;
          Alcotest.test_case "tridiagonal_solve" `Quick tridiagonal_rule;
          Alcotest.test_case "eigh" `Quick eigh_rule;
          Alcotest.test_case "qr" `Quick qr_rule;
          Alcotest.test_case "lu" `Quick lu_rule;
          Alcotest.test_case "svd" `Quick svd_rule;
        ] );
      ( "dot_general",
        [
          Alcotest.test_case "lhs and rhs" `Quick dot_lhs;
          Alcotest.test_case "batched" `Quick dot_batched;
        ] );
      ( "conv",
        [
          Alcotest.test_case "lhs and rhs" `Quick conv_lhs_rhs;
          Alcotest.test_case "groups" `Quick conv_groups;
        ] );
      ("pad", [ Alcotest.test_case "transpose" `Quick pad_rule ]);
      ("windows", [ Alcotest.test_case "transpose" `Quick windows ]);
    ]
