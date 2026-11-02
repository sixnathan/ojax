module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module NL = Ojax.Numpy.Lax_numpy
module RD = Ojax.Numpy.Reductions
module LA = Ojax.Numpy.Linalg
module TC = Ojax.Numpy.Tensor_contractions
module E = Ojax.Numpy.Extras

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let empty2 = f32 [| 0; 0 |] [||]
let empty1 = f32 [| 0 |] [||]
let scalar = f32 [||] [| 3.0 |]

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Ojax.Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "tracer escaped"

let values nd =
  let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
  let a = Array.make n 0.0 in
  let _ =
    Nd.fold
      (fun i x ->
        a.(i) <- x;
        i + 1)
      0 nd
  in
  a

let check name dtype shape expected f input () =
  let verify mode v =
    let nd = host v in
    Alcotest.(check (array int)) (name ^ " shape " ^ mode) shape (Nd.shape nd);
    Alcotest.(check bool) (name ^ " dtype " ^ mode) true (Nd.dtype nd = dtype);
    Alcotest.(check (array (float 0.0)))
      (name ^ " values " ^ mode)
      expected (values nd)
  in
  verify "eager" (f input);
  verify "jit"
    (List.hd (Ojax.Api.jit_flat (fun args -> [ f (List.hd args) ]) [ input ]))

let case name dtype shape expected f input =
  Alcotest.test_case name `Quick (check name dtype shape expected f input)

let () =
  Alcotest.run "empty_shapes"
    [
      ( "jax values",
        [
          case "cumsum" D.F32 [| 0; 0 |] [||] (RD.cumsum ~axis:0) empty2;
          case "diag" D.F32 [| 0 |] [||] NL.diag empty2;
          case "diff" D.F32 [| 0; 0 |] [||] (NL.diff ~axis:0) empty2;
          case "det" D.F32 [||] [| 1.0 |] LA.det empty2;
          case "inv" D.F32 [| 0; 0 |] [||] LA.inv empty2;
          case "corrcoef" D.F32 [| 0; 0 |] [||] (fun v -> NL.corrcoef v) empty2;
          case "repeat" D.F32 [| 2 |] [| 3.0; 3.0 |]
            (fun v -> NL.repeat ~axis:0 v 2)
            scalar;
          case "dot" D.F32 [||] [| 9.0 |]
            (fun v -> TC.dot v (NL.transpose v))
            scalar;
          case "unique" D.F32 [| 2 |] [| 7.0; 7.0 |]
            (fun v -> List.hd (E.unique ~size:2 ~fill_value:7.0 v))
            empty1;
          case "nonzero" D.I32 [| 2 |] [| 0.0; 0.0 |]
            (fun v ->
              List.hd
                (E.nonzero ~size:2 ~fill_value:(Ojax.Numpy.Jnp_util.py_int 1) v))
            (f32 [| 3; 0 |] [||]);
          case "hankel" D.F32 [| 0; 0 |] [||]
            (fun v -> Ojax.Scipy.Linalg_special.hankel v)
            empty1;
        ] );
    ]
