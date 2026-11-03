module U = Yojson.Safe.Util
module AC = Ojax.Numpy.Array_creation
module UF = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module Ad = Ojax.Interpreters.Ad
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module TU = Ojax.Tree_util
module Ode = Ojax.Ode

let name = "odeint"
let arr dt shape xs = Ojax.Types.Concrete (Nd.of_floats dt shape xs)
let leaf v = TU.Leaf v
let unleaf = function TU.Leaf v -> v | _ -> failwith "odeint: leaf"
let leaves t = TU.tree_leaves t

type inputs = {
  y0v : float array;
  kv : float;
  ts_stop : float;
  tl_stop : float;
  y1a : float array;
  y1b : float array;
}

let fixed =
  {
    y0v = [| 1.0; 2.0 |];
    kv = 0.5;
    ts_stop = 1.0;
    tl_stop = 3.0;
    y1a = [| 1.0; 0.0 |];
    y1b = [| 0.0; 1.0 |];
  }

let inputs_of row =
  match U.member "inputs" row with
  | `Null -> fixed
  | j ->
      let f k = U.to_number (U.member k j) in
      let fs k =
        Array.of_list (List.map U.to_number (U.to_list (U.member k j)))
      in
      {
        y0v = fs "y0";
        kv = f "k";
        ts_stop = f "ts_stop";
        tl_stop = f "tl_stop";
        y1a = fs "y1a";
        y1b = fs "y1b";
      }

let y0 i dt = arr dt [| 2 |] i.y0v
let ts i dt = AC.linspace ~num:5 ~dtype:dt 0.0 i.ts_stop
let tl i dt = AC.linspace ~num:4 ~dtype:dt 0.0 i.tl_stop
let kk i dt = arr dt [||] [| i.kv |]
let decay y _ _ = TU.Leaf (UF.negative (unleaf y))

let scaled y _ = function
  | [ k ] -> leaf (UF.multiply (UF.negative k) (unleaf y))
  | _ -> failwith "arity"

let forced y t = function
  | [ k ] ->
      let y = unleaf y in
      leaf (UF.subtract (UF.sin t) (UF.multiply (UF.multiply k y) y))
  | _ -> failwith "arity"

let osc y _ _ =
  match y with
  | TU.Tuple [ x; v ] -> TU.Tuple [ v; leaf (UF.negative (unleaf x)) ]
  | _ -> failwith "osc"

let y1 i dt =
  TU.Tuple [ leaf (arr dt [| 2 |] i.y1a); leaf (arr dt [| 2 |] i.y1b) ]

let ode ?rtol ?atol ?mxstep ?hmax f y t args =
  Ode.odeint ?rtol ?atol ?mxstep ?hmax f y t args

let sum_ode f t args = function
  | y :: rest -> R.sum (unleaf (ode f (leaf y) t (args rest)))
  | [] -> failwith "arity"

let value_cases i dt =
  let y0 = y0 i dt and ts = ts i dt and k = kk i dt and tl = tl i dt in
  [
    ("decay", fun () -> leaves (ode decay (leaf y0) ts []));
    ("scaled", fun () -> leaves (ode scaled (leaf y0) ts [ k ]));
    ("forced", fun () -> leaves (ode forced (leaf y0) tl [ k ]));
    ("osc", fun () -> leaves (ode osc (y1 i dt) tl []));
    ( "tol",
      fun () -> leaves (ode ~rtol:1e-3 ~atol:1e-4 forced (leaf y0) tl [ k ]) );
    ( "mxstep",
      fun () -> leaves (ode ~mxstep:(Ode.Mx_int 3) forced (leaf y0) tl [ k ]) );
    ("hmax", fun () -> leaves (ode ~hmax:0.05 scaled (leaf y0) ts [ k ]));
    ( "jit_scaled",
      fun () ->
        Ojax.Api.jit_flat
          (function
            | [ y0; ts; k ] -> leaves (ode scaled (leaf y0) ts [ k ])
            | _ -> failwith "arity")
          [ y0; ts; k ] );
    ( "grad_decay",
      fun () ->
        Ad.grad
          (function
            | [ y0; ts ] -> R.sum (unleaf (ode decay (leaf y0) ts []))
            | _ -> failwith "arity")
          [ y0; ts ] );
    ( "grad_scaled",
      fun () ->
        Ad.grad
          (function
            | [ y0; ts; k ] -> R.sum (unleaf (ode scaled (leaf y0) ts [ k ]))
            | _ -> failwith "arity")
          [ y0; ts; k ] );
    ( "grad_forced",
      fun () ->
        Ad.grad
          (function
            | [ y0; _; k ] ->
                let o = unleaf (ode forced (leaf y0) tl [ k ]) in
                R.sum (UF.multiply o o)
            | _ -> failwith "arity")
          [ y0; tl; k ] );
    ( "grad_osc",
      fun () ->
        Ad.grad
          (function
            | [ a; b ] -> (
                match ode osc (TU.Tuple [ leaf a; leaf b ]) tl [] with
                | TU.Tuple [ x; _ ] -> R.sum (unleaf x)
                | _ -> failwith "osc")
            | _ -> failwith "arity")
          (leaves (y1 i dt)) );
    ( "int_t",
      fun () ->
        leaves
          (ode decay (leaf y0)
             (Ojax.Numpy.Lax_numpy.arange
                ~dtype:(Ojax.Dtypes.default_int_dtype ())
                3.0)
             []) );
  ]

let traced dt =
  [
    ( "jaxpr_decay",
      function
      | [ y0; ts ] -> leaves (ode decay (leaf y0) ts [])
      | _ -> failwith "arity" );
    ( "jaxpr_scaled",
      function
      | [ y0; ts; k ] -> leaves (ode scaled (leaf y0) ts [ k ])
      | _ -> failwith "arity" );
    ( "jaxpr_grad_scaled",
      function
      | [ y0; ts; k ] ->
          Ad.grad
            (function
              | [ y0; k ] -> R.sum (unleaf (ode scaled (leaf y0) ts [ k ]))
              | _ -> failwith "arity")
            [ y0; k ]
      | _ -> failwith "arity" );
  ]
  |> List.map (fun (n, f) -> (n ^ "_" ^ dt, f))

let value_cases i =
  let tag dname dt =
    List.map (fun (n, f) -> (n ^ "_" ^ dname, f)) (value_cases i dt)
  in
  fun id ->
    match List.assoc_opt id (tag "f32" D.F32) with
    | Some f -> Some f
    | None -> List.assoc_opt id (tag "f64" D.F64)

let traced id =
  match List.assoc_opt id (traced "f32") with
  | Some f -> Some f
  | None -> List.assoc_opt id (traced "f64")

let aval_of_arg a : Ojax.Types.aval =
  match U.to_list a with
  | [ dt; shape ] ->
      {
        dtype =
          (match D.of_numpy_name (U.to_string dt) with
          | Some d -> d
          | None -> failwith "odeint: dtype");
        shape = Array.of_list (List.map U.to_int (U.to_list shape));
        weak_type = false;
      }
  | _ -> failwith "odeint: arg"

let read_rows path =
  let ic = open_in path in
  let rec loop acc =
    match input_line ic with
    | line -> loop (Yojson.Safe.from_string line :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  loop []

let str_opt row field =
  match U.member field row with `String s -> Some s | _ -> None

let host = function
  | Ojax.Types.Concrete nd -> nd
  | Ojax.Types.Device b -> Ojax.Pjrt.Buffer.to_host b
  | Ojax.Types.Tracer _ -> failwith "odeint: tracer result"

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let error_detail e =
  match Ojax.Py_exn.class_name e with
  | Some c -> c ^ ": " ^ Error_types.message_of e
  | None -> Error_types.ojax_class e ^ ": " ^ Error_types.message_of e

let out_text v =
  let raw = Npz.raw_of_nd (host v) in
  raw.Npz.raw_dtype ^ "["
  ^ String.concat ","
      (Array.to_list (Array.map string_of_int (Nd.shape (host v))))
  ^ "]" ^ hex raw.Npz.bytes

let jax_out o =
  U.to_string (U.member "dtype" o)
  ^ "["
  ^ String.concat ","
      (List.map
         (fun s -> string_of_int (U.to_int s))
         (U.to_list (U.member "shape" o)))
  ^ "]"
  ^ U.to_string (U.member "hex" o)

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log ?(name = id) jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(String.escaped jax ^ "\t" ^ String.escaped ojax)
      ~suite name 1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical else "differ")
  in
  let expected_error =
    match (str_opt row "error_class", str_opt row "message") with
    | Some c, Some m -> Some (c ^ ": " ^ m)
    | _ -> None
  in
  match U.member "args" row with
  | `List args -> (
      let avals = List.map aval_of_arg args in
      let closed =
        match traced id with
        | None -> Error "no_binding"
        | Some f -> (
            try Ok (Ojax.Jaxpr.make_jaxpr avals f)
            with e -> Error (error_detail e))
      in
      match (expected_error, closed) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            ojax;
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            ojax
      | None, Ok c ->
          let text f = try f c with e -> error_detail e in
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            (text Ojax.Pretty_printer.closed_jaxpr_to_jax_string);
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            (text (Ojax.Stablehlo.Emit.emit_jax_text ~name:"_lambda")))
  | _ -> (
      let got =
        match value_cases (inputs_of row) id with
        | None -> Error "no_binding"
        | Some f -> (
            try Ok (List.map out_text (f ())) with e -> Error (error_detail e))
      in
      match (expected_error, got) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          List.iteri
            (fun i o ->
              log ~name:(Printf.sprintf "%s:out%d" id i) (jax_out o) ojax)
            (U.to_list (U.member "outs" row))
      | None, Ok outs ->
          let jax = List.map jax_out (U.to_list (U.member "outs" row)) in
          if List.compare_lengths jax outs <> 0 then
            log ~name:(id ^ ":count")
              (string_of_int (List.length jax))
              (string_of_int (List.length outs))
          else
            List.iteri
              (fun i (j, o) -> log ~name:(Printf.sprintf "%s:out%d" id i) j o)
              (List.combine jax outs))

let path ~goldens_root =
  Filename.concat (Filename.concat goldens_root name) "cases.jsonl"

let suites ~goldens_root =
  if not (Sys.file_exists (path ~goldens_root)) then []
  else
    let rows = read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              let x64 = U.to_bool (U.member "x64" row) in
              if mode = "x64_on" = x64 then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
