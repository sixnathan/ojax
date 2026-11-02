module U = Yojson.Safe.Util
module T = Ojax.Types
module TU = Ojax.Tree_util
module C = Ojax.Core
module JU = Ojax.Numpy.Jnp_util
module UF = Ojax.Numpy.Ufuncs
module NL = Ojax.Numpy.Lax_numpy
module RD = Ojax.Numpy.Reductions
module AC = Ojax.Numpy.Array_creation
module LA = Ojax.Numpy.Linalg
module TC = Ojax.Numpy.Tensor_contractions
module SO = Ojax.Numpy.Sorting
module NN = Ojax.Nn.Functions
module SS = Ojax.Scipy.Special
module RC = Ojax.Random.Core
module A = Ojax.Api
module CD = Ojax.Custom_derivatives
module CF = Ojax.Lax_api
module PP = Ojax.Pretty_printer

let name = "compose_fuzz"
let leaf v = TU.Leaf v
let leaves = TU.tree_leaves
let as_leaf = function TU.Leaf v -> v | _ -> failwith "compose: not a leaf"
let arrays = List.map leaf
let values = List.map as_leaf

let tree_sum tree =
  match leaves tree with
  | [] -> failwith "compose: empty tree"
  | first :: rest ->
      List.fold_left
        (fun total v -> UF.add total (RD.sum v))
        (RD.sum first) rest

let unary op x =
  match op with
  | "sin" -> UF.sin x
  | "cos" -> UF.cos x
  | "tanh" -> UF.tanh x
  | "exp" -> UF.exp x
  | "expm1" -> UF.expm1 x
  | "sinh" -> UF.sinh x
  | "cosh" -> UF.cosh x
  | "arctan" -> UF.arctan x
  | "square" -> UF.square x
  | "abs" -> UF.abs x
  | "sum" -> RD.sum x
  | "mean" -> RD.mean x
  | "prod" -> RD.prod x
  | "max" -> RD.max x
  | "min" -> RD.min x
  | "var" -> RD.var x
  | "cumsum" -> RD.cumsum x
  | "cumprod" -> RD.cumprod x
  | "sort" -> SO.sort x
  | "log1p_sq" -> UF.log1p (UF.square x)
  | "sqrt_abs" -> UF.sqrt (UF.abs x)
  | "softmax" -> NN.softmax x
  | "logsumexp" -> SS.logsumexp x
  | "sigmoid" -> NN.sigmoid x
  | "relu" -> NN.relu x
  | "gelu" -> NN.gelu x
  | "softplus" -> NN.softplus x
  | "erf" -> SS.erf x
  | "gammaln" -> SS.gammaln x
  | "where_sq" -> JU.where (UF.greater x (JU.py_float 0.0)) (UF.square x) x
  | "clip" -> NL.clip ~min:(-1.0) ~max:1.0 x
  | "flip" -> NL.flip x
  | "ravel" -> NL.ravel x
  | "transpose" -> NL.transpose x
  | "concat" -> NL.concatenate [ x; x ]
  | "stack" -> NL.stack [ x; x ]
  | "norm" -> LA.norm x
  | "self_dot" -> TC.dot x x
  | "self_matmul" -> TC.matmul x x
  | "det" -> LA.det x
  | "inv" -> LA.inv x
  | "cholesky" -> LA.cholesky (TC.matmul x (NL.transpose x))
  | other -> failwith ("compose: unary " ^ other)

let binary op x y =
  match op with
  | "add" -> UF.add x y
  | "subtract" -> UF.subtract x y
  | "multiply" -> UF.multiply x y
  | "divide" -> UF.divide x y
  | "power" -> UF.power (UF.abs x) y
  | "arctan2" -> UF.arctan2 x y
  | "hypot" -> UF.hypot x y
  | "logaddexp" -> UF.logaddexp x y
  | "maximum" -> UF.maximum x y
  | "minimum" -> UF.minimum x y
  | "dot" -> TC.dot x y
  | "matmul" -> TC.matmul x y
  | "where_gt" -> JU.where (UF.greater x y) x y
  | other -> failwith ("compose: binary " ^ other)

let base op = function
  | [ x ] -> leaf (unary op x)
  | [ x; y ] -> leaf (binary op x y)
  | _ -> failwith "compose: base arity"

let scalarized f args = leaf (tree_sum (f args))

let step_value f c rest =
  UF.add c (UF.multiply (JU.py_float 0.125) (tree_sum (f (arrays (c :: rest)))))

let flat_call f =
  let shape = ref TU.Null_def in
  let flat args =
    let out = f (arrays args) in
    shape := TU.tree_structure out;
    leaves out
  in
  (flat, shape)

let axis_of = function `Null -> None | v -> Some (U.to_int v)

let apply_layer layer f =
  let argnum () = U.to_int (U.member "argnum" layer) in
  match U.to_string (U.member "t" layer) with
  | "jit" ->
      fun args ->
        let flat, shape = flat_call f in
        let out = A.jit_flat ~name:"g" flat (values args) in
        TU.tree_unflatten !shape out
  | "vmap" ->
      let in_axes = List.map axis_of (U.to_list (U.member "in_axes" layer)) in
      let out_axes = U.to_int (U.member "out_axes" layer) in
      fun args -> A.vmap ~out_axes f in_axes args
  | "grad" -> fun args -> A.grad ~argnums:(argnum ()) (scalarized f) args
  | "value_and_grad" ->
      fun args ->
        let v, g = A.value_and_grad ~argnums:(argnum ()) (scalarized f) args in
        TU.Tuple [ v; g ]
  | "jacfwd" -> fun args -> A.jacfwd ~argnums:(argnum ()) f args
  | "jacrev" -> fun args -> A.jacrev ~argnums:(argnum ()) f args
  | "hessian" -> fun args -> A.hessian ~argnums:(argnum ()) (scalarized f) args
  | "checkpoint" ->
      fun args ->
        let flat, shape = flat_call f in
        let out = Ojax.Checkpoint.checkpoint flat (values args) in
        TU.tree_unflatten !shape out
  | "jvp" ->
      fun args ->
        let tangents =
          List.map (TU.tree_map (fun x -> AC.full_like x 0.5)) args
        in
        let y, dy = A.jvp f args tangents in
        TU.Tuple [ y; dy ]
  | "vjp" ->
      fun args ->
        let y, pull = A.vjp f args in
        let cotangent = TU.tree_map (fun v -> AC.ones_like v) y in
        TU.Tuple [ y; TU.Tuple (pull cotangent) ]
  | "custom_jvp" ->
      fun args ->
        let flat, shape = flat_call f in
        let rule primals tangents =
          let y, dy = A.jvp f (arrays primals) (arrays tangents) in
          shape := TU.tree_structure y;
          ( leaves y,
            List.map (fun v -> UF.multiply v (JU.py_float 1.5)) (leaves dy) )
        in
        let out =
          CD.custom_jvp ~name:"g" ~jvp_name:"rule" flat ~jvp:rule (values args)
        in
        TU.tree_unflatten !shape out
  | "scan" -> (
      let length = U.to_int (U.member "length" layer) in
      fun args ->
        let a = values args in
        let rest = List.tl a in
        let body = function
          | c :: _ ->
              let s = step_value f c rest in
              [ s; RD.sum s ]
          | [] -> failwith "compose: scan body"
        in
        match CF.scan ~length body [ List.hd a ] [] with
        | [ c; ys ] -> TU.Tuple [ leaf c; leaf ys ]
        | _ -> failwith "compose: scan arity")
  | "while_loop" ->
      let length = U.to_int (U.member "length" layer) in
      fun args ->
        let a = values args in
        let rest = List.tl a in
        let test = function
          | i :: _ -> UF.less i (JU.py_int length)
          | [] -> failwith "compose: while test"
        in
        let body = function
          | [ i; c ] -> [ UF.add i (JU.py_int 1); step_value f c rest ]
          | _ -> failwith "compose: while body"
        in
        let start = Ojax.Numpy.Array_constructors.asarray (JU.py_int 0) in
        leaf (List.nth (CF.while_loop test body [ start; List.hd a ]) 1)
  | "cond" ->
      fun args ->
        let a = values args in
        let chosen v = [ as_leaf (scalarized f (arrays v)) ] in
        let other v = [ RD.sum (List.hd v) ] in
        leaf
          (List.hd
             (CF.cond
                (UF.greater (RD.sum (List.hd a)) (JU.py_float 0.0))
                chosen other a))
  | "switch" ->
      fun args ->
        let a = values args in
        let total = RD.sum (List.hd a) in
        let int32 v = C.bind1 (T.Convert_element_type Ojax.Dtype.I32) [ v ] in
        let index =
          UF.add
            (int32 (UF.greater total (JU.py_float 0.0)))
            (int32 (UF.greater total (JU.py_float 1.0)))
        in
        let chosen v = [ as_leaf (scalarized f (arrays v)) ] in
        let second v = [ RD.sum (List.hd v) ] in
        let third v = [ RD.sum (UF.sin (List.hd v)) ] in
        leaf (List.hd (CF.switch index [ chosen; second; third ] a))
  | other -> failwith ("compose: layer " ^ other)

let compose_program params =
  let op = U.to_string (U.member "base" params) in
  let start args = base op (values args) in
  List.fold_left
    (fun f layer -> apply_layer layer f)
    start
    (U.to_list (U.member "layers" params))

let promote_binop = function
  | "add" -> UF.add
  | "subtract" -> UF.subtract
  | "multiply" -> UF.multiply
  | "true_divide" -> UF.true_divide
  | "floor_divide" -> UF.floor_divide
  | "power" -> UF.power
  | "maximum" -> UF.maximum
  | "less" -> UF.less
  | "arctan2" -> UF.arctan2
  | "logaddexp" -> UF.logaddexp
  | "remainder" -> UF.remainder
  | "bitwise_and" -> UF.bitwise_and
  | other -> failwith ("compose: binop " ^ other)

let python_scalar spec =
  let v = U.member "v" spec in
  match U.to_string (U.member "py" spec) with
  | "bool" ->
      T.Concrete
        (Ojax.Ndarray.of_floats Ojax.Dtype.Bool [||]
           [| (if U.to_bool v then 1.0 else 0.0) |])
  | "int" -> JU.py_int (U.to_int v)
  | "float" -> JU.py_float (U.to_number v)
  | "complex" -> (
      match U.to_list v with
      | [ re; im ] ->
          JU.py_complex { Complex.re = U.to_number re; im = U.to_number im }
      | _ -> failwith "compose: complex scalar")
  | other -> failwith ("compose: python scalar " ^ other)

let promote_program params =
  let op = promote_binop (U.to_string (U.member "binop" params)) in
  let lhs = U.member "lhs" params and rhs = U.member "rhs" params in
  let is_arg spec = U.member "arg" spec <> `Null in
  fun args ->
    let a = values args in
    let operand spec position =
      if is_arg spec then List.nth a position else python_scalar spec
    in
    let right = if is_arg lhs then 1 else 0 in
    leaf (op (operand lhs 0) (operand rhs right))

let getitem k i =
  let shape = (C.get_aval k).shape in
  let limit = Array.copy shape in
  limit.(0) <- i + 1;
  let start = Array.make (Array.length shape) 0 in
  start.(0) <- i;
  Ojax.Lax_api.squeeze
    (C.bind1
       (T.Slice { start_indices = start; limit_indices = limit; strides = None })
       [ k ])
    [| 0 |]

let ints_of v = Array.of_list (List.map U.to_int (U.to_list v))

let prng_program params =
  let impl = U.to_string (U.member "impl" params) in
  let u32 n =
    T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| float_of_int n |])
  in
  fun args ->
    let seed = as_leaf (List.hd args) in
    let key = ref (RC.key ~impl seed) in
    let outs = ref [] in
    let member n step = U.to_int (U.member n step) in
    List.iter
      (fun step ->
        (match U.to_string (U.member "op" step) with
        | "fold_in" -> key := RC.fold_in !key (u32 (member "data" step))
        | "split" ->
            key :=
              getitem (RC.split !key (member "num" step)) (member "index" step)
        | "vmap_fold_in" ->
            let data = u32 (member "data" step) in
            let folded =
              A.vmap
                (fun k -> leaf (RC.fold_in (as_leaf (List.hd k)) data))
                [ Some 0 ]
                [ leaf (RC.split !key (member "num" step)) ]
            in
            key := getitem (as_leaf folded) (member "index" step)
        | other -> failwith ("compose: prng step " ^ other));
        outs := RC.key_data !key :: !outs)
      (U.to_list (U.member "steps" params));
    List.iter
      (fun draw ->
        let shape = ints_of (U.member "shape" draw) in
        let value =
          match U.to_string (U.member "op" draw) with
          | "uniform" -> RC.uniform !key ~shape ~minval:0.0 ~maxval:1.0
          | "normal" -> RC.normal !key ~shape
          | "bits" -> RC.bits !key ~shape
          | "randint" -> RC.randint !key ~shape ~minval:(-5) ~maxval:100
          | "bernoulli" -> RC.bernoulli !key ~shape ~p:0.25
          | "permutation" -> RC.permutation !key 5
          | other -> failwith ("compose: prng draw " ^ other)
        in
        outs := value :: !outs)
      (U.to_list (U.member "draws" params));
    TU.Tuple (arrays (List.rev !outs))

let program case =
  let params = U.member "params" case in
  match U.to_string (U.member "op" case) with
  | "compose" -> compose_program params
  | "promote" -> promote_program params
  | "prng" -> prng_program params
  | other -> failwith ("compose: family " ^ other)

let references = Hashtbl.create 1024

let load_references root =
  if Hashtbl.length references = 0 then
    let path =
      Filename.concat
        (Filename.concat (Filename.concat root name) "text")
        "reference.jsonl"
    in
    In_channel.with_open_text path (fun ic ->
        let rec go () =
          match In_channel.input_line ic with
          | None -> ()
          | Some line ->
              let row = Yojson.Safe.from_string line in
              Hashtbl.replace references
                (U.to_string (U.member "set" row)
                ^ "/"
                ^ U.to_string (U.member "case" row))
                row;
              go ()
        in
        go ())

let trace_debug n =
  {
    T.func_src_info = "f at compose:0";
    traced_for = "jit";
    arg_names = Some (List.init n (Printf.sprintf "xs[%d]"));
  }

let guarded f = try Ok (f ()) with e -> Error e

let dump_channel =
  lazy
    (Option.map
       (fun dir ->
         open_out_gen
           [ Open_append; Open_creat ]
           0o644
           (Filename.concat dir (Printf.sprintf "%d.jsonl" (Unix.getpid ()))))
       (Sys.getenv_opt "OJAX_COMPOSE_DUMP"))

let dump_text row field ours =
  match Lazy.force dump_channel with
  | None -> ()
  | Some oc ->
      let text =
        match ours with
        | Ok t -> t
        | Error e -> "ERROR " ^ Error_types.message_of e
      in
      Yojson.Safe.to_channel oc
        (`Assoc
           [
             ("case", `String row); ("field", `String field); ("v", `String text);
           ]);
      output_char oc '\n';
      flush oc

let text_row ~suite ~row reference field ours =
  dump_text (suite ^ "/" ^ row) field ours;
  let wanted = U.member field reference in
  let label = row ^ ":" ^ field in
  match (wanted, ours) with
  | `String w, Ok text when w = text -> Fuzz.log_row ~suite label 1 0 0.0 "same"
  | `String _, Ok _ -> Fuzz.log_row ~suite label 1 1 0.0 "text_diff"
  | `String _, Error e ->
      Fuzz.log_row ~suite label 0 (-1) 0.0
        ("ojax_raises:" ^ Fuzz.one_line (Error_types.message_of e))
  | _, Ok _ -> Fuzz.log_row ~suite label 0 (-2) 0.0 "jax_raises"
  | _, Error e -> (
      Fuzz.log_row ~suite label 0 0 0.0 Error_types.both_raise;
      match U.member (field ^ "_error") reference with
      | `String w ->
          let text_suite = "text_" ^ suite in
          if w = Error_types.message_of e then
            Fuzz.log_row ~suite:text_suite label 1 0 0.0 "same"
          else Fuzz.log_row ~suite:text_suite label 1 1 0.0 "text_diff"
      | _ -> ())

let check_text ~suite ~set ~root case avals run =
  load_references root;
  let case_id = U.to_string (U.member "case_id" case) in
  match Hashtbl.find_opt references (name ^ "/" ^ set ^ "/" ^ case_id) with
  | None -> Fuzz.log_row ~suite (case_id ^ ":text") 0 (-7) 0.0 "no_reference"
  | Some reference ->
      let shape = ref TU.Null_def in
      let lowered =
        guarded (fun () ->
            Ojax.Jaxpr.with_trace_debug
              (trace_debug (List.length avals))
              (fun () ->
                Ojax.Jaxpr.make_jaxpr avals (fun args ->
                    let out = run (arrays args) in
                    shape := TU.tree_structure out;
                    leaves out)))
      in
      let render f =
        match lowered with
        | Error e -> Error e
        | Ok closed -> guarded (fun () -> f closed)
      in
      text_row ~suite ~row:case_id reference "jaxpr"
        (render PP.closed_jaxpr_to_jax_string);
      text_row ~suite ~row:case_id reference "text"
        (render (fun closed ->
             Ojax.Stablehlo.Emit.emit_jax_text
               ~result_names:(Ojax.Stages.result_names_of_tree !shape)
               ~name:"f" closed))

let check_case ~suite ~root ~set ~set_dir ~x64 ~nd_of_npz ~string_of_dtype
    ~concrete case () =
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let case_id = U.to_string (U.member "case_id" case) in
  let family = U.to_string (U.member "op" case) in
  let args = U.to_list (U.member "args" case) in
  let arg_names = List.map (fun a -> U.to_string (U.member "name" a)) args in
  let b, table =
    Fuzz.bundle
      (Filename.concat (Filename.concat set_dir "ops") (family ^ ".npz"))
  in
  let input arg =
    let m = Hashtbl.find table (case_id ^ "/in_" ^ arg) in
    T.Concrete (nd_of_npz (Npz.decode b m.Fuzz.start m.Fuzz.descr m.Fuzz.shape))
  in
  let inputs = List.map input arg_names in
  let run = program case in
  let jax_error key =
    match U.member key case with `String n -> n | _ -> "unknown"
  in
  Fuzz.check_mode ~suite ~case_id ~mode:"eager" ~string_of_dtype ~concrete
    ~jax_error:(jax_error "eager_error") b table
    (Fuzz.output_names case "eager") (fun () -> leaves (run (arrays inputs)));
  Fuzz.check_mode ~suite ~case_id ~mode:"jit" ~string_of_dtype ~concrete
    ~jax_error:(jax_error "jit_error") b table (Fuzz.output_names case "jit")
    (fun () -> A.jit_flat (fun a -> leaves ((program case) (arrays a))) inputs);
  check_text ~suite ~set ~root case
    (List.map (fun v -> C.get_aval v) inputs)
    (program case)

let suites ~goldens_root ~nd_of_npz ~string_of_dtype ~concrete =
  List.filter_map
    (fun (set, x64) ->
      let set_dir = Filename.concat (Filename.concat goldens_root name) set in
      let path = Filename.concat set_dir "manifest.json" in
      if not (Sys.file_exists path) then None
      else
        let suite = name ^ ":" ^ set in
        let cases =
          Yojson.Safe.from_file path |> U.member "cases" |> U.to_list
        in
        Some
          ( suite,
            List.map
              (fun case ->
                Alcotest.test_case
                  (U.to_string (U.member "case_id" case))
                  `Quick
                  (check_case ~suite ~root:goldens_root ~set ~set_dir ~x64
                     ~nd_of_npz ~string_of_dtype ~concrete case))
              cases ))
    [ ("x64_off", false); ("x64_on", true) ]
