open Types
module Tree = Tree_util
module L = Lax_api
module NN = Nn.Functions

type layer = {
  init : value -> int array -> int array * value Tree.t;
  apply : ?rng:value -> value Tree.t -> value -> value;
}

let dtype_default () = Dtypes.default_float_dtype ()

let glorot_scale in_size out_size =
  sqrt (2.0 /. float_of_int (in_size + out_size))

let split2 key =
  let keys = Random.Core.split key 2 in
  (L.index_in_dim ~keepdims:false keys 0, L.index_in_dim ~keepdims:false keys 1)

let glorot key shape =
  let n = Array.length shape in
  let receptive =
    Array.fold_left ( * ) 1 (Array.sub shape 0 (Stdlib.max 0 (n - 2)))
  in
  let fan_in = shape.(n - 2) * receptive
  and fan_out = shape.(n - 1) * receptive in
  let std = glorot_scale fan_in fan_out in
  let sample = Random.Core.normal key ~shape in
  L.mul sample (L.full_like sample std)

let normal_init key shape =
  let sample = Random.Core.normal key ~shape in
  L.mul sample (L.full_like sample 0.01)

let dense ?(w_init = glorot) ?(b_init = normal_init) out_dim =
  {
    init =
      (fun rng input_shape ->
        let n = Array.length input_shape in
        let k1, k2 = split2 rng in
        let w = w_init k1 [| input_shape.(n - 1); out_dim |] in
        let b = b_init k2 [| out_dim |] in
        let out = Array.copy input_shape in
        out.(n - 1) <- out_dim;
        (out, Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ]));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ->
            let product =
              let nx = Array.length (Core.get_aval x).shape in
              L.dot_general x w
                ~lhs_contract:[| nx - 1 |]
                ~rhs_contract:[| 0 |] ~lhs_batch:[||] ~rhs_batch:[||]
            in
            L.add product
              (L.broadcast_in_dim b (Core.get_aval product).shape
                 [| Array.length (Core.get_aval product).shape - 1 |])
        | _ -> invalid_arg "dense: bad parameters");
  }

let elementwise f =
  {
    init = (fun _ input_shape -> (input_shape, Tree.Tuple []));
    apply = (fun ?rng:_ _ x -> f x);
  }

let tanh = elementwise L.tanh
let relu = elementwise NN.relu
let exp = elementwise L.exp
let log_softmax = elementwise (fun x -> NN.log_softmax x)
let softmax = elementwise (fun x -> NN.softmax x)
let softplus = elementwise NN.softplus
let sigmoid = elementwise NN.sigmoid
let elu = elementwise (fun x -> NN.elu x)
let leaky_relu = elementwise (fun x -> NN.leaky_relu x)
let selu = elementwise NN.selu
let gelu = elementwise (fun x -> NN.gelu x)
let identity = elementwise Fun.id

let flatten =
  {
    init =
      (fun _ input_shape ->
        let rest =
          Array.fold_left ( * ) 1
            (Array.sub input_shape 1 (Array.length input_shape - 1))
        in
        ([| input_shape.(0); rest |], Tree.Tuple []));
    apply =
      (fun ?rng:_ _ x ->
        let sh = (Core.get_aval x).shape in
        let rest =
          Array.fold_left ( * ) 1 (Array.sub sh 1 (Array.length sh - 1))
        in
        L.reshape x [| sh.(0); rest |]);
  }

let conv_dims_nhwc rank =
  let spatial = Array.init (rank - 2) (fun i -> i + 1) in
  let spatial_k = Array.init (rank - 2) Fun.id in
  {
    lhs_spec = Array.append [| 0; rank - 1 |] spatial;
    rhs_spec = Array.append [| rank - 1; rank - 2 |] spatial_k;
    out_spec = Array.append [| 0; rank - 1 |] spatial;
  }

let general_conv ?strides ?(padding = L.Valid) ?(w_init = glorot)
    ?(b_init = normal_init) out_chan filter_shape =
  let k = Array.length filter_shape in
  let strides = Option.value strides ~default:(Array.make k 1) in
  let dn = conv_dims_nhwc (k + 2) in
  {
    init =
      (fun rng input_shape ->
        let in_chan = input_shape.(k + 1) in
        let k1, k2 = split2 rng in
        let w = w_init k1 (Array.append filter_shape [| in_chan; out_chan |]) in
        let spatial = Array.sub input_shape 1 k in
        let pads = L.padtype_to_pads spatial filter_shape strides padding in
        let out_spatial =
          Array.mapi
            (fun i s ->
              let lo, hi = pads.(i) in
              ((s + lo + hi - filter_shape.(i)) / strides.(i)) + 1)
            spatial
        in
        let b = b_init k2 [| out_chan |] in
        ( Array.concat [ [| input_shape.(0) |]; out_spatial; [| out_chan |] ],
          Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ->
            let out =
              L.conv_general_dilated ~dimension_numbers:dn x w
                ~window_strides:strides padding
            in
            let sh = (Core.get_aval out).shape in
            L.add out (L.broadcast_in_dim b sh [| Array.length sh - 1 |])
        | _ -> invalid_arg "conv: bad parameters");
  }

let conv = general_conv

let char_index spec c =
  match String.index_opt spec c with
  | Some i -> i
  | None -> invalid_arg ("stax: dimension spec " ^ spec ^ " lacks a role")

let general_conv_transpose ?strides ?(padding = L.Valid) ?w_init ?b_init
    ((lhs_spec, rhs_spec, out_spec) as dimension_numbers) out_chan filter_shape
    =
  let k = Array.length filter_shape in
  let strides = Option.value strides ~default:(Array.make k 1) in
  let dn = L.conv_dimension_numbers dimension_numbers in
  let w_init =
    match w_init with
    | Some f -> f
    | None ->
        let init =
          Nn.Initializers.glorot_normal ~in_axis:(char_index rhs_spec 'I')
            ~out_axis:(char_index rhs_spec 'O') ()
        in
        fun key shape -> init key ~shape
  in
  let b_init =
    match b_init with
    | Some f -> f
    | None ->
        let init = Nn.Initializers.normal ~stddev:1e-6 () in
        fun key shape -> init key ~shape
  in
  let bias_shape =
    Array.init (String.length out_spec) (fun i ->
        if out_spec.[i] = 'C' then out_chan else 1)
  in
  {
    init =
      (fun rng input_shape ->
        let filter_index = ref 0 in
        let kernel_shape =
          Array.init (String.length rhs_spec) (fun i ->
              match rhs_spec.[i] with
              | 'O' -> out_chan
              | 'I' -> input_shape.(char_index lhs_spec 'C')
              | _ ->
                  let d = filter_shape.(!filter_index) in
                  incr filter_index;
                  d)
        in
        let output_shape =
          L.conv_transpose_shape_tuple input_shape kernel_shape strides padding
            dn
        in
        let k1, k2 = split2 rng in
        let w = w_init k1 kernel_shape in
        let b = b_init k2 bias_shape in
        (output_shape, Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ]));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ->
            let out =
              L.conv_transpose ~dimension_numbers:dn x w ~strides padding
            in
            let sh = (Core.get_aval out).shape in
            let dims = Array.init (Array.length sh) Fun.id in
            L.add out (L.broadcast_in_dim b sh dims)
        | _ -> invalid_arg "conv_transpose: bad parameters");
  }

let conv1d_transpose ?strides ?padding ?w_init ?b_init out_chan filter_shape =
  general_conv_transpose ?strides ?padding ?w_init ?b_init ("NHC", "HIO", "NHC")
    out_chan filter_shape

let conv_transpose ?strides ?padding ?w_init ?b_init out_chan filter_shape =
  general_conv_transpose ?strides ?padding ?w_init ?b_init
    ("NHWC", "HWIO", "NHWC") out_chan filter_shape

type io_shape = Shape of int array | Shapes of io_shape list
type io_value = Value of value | Values of io_value list

type net = {
  net_init : value -> io_shape -> io_shape * value Tree.t;
  net_apply : ?rng:value -> value Tree.t -> io_value -> io_value;
}

let of_layer (layer : layer) =
  {
    net_init =
      (fun rng shape ->
        match shape with
        | Shape s ->
            let out, params = layer.init rng s in
            (Shape out, params)
        | Shapes _ -> invalid_arg "stax: layer expects a single input shape");
    net_apply =
      (fun ?rng params x ->
        match x with
        | Value v -> Value (layer.apply ?rng params v)
        | Values _ -> invalid_arg "stax: layer expects a single input");
  }

let fan_out num =
  {
    net_init =
      (fun _ shape -> (Shapes (List.init num (fun _ -> shape)), Tree.Tuple []));
    net_apply = (fun ?rng:_ _ x -> Values (List.init num (fun _ -> x)));
  }

let input_values = function
  | Values vs ->
      List.map
        (function
          | Value v -> v | Values _ -> invalid_arg "stax: nested fan-in input")
        vs
  | Value _ -> invalid_arg "stax: fan-in expects several inputs"

let input_shapes = function
  | Shapes ss ->
      List.map
        (function
          | Shape s -> s | Shapes _ -> invalid_arg "stax: nested fan-in shape")
        ss
  | Shape _ -> invalid_arg "stax: fan-in expects several shapes"

let fan_in_sum =
  {
    net_init =
      (fun _ shapes -> (Shape (List.hd (input_shapes shapes)), Tree.Tuple []));
    net_apply =
      (fun ?rng:_ _ x ->
        match input_values x with
        | first :: rest -> Value (List.fold_left L.add first rest)
        | [] -> invalid_arg "stax: fan-in of nothing");
  }

let fan_in_concat ?(axis = -1) () =
  let wrap_axis rank =
    let r = axis mod rank in
    if r < 0 then r + rank else r
  in
  {
    net_init =
      (fun _ shapes ->
        let shapes = input_shapes shapes in
        let first = List.hd shapes in
        let ax = wrap_axis (Array.length first) in
        let out = Array.copy first in
        out.(ax) <- List.fold_left (fun acc s -> acc + s.(ax)) 0 shapes;
        (Shape out, Tree.Tuple []));
    net_apply =
      (fun ?rng:_ _ x ->
        let vs = input_values x in
        let rank = Array.length (Core.get_aval (List.hd vs)).shape in
        Value (Core.bind1 (Concatenate (wrap_axis rank)) vs));
  }

let split_n key n =
  let keys = Random.Core.split key n in
  List.init n (fun i -> L.index_in_dim ~keepdims:false keys i)

let split_opt rng n =
  match rng with
  | Some key -> List.map Option.some (split_n key n)
  | None -> List.init n (fun _ -> None)

let param_list = function
  | Tree.Tuple ps -> ps
  | _ -> invalid_arg "stax: expected a sequence of parameters"

let serial_net nets =
  {
    net_init =
      (fun rng input_shape ->
        let _, shape, params_rev =
          List.fold_left
            (fun (key, shape, acc) net ->
              let next, layer_key = split2 key in
              let out_shape, p = net.net_init layer_key shape in
              (next, out_shape, p :: acc))
            (rng, input_shape, []) nets
        in
        (shape, Tree.Tuple (List.rev params_rev)));
    net_apply =
      (fun ?rng params x ->
        let rngs = split_opt rng (List.length nets) in
        let rec go x nets ps rngs =
          match (nets, ps, rngs) with
          | net :: nets, p :: ps, r :: rngs ->
              go (net.net_apply ?rng:r p x) nets ps rngs
          | _ -> x
        in
        go x nets (param_list params) rngs);
  }

let parallel nets =
  let n = List.length nets in
  {
    net_init =
      (fun rng input_shape ->
        let shapes =
          match input_shape with
          | Shapes ss -> ss
          | Shape _ -> invalid_arg "stax: parallel expects several shapes"
        in
        let rec go nets keys shapes =
          match (nets, keys, shapes) with
          | net :: nets, key :: keys, s :: shapes ->
              net.net_init key s :: go nets keys shapes
          | _ -> []
        in
        let results = go nets (split_n rng n) shapes in
        (Shapes (List.map fst results), Tree.Tuple (List.map snd results)));
    net_apply =
      (fun ?rng params x ->
        let xs =
          match x with
          | Values vs -> vs
          | Value _ -> invalid_arg "stax: parallel expects several inputs"
        in
        let rec go nets ps xs rngs =
          match (nets, ps, xs, rngs) with
          | net :: nets, p :: ps, v :: xs, r :: rngs ->
              net.net_apply ?rng:r p v :: go nets ps xs rngs
          | _ -> []
        in
        Values (go nets (param_list params) xs (split_opt rng n)));
  }

let pool_layer ~reducer ~init_value ?(normalize = false) ?strides
    ?(padding = L.Valid) window_shape =
  let k = Array.length window_shape in
  let strides = Option.value strides ~default:(Array.make k 1) in
  let wrap a fill = Array.concat [ [| fill |]; a; [| fill |] ] in
  let window = wrap window_shape 1 and stride = wrap strides 1 in
  let run x =
    let sh = (Core.get_aval x).shape in
    let pads = L.padtype_to_pads sh window stride padding in
    L.reduce_window x
      (L.full [||] init_value (Core.get_aval x).dtype)
      reducer ~window_dimensions:window ~window_strides:stride pads
  in
  {
    init =
      (fun _ input_shape ->
        let pads = L.padtype_to_pads input_shape window stride padding in
        let out =
          Array.mapi
            (fun i s ->
              let lo, hi = pads.(i) in
              ((s + lo + hi - window.(i)) / stride.(i)) + 1)
            input_shape
        in
        (out, Tree.Tuple []));
    apply =
      (fun ?rng:_ _ x ->
        let summed = run x in
        if not normalize then summed
        else
          let size = float_of_int (Array.fold_left ( * ) 1 window_shape) in
          L.div summed (L.full_like summed size));
  }

let max_pool ?strides ?padding window =
  pool_layer ~reducer:L.max ~init_value:Float.neg_infinity ?strides ?padding
    window

let sum_pool ?strides ?padding window =
  pool_layer ~reducer:L.add ~init_value:0.0 ?strides ?padding window

let avg_pool ?strides ?padding window =
  pool_layer ~reducer:L.add ~init_value:0.0 ~normalize:true ?strides ?padding
    window

let dropout ?(mode = `Train) rate =
  {
    init = (fun _ input_shape -> (input_shape, Tree.Tuple []));
    apply =
      (fun ?rng _ x ->
        match mode with
        | `Test -> x
        | `Train -> (
            match rng with
            | None ->
                invalid_arg
                  "dropout requires an rng in training mode; pass it to apply"
            | Some key ->
                let sh = (Core.get_aval x).shape in
                let keep = Random.Core.bernoulli key ~shape:sh ~p:rate in
                L.select keep (L.div x (L.full_like x rate)) (L.full_like x 0.0)
            ));
  }

let serial layers =
  {
    init =
      (fun rng input_shape ->
        let _, _, params_rev, out =
          List.fold_left
            (fun (key, shape, acc, _) layer ->
              let k1, k2 = split2 key in
              let out_shape, p = layer.init k1 shape in
              (k2, out_shape, p :: acc, out_shape))
            (rng, input_shape, [], input_shape)
            layers
        in
        (out, Tree.Tuple (List.rev params_rev)));
    apply =
      (fun ?rng params x ->
        match params with
        | Tree.Tuple ps ->
            List.fold_left2 (fun v layer p -> layer.apply ?rng p v) x layers ps
        | _ -> invalid_arg "serial: bad parameters");
  }

let batch_norm ?(axis = [| 0 |]) ?(epsilon = 1e-5) () =
  {
    init =
      (fun rng input_shape ->
        let n = Array.length input_shape in
        let axes = Array.map (fun a -> if a < 0 then a + n else a) axis in
        let kept =
          Array.of_list
            (List.filter (fun d -> not (Array.mem d axes)) (List.init n Fun.id))
        in
        let sh = Array.map (fun d -> input_shape.(d)) kept in
        ignore rng;
        let beta = L.full sh 0.0 (dtype_default ()) in
        let gamma = L.full sh 1.0 (dtype_default ()) in
        (input_shape, Tree.Tuple [ Tree.Leaf beta; Tree.Leaf gamma ]));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf beta; Tree.Leaf gamma ] ->
            let sh = (Core.get_aval x).shape in
            let n = Array.length sh in
            let axes = Array.map (fun a -> if a < 0 then a + n else a) axis in
            let kept =
              Array.of_list
                (List.filter
                   (fun d -> not (Array.mem d axes))
                   (List.init n Fun.id))
            in
            let z = NN.standardize ~axis:axes.(0) ~epsilon x in
            let lift v = L.broadcast_in_dim v sh kept in
            L.add (L.mul (lift gamma) z) (lift beta)
        | _ -> invalid_arg "batch_norm: bad parameters");
  }
