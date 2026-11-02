module T = Ojax.Types
module N = Ojax.Ndarray
module D = Ojax.Dtype
module JU = Ojax.Numpy.Jnp_util

exception No_binding of string

type json = Yojson.Safe.t

let no name = raise (No_binding name)

let member options name =
  match options with `Assoc l -> List.assoc_opt name l | _ -> None

let refuse options names =
  List.iter (fun n -> if member options n <> None then no ("option:" ^ n)) names

let get options name default =
  match member options name with Some j -> j | None -> default

let opt options name decode =
  match member options name with
  | None | Some `Null -> None
  | Some j -> Some (decode j)

let opt_keep options name decode =
  match member options name with None -> None | Some j -> Some (decode j)

let int_ : json -> int = function `Int n -> n | _ -> no "type:int"
let bool_ : json -> bool = function `Bool b -> b | _ -> no "type:bool"

let float_ : json -> float = function
  | `Int n -> float_of_int n
  | `Float f -> f
  | `String "inf" -> Float.infinity
  | `String "-inf" -> Float.neg_infinity
  | _ -> no "type:float"

let string_ : json -> string = function `String s -> s | _ -> no "type:string"

let elements : json -> json list = function
  | `List l -> l
  | `Assoc [ ("$tuple", `List l) ] -> l
  | _ -> no "type:sequence"

let int_list : json -> int list = function
  | `Null -> []
  | `Int n -> [ n ]
  | j -> List.map int_ (elements j)

let int_array j = Array.of_list (int_list j)

let int_sequence : json -> int list = function
  | `Int _ ->
      raise
        (Ojax.Py_exn.Type_error
           "option_args: a bare int given where the signature takes only a \
            sequence of ints")
  | j -> int_list j

let int_sequence_array j = Array.of_list (int_sequence j)
let float_array j = Array.of_list (List.map float_ (elements j))

let int_option : json -> int option = function
  | `Null -> None
  | j -> Some (int_ j)

let float_option : json -> float option = function
  | `Null -> None
  | j -> Some (float_ j)

let bool_option : json -> bool option = function
  | `Null -> None
  | j -> Some (bool_ j)

let string_option : json -> string option = function
  | `Null -> None
  | j -> Some (string_ j)

let int_array_option : json -> int array option = function
  | `Null -> None
  | j -> Some (int_array j)

let int_pair j =
  match elements j with [ a; b ] -> (int_ a, int_ b) | _ -> no "type:pair"

let int_pairs j = Array.of_list (List.map int_pair (elements j))

let string_pair j =
  match elements j with
  | [ a; b ] -> (string_ a, string_ b)
  | _ -> no "type:pair"

let trans : json -> int = function
  | `Int n -> n
  | `String s -> Ojax.Scipy.Linalg.trans_of_string s
  | _ -> no "type:trans"

let rounding_method : json -> Ojax.Lax_config.rounding_method = function
  | `Int 0 -> Ojax.Lax_config.Away_from_zero
  | `Int 1 -> Ojax.Lax_config.To_nearest_even
  | _ -> no "type:rounding_method"

let dtype : json -> D.t = function
  | `String s -> Ojax.Numpy.Extras.dtype s
  | _ -> no "type:dtype"

let precision : json -> Ojax.Lax_config.precision =
  let level = function
    | `String "default" -> Ojax.Lax_config.Precision_default
    | `String "high" -> Ojax.Lax_config.Precision_high
    | `String "highest" -> Ojax.Lax_config.Precision_highest
    | _ -> no "type:precision"
  in
  function
  | `String s when not (List.mem s [ "default"; "high"; "highest" ]) ->
      Ojax.Lax_config.precision_of_string s
  | `String _ as j ->
      let l = level j in
      Ojax.Lax_config.Precision (l, l)
  | j -> (
      match elements j with
      | [ a; b ] -> Ojax.Lax_config.Precision (level a, level b)
      | _ -> no "type:precision")

let default_float : json -> D.t = function
  | `Null -> Ojax.Dtypes.default_float_dtype ()
  | j -> dtype j

let default_int : json -> D.t = function
  | `Null -> Ojax.Dtypes.default_int_dtype ()
  | j -> dtype j

let padding : json -> Ojax.Lax_api.padding = function
  | `String "SAME" -> Ojax.Lax_api.Same
  | `String "VALID" -> Ojax.Lax_api.Valid
  | `String "SAME_LOWER" -> Ojax.Lax_api.Same_lower
  | `String s -> failwith ("Unknown padding " ^ s)
  | j -> Ojax.Lax_api.Explicit (int_pairs j)

let ord : json -> Ojax.Numpy.Linalg.ord = function
  | `Null -> Ojax.Numpy.Linalg.Onone
  | `String "fro" -> Ojax.Numpy.Linalg.Ofro
  | `String "nuc" -> Ojax.Numpy.Linalg.Onuc
  | j -> Ojax.Numpy.Linalg.Onum (float_ j)

let norm_axis : json -> Ojax.Numpy.Linalg.norm_axis = function
  | `Null -> Ojax.Numpy.Linalg.Anone
  | `Int n -> Ojax.Numpy.Linalg.Aint n
  | j ->
      let a, b = int_pair j in
      Ojax.Numpy.Linalg.Apair (a, b)

let td_axes : json -> Ojax.Numpy.Tensor_contractions.td_axes = function
  | `Int n -> Ojax.Numpy.Tensor_contractions.Ax_int n
  | j -> (
      match elements j with
      | [ a; b ] ->
          Ojax.Numpy.Tensor_contractions.Ax_pair (int_array a, int_array b)
      | _ -> no "type:td_axes")

let seed_key n =
  Ojax.Random.Core.key
    (T.Concrete
       (N.of_floats
          (Ojax.Dtypes.default_int_dtype ())
          [||]
          [| float_of_int n |]))

let value arr : json -> T.value = function
  | `Assoc [ ("$arr", `String name) ] -> arr name
  | `Assoc [ ("$key", `Int n) ] -> seed_key n
  | `Int n -> JU.py_int n
  | `Float f -> JU.py_float f
  | `Bool b ->
      T.Concrete (N.of_floats D.Bool [||] [| (if b then 1.0 else 0.0) |])
  | _ -> no "type:value"

let value_list arr : json -> T.value list = function
  | `Assoc [ (("$list" | "$star"), `List names) ] ->
      List.map (fun n -> arr (string_ n)) names
  | `Assoc [ ("$arr", `String name) ] -> [ arr name ]
  | `List items -> List.map (value arr) items
  | _ -> no "type:value_list"

let value_option arr : json -> T.value option = function
  | `Null -> None
  | j -> Some (value arr j)

let einsum_optimize : json -> Ojax.Numpy.Einsum_paths.optimize = function
  | `Bool true -> Ojax.Numpy.Einsum_paths.Optimal
  | `Bool false -> Ojax.Numpy.Einsum_paths.Unoptimized
  | `String s -> Ojax.Numpy.Einsum_paths.Named s
  | j -> Ojax.Numpy.Einsum_paths.Path (List.map int_list (elements j))

let axis_arg : json -> Ojax.Numpy.Reductions.axis_arg = function
  | `Int n -> Ojax.Numpy.Reductions.Axis n
  | j -> Ojax.Numpy.Reductions.Axes (int_array j)

let eig_implementation : json -> Ojax.Lax_config.eig_implementation = function
  | _ -> no "type:enum"

let eigh_implementation : json -> Ojax.Lax_config.eigh_implementation = function
  | _ -> no "type:enum"

let float_pair j =
  match elements j with [ a; b ] -> (float_ a, float_ b) | _ -> no "type:pair"

let sections arr : json -> Ojax.Numpy.Lax_numpy.sections = function
  | `Int n -> Ojax.Numpy.Lax_numpy.Count n
  | `Assoc [ ("$arr", _) ] as j ->
      Ojax.Numpy.Lax_numpy.Index_array (value arr j)
  | j -> Ojax.Numpy.Lax_numpy.Indices (int_array j)

let window_pair : json -> int * int = function
  | `Int n -> (n, n)
  | j -> int_pair j
