module A = Option_args

let run op positional options arr =
  ignore positional;
  ignore options;
  ignore arr;
  match op with
  | "jnp.all" ->
      [
        Ojax.Numpy.Reductions.all
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.allclose" ->
      [
        Ojax.Numpy.Lax_numpy.allclose
          ?rtol:(A.opt options "rtol" A.float_)
          ?atol:(A.opt options "atol" A.float_)
          ?equal_nan:(A.opt options "equal_nan" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.amax" ->
      [
        Ojax.Numpy.Reductions.amax
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.amin" ->
      [
        Ojax.Numpy.Reductions.amin
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.angle" ->
      [
        Ojax.Numpy.Lax_numpy.angle
          ?deg:(A.opt options "deg" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.any" ->
      [
        Ojax.Numpy.Reductions.any
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.append" ->
      [
        Ojax.Numpy.Lax_numpy.append
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.arange" ->
      [
        Ojax.Numpy.Lax_numpy.arange
          ?start:(Some (A.float_ positional.(0)))
          ?step:(A.opt options "step" A.float_)
          ~dtype:(A.dtype (A.get options "dtype" `Null))
          (A.float_ (A.get options "stop" `Null));
      ]
  | "jnp.argmax" ->
      [
        Ojax.Numpy.Lax_numpy.argmax
          ?axis:(A.opt options "axis" A.int_)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.argmin" ->
      [
        Ojax.Numpy.Lax_numpy.argmin
          ?axis:(A.opt options "axis" A.int_)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.argsort" ->
      [
        Ojax.Numpy.Sorting.argsort
          ?axis:(A.opt_keep options "axis" A.int_option)
          ?kind:(A.opt options "kind" A.string_)
          ?order:(A.opt options "order" A.string_)
          ?stable:(A.opt options "stable" A.bool_)
          ?descending:(A.opt options "descending" A.bool_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0));
      ]
  | "jnp.argwhere" ->
      [
        Ojax.Numpy.Extras.argwhere
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.around" ->
      [
        Ojax.Numpy.Lax_numpy.around
          ?decimals:(A.opt options "decimals" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.array_equal" ->
      [
        Ojax.Numpy.Lax_numpy.array_equal
          ?equal_nan:(A.opt options "equal_nan" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.array_split" ->
      Ojax.Numpy.Lax_numpy.array_split
        ?axis:(A.opt options "axis" A.int_)
        ((A.value arr) positional.(0))
        ((A.sections arr) positional.(1))
  | "jnp.asarray" ->
      [
        Ojax.Numpy.Array_constructors.asarray
          ?dtype:(A.opt options "dtype" A.dtype)
          ?order:(A.opt options "order" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.average" ->
      A.refuse options [ "returned" ];
      [
        Ojax.Numpy.Reductions.average
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.bincount" ->
      [
        Ojax.Numpy.Extras.bincount
          ?weights:(A.opt options "weights" (A.value arr))
          ?minlength:(A.opt options "minlength" A.int_)
          ?length:(A.opt options "length" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.choose" ->
      [
        Ojax.Numpy.Extras.choose
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value_list arr) positional.(1));
      ]
  | "jnp.clip" ->
      [
        Ojax.Numpy.Lax_numpy.clip
          ?min:(A.opt options "min" A.float_)
          ?max:(A.opt options "max" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.compress" ->
      [
        Ojax.Numpy.Extras.compress
          ?axis:(A.opt options "axis" A.int_)
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.concat" ->
      [
        Ojax.Numpy.Lax_numpy.concat
          ?axis:(A.opt options "axis" A.int_)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.concatenate" ->
      [
        Ojax.Numpy.Lax_numpy.concatenate
          ?axis:(A.opt options "axis" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.convolve" ->
      [
        Ojax.Numpy.Lax_numpy.convolve
          ?mode:(A.opt options "mode" A.string_)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.copy" ->
      [
        Ojax.Numpy.Lax_numpy.copy
          ?order:(A.opt options "order" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.corrcoef" ->
      [
        Ojax.Numpy.Lax_numpy.corrcoef
          ?y:(A.opt options "y" (A.value arr))
          ?rowvar:(A.opt options "rowvar" A.bool_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0));
      ]
  | "jnp.correlate" ->
      [
        Ojax.Numpy.Lax_numpy.correlate
          ?mode:(A.opt options "mode" A.string_)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.count_nonzero" ->
      [
        Ojax.Numpy.Reductions.count_nonzero
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.cov" ->
      [
        Ojax.Numpy.Lax_numpy.cov
          ?y:(A.opt options "y" (A.value arr))
          ?rowvar:(A.opt options "rowvar" A.bool_)
          ?bias:(A.opt options "bias" A.bool_)
          ?ddof:(A.opt options "ddof" A.int_)
          ?fweights:(A.opt options "fweights" (A.value arr))
          ?aweights:(A.opt options "aweights" (A.value arr))
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0));
      ]
  | "jnp.cross" ->
      [
        Ojax.Numpy.Lax_numpy.cross
          ?axisa:(A.opt options "axisa" A.int_)
          ?axisb:(A.opt options "axisb" A.int_)
          ?axisc:(A.opt options "axisc" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.cumprod" ->
      [
        Ojax.Numpy.Reductions.cumprod
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.cumsum" ->
      [
        Ojax.Numpy.Reductions.cumsum
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.cumulative_prod" ->
      [
        Ojax.Numpy.Reductions.cumulative_prod
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ?include_initial:(A.opt options "include_initial" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.cumulative_sum" ->
      [
        Ojax.Numpy.Reductions.cumulative_sum
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ?include_initial:(A.opt options "include_initial" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.delete" ->
      [
        Ojax.Numpy.Extras.delete_array
          ?axis:(A.opt options "axis" A.int_)
          ?assume_unique_indices:(A.opt options "assume_unique_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.diag" ->
      [
        Ojax.Numpy.Lax_numpy.diag ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.diag_indices" ->
      Ojax.Numpy.Lax_numpy.diag_indices
        ?ndim:(A.opt options "ndim" A.int_)
        (A.int_ positional.(0))
  | "jnp.diagflat" ->
      [
        Ojax.Numpy.Lax_numpy.diagflat ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.diagonal" ->
      [
        Ojax.Numpy.Lax_numpy.diagonal
          ?offset:(A.opt options "offset" A.int_)
          ?axis1:(A.opt options "axis1" A.int_)
          ?axis2:(A.opt options "axis2" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.diff" ->
      [
        Ojax.Numpy.Lax_numpy.diff ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?prepend:(A.opt options "prepend" (A.value arr))
          ?append:(A.opt options "append" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.digitize" ->
      [
        Ojax.Numpy.Lax_numpy.digitize
          ?right:(A.opt options "right" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.dot" ->
      [
        Ojax.Numpy.Tensor_contractions.dot
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.dstack" ->
      [
        Ojax.Numpy.Lax_numpy.dstack
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.ediff1d" ->
      [
        Ojax.Numpy.Lax_numpy.ediff1d
          ?to_end:(A.opt options "to_end" (A.value arr))
          ?to_begin:(A.opt options "to_begin" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.einsum" ->
      [
        Ojax.Numpy.Einsum.einsum
          ?optimize:(A.opt options "optimize" A.einsum_optimize)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          (A.string_ positional.(0))
          ((A.value_list arr) (A.get options "out" `Null));
      ]
  | "jnp.empty" ->
      [
        Ojax.Numpy.Array_creation.empty
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_array positional.(0));
      ]
  | "jnp.empty_like" ->
      [
        Ojax.Numpy.Array_creation.empty_like
          ?dtype:(A.opt options "dtype" A.dtype)
          ?shape:(A.opt options "shape" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "jnp.extract" ->
      [
        Ojax.Numpy.Extras.extract
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.eye" ->
      [
        Ojax.Numpy.Lax_numpy.eye ?m:(A.opt options "M" A.int_)
          ?k:(A.opt options "k" A.int_)
          ~dtype:(A.default_float (A.get options "dtype" `Null))
          (A.int_ positional.(0));
      ]
  | "jnp.fill_diagonal" ->
      [
        Ojax.Numpy.Extras.fill_diagonal
          ?wrap:(A.opt options "wrap" A.bool_)
          ?inplace:(A.opt options "inplace" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.flatnonzero" ->
      [
        Ojax.Numpy.Extras.flatnonzero
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.flip" ->
      [
        Ojax.Numpy.Lax_numpy.flip
          ?axis:(A.opt options "axis" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "jnp.full" ->
      [
        Ojax.Numpy.Array_creation.full
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_array positional.(0))
          (A.float_ positional.(1));
      ]
  | "jnp.full_like" ->
      [
        Ojax.Numpy.Array_creation.full_like
          ?dtype:(A.opt options "dtype" A.dtype)
          ?shape:(A.opt options "shape" A.int_array)
          ((A.value arr) positional.(0))
          (A.float_ positional.(1));
      ]
  | "jnp.geomspace" ->
      [
        Ojax.Numpy.Array_creation.geomspace_v
          ?num:(A.opt options "num" A.int_)
          ?endpoint:(A.opt options "endpoint" A.bool_)
          ?axis:(A.opt options "axis" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.gradient" ->
      Ojax.Numpy.Extras.gradient
        ?axis:(A.opt options "axis" A.int_list)
        ?edge_order:(A.opt options "edge_order" A.int_)
        ((A.value arr) positional.(0))
  | "jnp.histogram" ->
      let a, b =
        Ojax.Numpy.Extras.histogram
          ?bins:(A.opt options "bins" A.int_)
          ?range:(A.opt options "range" A.float_pair)
          ?weights:(A.opt options "weights" (A.value arr))
          ?density:(A.opt options "density" A.bool_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "jnp.histogram2d" ->
      let a, b, c =
        Ojax.Numpy.Extras.histogram2d
          ?bins:(A.opt options "bins" A.int_array)
          ?weights:(A.opt options "weights" (A.value arr))
          ?density:(A.opt options "density" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
      in
      [ a; b; c ]
  | "jnp.histogram_bin_edges" ->
      [
        Ojax.Numpy.Extras.histogram_bin_edges
          ?bins:(A.opt options "bins" A.int_)
          ?range:(A.opt options "range" A.float_pair)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.hstack" ->
      [
        Ojax.Numpy.Lax_numpy.hstack
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.identity" ->
      [
        Ojax.Numpy.Lax_numpy.identity
          ~dtype:(A.default_float (A.get options "dtype" `Null))
          (A.int_ positional.(0));
      ]
  | "jnp.indices" ->
      A.refuse options [ "sparse" ];
      [
        Ojax.Numpy.Lax_numpy.indices
          ~dtype:(A.default_int (A.get options "dtype" `Null))
          (A.int_array positional.(0));
      ]
  | "jnp.inner" ->
      [
        Ojax.Numpy.Tensor_contractions.inner
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.insert" ->
      [
        Ojax.Numpy.Extras.insert
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.interp" ->
      [
        Ojax.Numpy.Extras.interp
          ?left:(A.opt options "left" A.float_)
          ?right:(A.opt options "right" A.float_)
          ?period:(A.opt options "period" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.intersect1d" ->
      A.refuse options [ "return_indices" ];
      [
        Ojax.Numpy.Extras.intersect1d
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.isclose" ->
      [
        Ojax.Numpy.Lax_numpy.isclose
          ?rtol:(A.opt options "rtol" A.float_)
          ?atol:(A.opt options "atol" A.float_)
          ?equal_nan:(A.opt options "equal_nan" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.isin" ->
      [
        Ojax.Numpy.Setops.isin
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          ?invert:(A.opt options "invert" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.lexsort" ->
      [
        Ojax.Numpy.Sorting.lexsort
          ?axis:(A.opt options "axis" A.int_)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.linspace" ->
      A.refuse options [ "retstep" ];
      [
        Ojax.Numpy.Array_creation.linspace_v
          ?num:(A.opt options "num" A.int_)
          ?endpoint:(A.opt options "endpoint" A.bool_)
          ?axis:(A.opt options "axis" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.logspace" ->
      [
        Ojax.Numpy.Array_creation.logspace_v
          ?num:(A.opt options "num" A.int_)
          ?endpoint:(A.opt options "endpoint" A.bool_)
          ?base:(A.opt options "base" A.float_)
          ?axis:(A.opt options "axis" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.matmul" ->
      [
        Ojax.Numpy.Tensor_contractions.matmul
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.max" ->
      [
        Ojax.Numpy.Reductions.max
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.mean" ->
      [
        Ojax.Numpy.Reductions.mean
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.median" ->
      [
        Ojax.Numpy.Reductions.median
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.meshgrid" ->
      Ojax.Numpy.Lax_numpy.meshgrid
        ?indexing:(A.opt options "indexing" A.string_)
        ?sparse:(A.opt options "sparse" A.bool_)
        ((A.value_list arr) positional.(0))
  | "jnp.min" ->
      [
        Ojax.Numpy.Reductions.min
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nan_to_num" ->
      [
        Ojax.Numpy.Lax_numpy.nan_to_num
          ?nan:(A.opt options "nan" A.float_)
          ?posinf:(A.opt options "posinf" A.float_)
          ?neginf:(A.opt options "neginf" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanargmax" ->
      [
        Ojax.Numpy.Lax_numpy.nanargmax
          ?axis:(A.opt options "axis" A.int_)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanargmin" ->
      [
        Ojax.Numpy.Lax_numpy.nanargmin
          ?axis:(A.opt options "axis" A.int_)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nancumprod" ->
      [
        Ojax.Numpy.Reductions.nancumprod
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nancumsum" ->
      [
        Ojax.Numpy.Reductions.nancumsum
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanmax" ->
      [
        Ojax.Numpy.Reductions.nanmax
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanmean" ->
      [
        Ojax.Numpy.Reductions.nanmean
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanmedian" ->
      [
        Ojax.Numpy.Reductions.nanmedian
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanmin" ->
      [
        Ojax.Numpy.Reductions.nanmin
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanpercentile" ->
      [
        Ojax.Numpy.Reductions.nanpercentile
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.nanprod" ->
      [
        Ojax.Numpy.Reductions.nanprod
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanquantile" ->
      [
        Ojax.Numpy.Reductions.nanquantile
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.nanstd" ->
      [
        Ojax.Numpy.Reductions.nanstd
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ddof:(A.opt options "ddof" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ?mean:(A.opt options "mean" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nansum" ->
      [
        Ojax.Numpy.Reductions.nansum
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nanvar" ->
      [
        Ojax.Numpy.Reductions.nanvar
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ddof:(A.opt options "ddof" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ?mean:(A.opt options "mean" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.nonzero" ->
      Ojax.Numpy.Extras.nonzero
        ?size:(A.opt options "size" A.int_)
        ?fill_value:(A.opt options "fill_value" (A.value arr))
        ((A.value arr) positional.(0))
  | "jnp.ones" ->
      [
        Ojax.Numpy.Array_creation.ones
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_array positional.(0));
      ]
  | "jnp.ones_like" ->
      [
        Ojax.Numpy.Array_creation.ones_like
          ?dtype:(A.opt options "dtype" A.dtype)
          ?shape:(A.opt options "shape" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "jnp.packbits" ->
      [
        Ojax.Numpy.Extras.packbits
          ?axis:(A.opt options "axis" A.int_)
          ?bitorder:(A.opt options "bitorder" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.percentile" ->
      [
        Ojax.Numpy.Reductions.percentile
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.place" ->
      [
        Ojax.Numpy.Extras.place
          ?inplace:(A.opt options "inplace" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.polyder" ->
      [
        Ojax.Numpy.Polynomial.polyder ?m:(A.opt options "m" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.polydiv" ->
      let a, b =
        Ojax.Numpy.Extras.polydiv
          ?trim_leading_zeros:(A.opt options "trim_leading_zeros" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
      in
      [ a; b ]
  | "jnp.polyfit" ->
      A.refuse options [ "cov"; "full" ];
      [
        Ojax.Numpy.Extras.polyfit
          ?rcond:(A.opt options "rcond" A.float_)
          ?w:(A.opt options "w" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.int_ positional.(2));
      ]
  | "jnp.polyint" ->
      [
        Ojax.Numpy.Polynomial.polyint ?m:(A.opt options "m" A.int_)
          ?k:(A.opt options "k" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "jnp.polymul" ->
      [
        Ojax.Numpy.Polynomial.polymul
          ?trim_leading_zeros:(A.opt options "trim_leading_zeros" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.polyval" ->
      [
        Ojax.Numpy.Polynomial.polyval
          ?unroll:(A.opt options "unroll" A.int_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.prod" ->
      [
        Ojax.Numpy.Reductions.prod
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ?promote_integers:(A.opt options "promote_integers" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.ptp" ->
      [
        Ojax.Numpy.Reductions.ptp
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.put" ->
      [
        Ojax.Numpy.Indexing.put
          ?mode:(A.opt options "mode" A.string_)
          ?inplace:(A.opt options "inplace" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.put_along_axis" ->
      [
        Ojax.Numpy.Indexing.put_along_axis
          ?axis:(Some (A.int_ positional.(3)))
          ?mode:(A.opt options "mode" A.string_)
          ?inplace:(A.opt options "inplace" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.quantile" ->
      [
        Ojax.Numpy.Reductions.quantile
          ?axis:(A.opt options "axis" A.axis_arg)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?method_:(A.opt options "method" A.string_)
          ?weights:(A.opt options "weights" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.ravel" ->
      [
        Ojax.Numpy.Lax_numpy.ravel
          ?order:(A.opt options "order" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.ravel_multi_index" ->
      [
        Ojax.Numpy.Extras.ravel_multi_index
          ?mode:(A.opt options "mode" A.string_)
          ?order:(A.opt options "order" A.string_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0))
          (A.int_array positional.(1));
      ]
  | "jnp.repeat" ->
      [
        Ojax.Numpy.Lax_numpy.repeat
          ?axis:(A.opt options "axis" A.int_)
          ?total_repeat_length:(A.opt options "total_repeat_length" A.int_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1));
      ]
  | "jnp.reshape" ->
      [
        Ojax.Numpy.Lax_numpy.reshape
          ?order:(A.opt options "order" A.string_)
          ((A.value arr) positional.(0))
          (A.int_array positional.(1));
      ]
  | "jnp.roll" ->
      [
        Ojax.Numpy.Lax_numpy.roll
          ?axis:(A.opt options "axis" A.int_array)
          ((A.value arr) positional.(0))
          (A.int_array positional.(1));
      ]
  | "jnp.rollaxis" ->
      [
        Ojax.Numpy.Lax_numpy.rollaxis
          ?start:(A.opt options "start" A.int_)
          (A.int_ positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.roots" ->
      [
        Ojax.Numpy.Extras.roots
          ?strip_zeros:(A.opt options "strip_zeros" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.rot90" ->
      [
        Ojax.Numpy.Lax_numpy.rot90 ?k:(A.opt options "k" A.int_)
          ?axes:(A.opt options "axes" A.int_pair)
          ((A.value arr) positional.(0));
      ]
  | "jnp.round" ->
      [
        Ojax.Numpy.Lax_numpy.round
          ?decimals:(A.opt options "decimals" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.searchsorted" ->
      [
        Ojax.Numpy.Lax_numpy.searchsorted
          ?side:(A.opt options "side" A.string_)
          ?sorter:(A.opt options "sorter" (A.value arr))
          ?method_:(A.opt options "method" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.select" ->
      [
        Ojax.Numpy.Lax_numpy.select
          ?default:(A.opt options "default" (A.value arr))
          ((A.value_list arr) positional.(0))
          ((A.value_list arr) positional.(1));
      ]
  | "jnp.setdiff1d" ->
      [
        Ojax.Numpy.Extras.setdiff1d
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.setxor1d" ->
      [
        Ojax.Numpy.Extras.setxor1d
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.sort" ->
      [
        Ojax.Numpy.Sorting.sort
          ?axis:(A.opt_keep options "axis" A.int_option)
          ?kind:(A.opt options "kind" A.string_)
          ?order:(A.opt options "order" A.string_)
          ?stable:(A.opt options "stable" A.bool_)
          ?descending:(A.opt options "descending" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.split" ->
      Ojax.Numpy.Lax_numpy.split
        ?axis:(A.opt options "axis" A.int_)
        ((A.value arr) positional.(0))
        ((A.sections arr) positional.(1))
  | "jnp.squeeze" ->
      [
        Ojax.Numpy.Lax_numpy.squeeze
          ?axis:(A.opt options "axis" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "jnp.stack" ->
      [
        Ojax.Numpy.Lax_numpy.stack
          ?axis:(A.opt options "axis" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.std" ->
      [
        Ojax.Numpy.Reductions.std
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ddof:(A.opt options "ddof" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ?mean:(A.opt options "mean" (A.value arr))
          ?correction:(A.opt options "correction" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.sum" ->
      [
        Ojax.Numpy.Reductions.sum
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?initial:(A.opt options "initial" (A.value arr))
          ?where:(A.opt options "where" (A.value arr))
          ?promote_integers:(A.opt options "promote_integers" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.take" ->
      [
        Ojax.Numpy.Indexing.take
          ?axis:(A.opt options "axis" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ?unique_indices:(A.opt options "unique_indices" A.bool_)
          ?indices_are_sorted:(A.opt options "indices_are_sorted" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.take_along_axis" ->
      [
        Ojax.Numpy.Indexing.take_along_axis
          ?axis:(A.opt options "axis" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ?wrap_negative_indices:(A.opt options "wrap_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.tensordot" ->
      [
        Ojax.Numpy.Tensor_contractions.tensordot
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ?axes:(A.opt options "axes" A.td_axes)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.trace" ->
      [
        Ojax.Numpy.Lax_numpy.trace
          ?offset:(A.opt options "offset" A.int_)
          ?axis1:(A.opt options "axis1" A.int_)
          ?axis2:(A.opt options "axis2" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0));
      ]
  | "jnp.transpose" ->
      [
        Ojax.Numpy.Lax_numpy.transpose
          ?axes:(A.opt options "axes" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "jnp.trapezoid" ->
      [
        Ojax.Numpy.Lax_numpy.trapezoid
          ?x:(A.opt options "x" (A.value arr))
          ?dx:(A.opt options "dx" A.float_)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.tri" ->
      [
        Ojax.Numpy.Lax_numpy.tri ?m:(A.opt options "M" A.int_)
          ?k:(A.opt options "k" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_ positional.(0));
      ]
  | "jnp.tril" ->
      [
        Ojax.Numpy.Lax_numpy.tril ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.tril_indices" ->
      let a, b =
        Ojax.Numpy.Extras.tril_indices ?k:(A.opt options "k" A.int_)
          ?m:(A.opt options "m" A.int_)
          (A.int_ positional.(0))
      in
      [ a; b ]
  | "jnp.tril_indices_from" ->
      let a, b =
        Ojax.Numpy.Extras.tril_indices_from ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "jnp.trim_zeros" ->
      [
        Ojax.Numpy.Extras.trim_zeros
          ?trim:(A.opt options "trim" A.string_)
          ?axis:(A.opt options "axis" A.int_list)
          ((A.value arr) positional.(0));
      ]
  | "jnp.triu" ->
      [
        Ojax.Numpy.Lax_numpy.triu ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.triu_indices" ->
      let a, b =
        Ojax.Numpy.Extras.triu_indices ?k:(A.opt options "k" A.int_)
          ?m:(A.opt options "m" A.int_)
          (A.int_ positional.(0))
      in
      [ a; b ]
  | "jnp.triu_indices_from" ->
      let a, b =
        Ojax.Numpy.Extras.triu_indices_from ?k:(A.opt options "k" A.int_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "jnp.union1d" ->
      [
        Ojax.Numpy.Extras.union1d
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.unique" ->
      Ojax.Numpy.Extras.unique
        ?return_index:(A.opt options "return_index" A.bool_)
        ?return_inverse:(A.opt options "return_inverse" A.bool_)
        ?return_counts:(A.opt options "return_counts" A.bool_)
        ?axis:(A.opt options "axis" A.int_)
        ?equal_nan:(A.opt options "equal_nan" A.bool_)
        ?size:(A.opt options "size" A.int_)
        ?fill_value:(A.opt options "fill_value" A.float_)
        ?sorted:(A.opt options "sorted" A.bool_)
        ((A.value arr) positional.(0))
  | "jnp.unique_counts" ->
      let a, b =
        Ojax.Numpy.Extras.unique_counts
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "jnp.unique_inverse" ->
      let a, b =
        Ojax.Numpy.Extras.unique_inverse
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "jnp.unique_values" ->
      [
        Ojax.Numpy.Extras.unique_values
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.unpackbits" ->
      [
        Ojax.Numpy.Extras.unpackbits
          ?axis:(A.opt options "axis" A.int_)
          ?count:(A.opt options "count" A.int_)
          ?bitorder:(A.opt options "bitorder" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.unstack" ->
      Ojax.Numpy.Lax_numpy.unstack
        ?axis:(A.opt options "axis" A.int_)
        ((A.value arr) positional.(0))
  | "jnp.unwrap" ->
      [
        Ojax.Numpy.Lax_numpy.unwrap
          ?discont:(A.opt options "discont" A.float_)
          ?axis:(A.opt options "axis" A.int_)
          ?period:(A.opt options "period" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.vander" ->
      [
        Ojax.Numpy.Lax_numpy.vander ?n:(A.opt options "N" A.int_)
          ?increasing:(A.opt options "increasing" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.var" ->
      [
        Ojax.Numpy.Reductions.var
          ?dtype:(A.opt options "dtype" A.dtype)
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ddof:(A.opt options "ddof" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ?mean:(A.opt options "mean" (A.value arr))
          ?correction:(A.opt options "correction" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "jnp.vdot" ->
      [
        Ojax.Numpy.Tensor_contractions.vdot
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.vecdot" ->
      [
        Ojax.Numpy.Tensor_contractions.vecdot
          ?axis:(A.opt options "axis" A.int_)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "jnp.vstack" ->
      [
        Ojax.Numpy.Lax_numpy.vstack
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value_list arr) positional.(0));
      ]
  | "jnp.where" ->
      [
        Ojax.Numpy.Lax_numpy.where_
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "jnp.zeros" ->
      [
        Ojax.Numpy.Array_creation.zeros
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_array positional.(0));
      ]
  | "jnp.zeros_like" ->
      [
        Ojax.Numpy.Array_creation.zeros_like
          ?dtype:(A.opt options "dtype" A.dtype)
          ?shape:(A.opt options "shape" A.int_array)
          ((A.value arr) positional.(0));
      ]
  | "linalg.cholesky" ->
      [
        Ojax.Numpy.Linalg.cholesky
          ?upper:(A.opt options "upper" A.bool_)
          ?symmetrize_input:(A.opt options "symmetrize_input" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "linalg.cross" ->
      [
        Ojax.Numpy.Linalg.cross
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "linalg.diagonal" ->
      [
        Ojax.Numpy.Linalg.diagonal
          ?offset:(A.opt options "offset" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "linalg.eigh" ->
      let a, b =
        Ojax.Numpy.Linalg.eigh
          ?uplo:(A.opt options "UPLO" A.string_)
          ?symmetrize_input:(A.opt options "symmetrize_input" A.bool_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "linalg.eigvalsh" ->
      [
        Ojax.Numpy.Linalg.eigvalsh
          ?uplo:(A.opt options "UPLO" A.string_)
          ?symmetrize_input:(A.opt options "symmetrize_input" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "linalg.lstsq" ->
      let a, b, c, d =
        Ojax.Numpy.Linalg.lstsq
          ?rcond:(A.opt options "rcond" A.float_)
          ?numpy_resid:(A.opt options "numpy_resid" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
      in
      [ a; b; c; d ]
  | "linalg.matmul" ->
      [
        Ojax.Numpy.Linalg.matmul
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "linalg.matrix_norm" ->
      [
        Ojax.Numpy.Linalg.matrix_norm
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ord:(A.opt options "ord" A.ord)
          ((A.value arr) positional.(0));
      ]
  | "linalg.matrix_rank" ->
      [
        Ojax.Numpy.Linalg.matrix_rank
          ?rtol:(A.opt options "rtol" (A.value arr))
          ?hermitian:(A.opt options "hermitian" A.bool_)
          ?tol:(A.opt options "tol" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "linalg.multi_dot" ->
      [
        Ojax.Numpy.Linalg.multi_dot
          ?precision:(A.opt options "precision" A.precision)
          ((A.value_list arr) positional.(0));
      ]
  | "linalg.norm" ->
      [
        Ojax.Numpy.Linalg.norm
          ?ord:(A.opt options "ord" A.ord)
          ?axis:(A.opt options "axis" A.norm_axis)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "linalg.pinv" ->
      [
        Ojax.Numpy.Linalg.pinv
          ?rtol:(A.opt options "rtol" (A.value arr))
          ?hermitian:(A.opt options "hermitian" A.bool_)
          ?rcond:(A.opt options "rcond" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "linalg.qr" ->
      Ojax.Numpy.Linalg.qr
        ?mode:(A.opt options "mode" A.string_)
        ((A.value arr) positional.(0))
  | "linalg.svd" ->
      Ojax.Numpy.Linalg.svd
        ?full_matrices:(A.opt options "full_matrices" A.bool_)
        ?compute_uv:(A.opt options "compute_uv" A.bool_)
        ?hermitian:(A.opt options "hermitian" A.bool_)
        ?subset_by_index:(A.opt options "subset_by_index" A.int_pair)
        ((A.value arr) positional.(0))
  | "linalg.tensordot" ->
      [
        Ojax.Numpy.Linalg.tensordot
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ?axes:(A.opt options "axes" A.td_axes)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "linalg.tensorinv" ->
      [
        Ojax.Numpy.Linalg.tensorinv
          ?ind:(A.opt options "ind" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "linalg.tensorsolve" ->
      [
        Ojax.Numpy.Linalg.tensorsolve
          ?axes:(A.opt options "axes" A.int_array)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "linalg.trace" ->
      [
        Ojax.Numpy.Linalg.trace
          ?offset:(A.opt options "offset" A.int_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ((A.value arr) positional.(0));
      ]
  | "linalg.vecdot" ->
      [
        Ojax.Numpy.Linalg.vecdot
          ?axis:(A.opt options "axis" A.int_)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred:(A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "linalg.vector_norm" ->
      [
        Ojax.Numpy.Linalg.vector_norm
          ?axis:(A.opt_keep options "axis" A.int_array_option)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?ord:(A.opt options "ord" A.ord)
          ((A.value arr) positional.(0));
      ]
  | "fft.fft" ->
      [
        Ojax.Numpy.Fft.fft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.fft2" ->
      [
        Ojax.Numpy.Fft.fft2
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.fftfreq" ->
      [
        Ojax.Numpy.Fft.fftfreq
          ?d:(A.opt options "d" A.float_)
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_ positional.(0));
      ]
  | "fft.fftn" ->
      [
        Ojax.Numpy.Fft.fftn
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.fftshift" ->
      [
        Ojax.Numpy.Fft.fftshift
          ?axes:(A.opt options "axes" A.int_list)
          ((A.value arr) positional.(0));
      ]
  | "fft.hfft" ->
      [
        Ojax.Numpy.Fft.hfft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.ifft" ->
      [
        Ojax.Numpy.Fft.ifft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.ifft2" ->
      [
        Ojax.Numpy.Fft.ifft2
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.ifftn" ->
      [
        Ojax.Numpy.Fft.ifftn
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.ifftshift" ->
      [
        Ojax.Numpy.Fft.ifftshift
          ?axes:(A.opt options "axes" A.int_list)
          ((A.value arr) positional.(0));
      ]
  | "fft.ihfft" ->
      [
        Ojax.Numpy.Fft.ihfft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.irfft" ->
      [
        Ojax.Numpy.Fft.irfft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.irfft2" ->
      [
        Ojax.Numpy.Fft.irfft2
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.irfftn" ->
      [
        Ojax.Numpy.Fft.irfftn
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.rfft" ->
      [
        Ojax.Numpy.Fft.rfft ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.rfft2" ->
      [
        Ojax.Numpy.Fft.rfft2
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "fft.rfftfreq" ->
      [
        Ojax.Numpy.Fft.rfftfreq
          ?d:(A.opt options "d" A.float_)
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_ positional.(0));
      ]
  | "fft.rfftn" ->
      [
        Ojax.Numpy.Fft.rfftn
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "lax.approx_max_k" ->
      let a, b =
        Ojax.Lax_extras.approx_max_k
          ?reduction_dimension:(A.opt options "reduction_dimension" A.int_)
          ?recall_target:(A.opt options "recall_target" A.float_)
          ?reduction_input_size_override:
            (A.opt options "reduction_input_size_override" A.int_)
          ?aggregate_to_topk:(A.opt options "aggregate_to_topk" A.bool_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
      in
      [ a; b ]
  | "lax.approx_min_k" ->
      let a, b =
        Ojax.Lax_extras.approx_min_k
          ?reduction_dimension:(A.opt options "reduction_dimension" A.int_)
          ?recall_target:(A.opt options "recall_target" A.float_)
          ?reduction_input_size_override:
            (A.opt options "reduction_input_size_override" A.int_)
          ?aggregate_to_topk:(A.opt options "aggregate_to_topk" A.bool_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
      in
      [ a; b ]
  | "lax.batch_matmul" ->
      [
        Ojax.Lax_api.batch_matmul
          ?precision:(A.opt options "precision" A.precision)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax.collapse" ->
      [
        Ojax.Lax_api.collapse
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
          (A.int_ (A.get options "stop_dimension" `Null));
      ]
  | "lax.conv" ->
      [
        Ojax.Lax_api.conv
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ~window_strides:(A.int_array positional.(2))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.padding positional.(3));
      ]
  | "lax.conv_general_dilated" ->
      [
        Ojax.Lax_api.conv_general_dilated
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ?lhs_dilation:(A.opt options "lhs_dilation" A.int_array)
          ?rhs_dilation:(A.opt options "rhs_dilation" A.int_array)
          ?feature_group_count:(A.opt options "feature_group_count" A.int_)
          ?batch_group_count:(A.opt options "batch_group_count" A.int_)
          ~window_strides:(A.int_array positional.(2))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.padding positional.(3));
      ]
  | "lax.conv_general_dilated_local" ->
      [
        Ojax.Lax_extras.conv_general_dilated_local
          ?window_strides:(Some (A.int_array positional.(2)))
          ?padding:(Some (A.padding positional.(3)))
          ?lhs_dilation:(A.opt options "lhs_dilation" A.int_array)
          ?rhs_dilation:(A.opt options "rhs_dilation" A.int_array)
          ?precision:(A.opt options "precision" A.precision)
          ~filter_shape:(A.int_array positional.(4))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax.conv_general_dilated_patches" ->
      [
        Ojax.Lax_api.conv_general_dilated_patches
          ?window_strides:(Some (A.int_array positional.(2)))
          ?padding:(Some (A.padding positional.(3)))
          ?lhs_dilation:(A.opt options "lhs_dilation" A.int_array)
          ?rhs_dilation:(A.opt options "rhs_dilation" A.int_array)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ~filter_shape:(A.int_array positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "lax.conv_transpose" ->
      [
        Ojax.Lax_api.conv_transpose
          ?rhs_dilation:(A.opt options "rhs_dilation" A.int_array)
          ?transpose_kernel:(A.opt options "transpose_kernel" A.bool_)
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ?use_consistent_padding:
            (A.opt options "use_consistent_padding" A.bool_)
          ~strides:(A.int_array positional.(2))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.padding positional.(3));
      ]
  | "lax.conv_with_general_padding" ->
      [
        Ojax.Lax_api.conv_with_general_padding
          ?lhs_dilation:(A.int_array_option positional.(4))
          ?rhs_dilation:(A.int_array_option positional.(5))
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ~window_strides:(A.int_array positional.(2))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.int_pairs positional.(3));
      ]
  | "lax.cumlogsumexp" ->
      [
        Ojax.Lax_api.cumlogsumexp
          ?axis:(A.opt options "axis" A.int_)
          ?reverse:(A.opt options "reverse" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax.cummax" ->
      [
        Ojax.Lax_api.cummax
          ?axis:(A.opt options "axis" A.int_)
          ?reverse:(A.opt options "reverse" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax.cummin" ->
      [
        Ojax.Lax_api.cummin
          ?axis:(A.opt options "axis" A.int_)
          ?reverse:(A.opt options "reverse" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax.cumprod" ->
      [
        Ojax.Lax_api.cumprod
          ?axis:(A.opt options "axis" A.int_)
          ?reverse:(A.opt options "reverse" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax.cumsum" ->
      [
        Ojax.Lax_api.cumsum
          ?axis:(A.opt options "axis" A.int_)
          ?reverse:(A.opt options "reverse" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax.dot" ->
      [
        Ojax.Lax_api.dot
          ?precision:(A.opt options "precision" A.precision)
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax.dynamic_index_in_dim" ->
      [
        Ojax.Lax_api.dynamic_index_in_dim
          ?axis:(A.opt options "axis" A.int_)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax.dynamic_slice" ->
      [
        Ojax.Lax_api.dynamic_slice
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value_list arr) positional.(1))
          (A.int_array positional.(2));
      ]
  | "lax.dynamic_slice_in_dim" ->
      [
        Ojax.Lax_api.dynamic_slice_in_dim
          ?axis:(A.opt options "axis" A.int_)
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          (A.int_ positional.(2));
      ]
  | "lax.dynamic_update_index_in_dim" ->
      [
        Ojax.Lax_api.dynamic_update_index_in_dim
          ?axis:(Some (A.int_ positional.(3)))
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "lax.dynamic_update_slice" ->
      [
        Ojax.Lax_api.dynamic_update_slice
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value_list arr) positional.(2));
      ]
  | "lax.dynamic_update_slice_in_dim" ->
      [
        Ojax.Lax_api.dynamic_update_slice_in_dim
          ?axis:(Some (A.int_ positional.(3)))
          ?allow_negative_indices:
            (A.opt options "allow_negative_indices" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "lax.full" ->
      [
        Ojax.Lax_api.full
          (A.int_array positional.(0))
          (A.float_ positional.(1))
          (A.dtype (A.get options "dtype" `Null));
      ]
  | "lax.full_like" ->
      [
        Ojax.Lax_api.full_like
          ?dtype:(A.opt options "dtype" A.dtype)
          ?shape:(A.opt options "shape" A.int_array)
          ((A.value arr) positional.(0))
          (A.float_ positional.(1));
      ]
  | "lax.mul" ->
      [
        Ojax.Lax_api.mul_with
          ?out_dtype:(A.opt options "out_dtype" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax.reshape" ->
      [
        Ojax.Lax_api.reshape
          ?dimensions:(A.opt options "dimensions" A.int_array)
          ((A.value arr) positional.(0))
          (A.int_array positional.(1));
      ]
  | "lax.round" ->
      [
        Ojax.Lax_api.round
          ?rounding_method:(A.opt options "rounding_method" A.rounding_method)
          ((A.value arr) positional.(0));
      ]
  | "lax.slice" ->
      [
        Ojax.Lax_api.slice
          ?strides:(A.opt options "strides" A.int_array)
          ((A.value arr) positional.(0))
          (A.int_array positional.(1))
          (A.int_array positional.(2));
      ]
  | "lax.slice_in_dim" ->
      [
        Ojax.Lax_api.slice_in_dim
          ?stride:(A.opt options "stride" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
          (A.int_ positional.(2));
      ]
  | "lax.sort" ->
      Ojax.Lax_api.sort
        ?dimension:(A.opt options "dimension" A.int_)
        ?is_stable:(A.opt options "is_stable" A.bool_)
        ?num_keys:(A.opt options "num_keys" A.int_)
        ((A.value_list arr) positional.(0))
  | "lax.sort_key_val" ->
      let a, b =
        Ojax.Lax_api.sort_key_val
          ?dimension:(A.opt options "dimension" A.int_)
          ?is_stable:(A.opt options "is_stable" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
      in
      [ a; b ]
  | "lax.split" ->
      Ojax.Lax_api.split
        ((A.value arr) positional.(0))
        (A.int_array positional.(1))
        (A.int_ (A.get options "axis" (`Int 0)))
  | "lax.stack" ->
      [
        Ojax.Lax_api.stack
          ((A.value_list arr) positional.(0))
          (A.int_ (A.get options "axis" (`Int 0)));
      ]
  | "lax.top_k" ->
      let a, b =
        Ojax.Lax_api.top_k
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1))
      in
      [ a; b ]
  | "lax.unstack" ->
      Ojax.Lax_api.unstack
        ((A.value arr) positional.(0))
        (A.int_ (A.get options "axis" (`Int 0)))
  | "lax_linalg.cholesky" ->
      [
        Ojax.Lax.Linalg.cholesky
          ?symmetrize_input:(A.opt options "symmetrize_input" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "lax_linalg.eig" ->
      Ojax.Lax.Linalg.eig
        ?implementation:(A.opt options "implementation" A.eig_implementation)
        ?compute_left_eigenvectors:
          (A.opt options "compute_left_eigenvectors" A.bool_)
        ?compute_right_eigenvectors:
          (A.opt options "compute_right_eigenvectors" A.bool_)
        ?enable_eigvec_derivs:(A.opt options "enable_eigvec_derivs" A.bool_)
        ((A.value arr) positional.(0))
  | "lax_linalg.eigh" ->
      let a, b =
        Ojax.Lax.Linalg.eigh
          ?implementation:(A.opt options "implementation" A.eigh_implementation)
          ?lower:(A.opt options "lower" A.bool_)
          ?symmetrize_input:(A.opt options "symmetrize_input" A.bool_)
          ?sort_eigenvalues:(A.opt options "sort_eigenvalues" A.bool_)
          ?subset_by_index:(A.opt options "subset_by_index" A.int_pair)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "lax_linalg.ormqr" ->
      [
        Ojax.Linalg_extras.ormqr
          ?left:(A.opt options "left" A.bool_)
          ?transpose:(A.opt options "transpose" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "lax_linalg.qdwh" ->
      let a, b, c, d =
        Ojax.Qdwh.qdwh
          ?is_hermitian:(A.opt options "is_hermitian" A.bool_)
          ?max_iterations:(A.opt options "max_iterations" A.int_)
          ?eps:(A.opt options "eps" A.float_)
          ((A.value arr) positional.(0))
      in
      [ a; b; c; d ]
  | "lax_linalg.qr" ->
      A.refuse options [ "pivoting" ];
      let a, b =
        Ojax.Lax.Linalg.qr
          ?full_matrices:(A.opt options "full_matrices" A.bool_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "lax_linalg.schur" ->
      Ojax.Lax.Linalg.schur
        ?compute_schur_vectors:(A.opt options "compute_schur_vectors" A.bool_)
        ?sort_eig_vals:(A.opt options "sort_eig_vals" A.bool_)
        ((A.value arr) positional.(0))
  | "lax_linalg.svd" ->
      Ojax.Lax.Linalg.svd
        ?full_matrices:(A.opt options "full_matrices" A.bool_)
        ?compute_uv:(A.opt options "compute_uv" A.bool_)
        ?subset_by_index:(A.opt options "subset_by_index" A.int_pair)
        ((A.value arr) positional.(0))
  | "lax_linalg.symmetric_product" ->
      [
        Ojax.Linalg_extras.symmetric_product
          ?alpha:(A.opt options "alpha" A.float_)
          ?beta:(A.opt options "beta" A.float_)
          ?symmetrize_output:(A.opt options "symmetrize_output" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax_linalg.triangular_solve" ->
      [
        Ojax.Lax.Linalg.triangular_solve
          ?left_side:(A.opt options "left_side" A.bool_)
          ?lower:(A.opt options "lower" A.bool_)
          ?transpose_a:(A.opt options "transpose_a" A.bool_)
          ?conjugate_a:(A.opt options "conjugate_a" A.bool_)
          ?unit_diagonal:(A.opt options "unit_diagonal" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "lax_linalg.tridiagonal" ->
      let a, b, c, d =
        Ojax.Lax.Linalg.tridiagonal
          ?lower:(A.opt options "lower" A.bool_)
          ((A.value arr) positional.(0))
      in
      [ a; b; c; d ]
  | "lax_linalg.tridiagonal_solve" ->
      [
        Ojax.Lax.Linalg.tridiagonal_solve
          ?perturb_singular:(A.opt options "perturb_singular" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2))
          ((A.value arr) positional.(3));
      ]
  | "nn.celu" ->
      [
        Ojax.Nn.Functions.celu
          ?alpha:(A.opt options "alpha" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "nn.dot_product_attention" ->
      A.refuse options [ "return_residual" ];
      [
        Ojax.Nn.Functions.dot_product_attention
          ?bias:(A.opt options "bias" (A.value arr))
          ?mask:(A.opt options "mask" (A.value arr))
          ?scale:(A.opt options "scale" A.float_)
          ?is_causal:(A.opt options "is_causal" A.bool_)
          ?query_seq_lengths:(A.opt options "query_seq_lengths" (A.value arr))
          ?key_value_seq_lengths:
            (A.opt options "key_value_seq_lengths" (A.value arr))
          ?local_window_size:(A.opt options "local_window_size" A.window_pair)
          ?implementation:(A.opt options "implementation" A.string_)
          ~query:((A.value arr) positional.(0))
          ~key:((A.value arr) positional.(1))
          ~value:((A.value arr) positional.(2))
          ();
      ]
  | "nn.elu" ->
      [
        Ojax.Nn.Functions.elu
          ?alpha:(A.opt options "alpha" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "nn.gelu" ->
      [
        Ojax.Nn.Functions.gelu
          ?approximate:(A.opt options "approximate" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "nn.glu" ->
      [
        Ojax.Nn.Functions.glu
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "nn.leaky_relu" ->
      [
        Ojax.Nn.Functions.leaky_relu
          ?negative_slope:(A.opt options "negative_slope" A.float_)
          ((A.value arr) positional.(0));
      ]
  | "nn.log_softmax" ->
      [
        Ojax.Nn.Functions.log_softmax
          ?axis:(A.opt options "axis" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "nn.logmeanexp" ->
      [
        Ojax.Nn.Functions.logmeanexp
          ?axis:(A.opt options "axis" A.int_array)
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "nn.logsumexp" ->
      A.refuse options [ "return_sign" ];
      [
        Ojax.Nn.Functions.logsumexp
          ?axis:(A.opt options "axis" A.int_array)
          ?b:(A.opt options "b" (A.value arr))
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "nn.scaled_matmul" ->
      [
        Ojax.Nn.Functions.scaled_matmul
          ?preferred_element_type:
            (A.opt options "preferred_element_type" A.dtype)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2))
          ((A.value arr) positional.(3));
      ]
  | "nn.softmax" ->
      [
        Ojax.Nn.Functions.softmax
          ?axis:(A.opt options "axis" A.int_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "nn.squareplus" ->
      [
        (match A.get options "b" `Null with
        | `Null -> Ojax.Nn.Functions.squareplus ((A.value arr) positional.(0))
        | b ->
            Ojax.Nn.Functions.squareplus_value ~b:(A.value arr b)
              ((A.value arr) positional.(0)));
      ]
  | "nn.standardize" ->
      [
        Ojax.Nn.Functions.standardize
          ?axis:(A.opt options "axis" A.int_)
          ?mean:(A.opt options "mean" (A.value arr))
          ?variance:(A.opt options "variance" (A.value arr))
          ?epsilon:(A.opt options "epsilon" A.float_)
          ?where:(A.opt options "where" (A.value arr))
          ?algorithm:(A.opt options "algorithm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "random.ball" ->
      [
        Ojax.Random_extras.ball
          ?dtype:(A.opt options "dtype" A.dtype)
          ?p:(A.opt options "p" A.float_)
          ~d:(A.int_ positional.(1))
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.bernoulli" ->
      [
        Ojax.Random.Core.bernoulli
          ?mode:(A.opt options "mode" A.string_)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~p:(A.float_ (A.get options "p" (`Float 0.5)))
          ((A.value arr) positional.(0));
      ]
  | "random.beta" ->
      [
        Ojax.Random.Core.beta
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~a:(A.float_ positional.(1))
          ~b:(A.float_ positional.(2))
          ((A.value arr) positional.(0));
      ]
  | "random.binomial" ->
      [
        Ojax.Random.Core.binomial
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~n:(A.float_ positional.(1))
          ~p:(A.float_ positional.(2))
          ((A.value arr) positional.(0));
      ]
  | "random.bits" ->
      [
        Ojax.Random.Core.bits
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.categorical" ->
      [
        Ojax.Random.Core.categorical_full
          ?axis:(A.opt options "axis" A.int_)
          ?shape:(A.opt options "shape" A.int_array)
          ?replace:(A.opt options "replace" A.bool_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "random.cauchy" ->
      [
        Ojax.Random.Core.cauchy
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.chisquare" ->
      [
        Ojax.Random.Core.chisquare
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~df:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.choice" ->
      [
        Ojax.Random.Core.choice_array
          ?shape:(A.opt options "shape" A.int_array)
          ?replace:(A.opt options "replace" A.bool_)
          ?p:(A.opt options "p" (A.value arr))
          ?axis:(A.opt options "axis" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "random.dirichlet" ->
      [
        Ojax.Random.Core.dirichlet
          ?dtype:(A.opt options "dtype" A.dtype)
          ~alpha:((A.value arr) positional.(1))
          ~shape:(A.int_array (A.get options "shape" `Null))
          ((A.value arr) positional.(0));
      ]
  | "random.double_sided_maxwell" ->
      [
        Ojax.Random_extras.double_sided_maxwell
          ?dtype:(A.opt options "dtype" A.dtype)
          ~loc:(A.float_ positional.(1))
          ~scale:(A.float_ positional.(2))
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.exponential" ->
      [
        Ojax.Random.Core.exponential
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.f" ->
      [
        Ojax.Random.Core.f
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~dfnum:(A.float_ positional.(1))
          ~dfden:(A.float_ positional.(2))
          ((A.value arr) positional.(0));
      ]
  | "random.gamma" ->
      [
        Ojax.Random.Core.gamma
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~a:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.generalized_normal" ->
      [
        Ojax.Random.Core.generalized_normal
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ~p:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.geometric" ->
      [
        Ojax.Random.Core.geometric
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~p:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.gumbel" ->
      [
        Ojax.Random.Core.gumbel
          ?dtype:(A.opt options "dtype" A.dtype)
          ?mode:(A.opt options "mode" A.string_)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.key" ->
      [
        Ojax.Random.Core.key
          ?impl:(A.opt options "impl" A.string_)
          ?dtype:(A.opt options "dtype" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "random.laplace" ->
      [
        Ojax.Random.Core.laplace
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.loggamma" ->
      [
        Ojax.Random.Core.loggamma
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~a:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.logistic" ->
      [
        Ojax.Random.Core.logistic
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.lognormal" ->
      [
        Ojax.Random.Core.lognormal
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~sigma:(A.float_ (A.get options "sigma" (`Float 1.0)))
          ((A.value arr) positional.(0));
      ]
  | "random.maxwell" ->
      [
        Ojax.Random_extras.maxwell
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.multivariate_normal" ->
      [
        Ojax.Random_extras.multivariate_normal
          ?shape:(A.opt options "shape" A.int_array)
          ?method_:(A.opt options "method" A.string_)
          ?dtype:(A.opt options "dtype" A.dtype)
          ~mean:((A.value arr) positional.(1))
          ~cov:((A.value arr) positional.(2))
          ((A.value arr) positional.(0));
      ]
  | "random.normal" ->
      [
        Ojax.Random.Core.normal
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.orthogonal" ->
      [
        Ojax.Random_extras.orthogonal
          ?dtype:(A.opt options "dtype" A.dtype)
          ?m:(A.opt options "m" A.int_)
          ~n:(A.int_ positional.(1))
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.pareto" ->
      [
        Ojax.Random.Core.pareto
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~b:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.permutation" ->
      [
        Ojax.Random.Core.permutation_array
          ?axis:(A.opt options "axis" A.int_)
          ?independent:(A.opt options "independent" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "random.poisson" ->
      [
        Ojax.Random.Core.poisson
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~lam:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.rademacher" ->
      [
        Ojax.Random.Core.rademacher
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ((A.value arr) positional.(0));
      ]
  | "random.randint" ->
      [
        Ojax.Random.Core.randint
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array positional.(1))
          ~minval:(A.int_ positional.(2))
          ~maxval:(A.int_ positional.(3))
          ((A.value arr) positional.(0));
      ]
  | "random.rayleigh" ->
      [
        Ojax.Random.Core.rayleigh
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~scale:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.split" ->
      [
        Ojax.Random.Core.split
          ((A.value arr) positional.(0))
          (A.int_ (A.get options "num" (`Int 2)));
      ]
  | "random.t" ->
      [
        Ojax.Random.Core.t
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~df:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.triangular" ->
      [
        Ojax.Random.Core.triangular
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~left:(A.float_ positional.(1))
          ~mode:(A.float_ positional.(2))
          ~right:(A.float_ positional.(3))
          ((A.value arr) positional.(0));
      ]
  | "random.truncated_normal" ->
      [
        Ojax.Random.Core.truncated_normal
          ?dtype:(A.opt options "dtype" A.dtype)
          ~lower:(A.float_ positional.(1))
          ~upper:(A.float_ positional.(2))
          ~shape:(A.int_array (A.get options "shape" `Null))
          ((A.value arr) positional.(0));
      ]
  | "random.uniform" ->
      [
        Ojax.Random.Core.uniform
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ~minval:(A.float_ (A.get options "minval" (`Float 0.0)))
          ~maxval:(A.float_ (A.get options "maxval" (`Float 1.0)))
          ((A.value arr) positional.(0));
      ]
  | "random.wald" ->
      [
        Ojax.Random.Core.wald
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" `Null))
          ~mean:(A.float_ positional.(1))
          ((A.value arr) positional.(0));
      ]
  | "random.weibull_min" ->
      [
        Ojax.Random.Core.weibull_min
          ?dtype:(A.opt options "dtype" A.dtype)
          ~shape:(A.int_array (A.get options "shape" (`List [])))
          ~scale:(A.float_ positional.(1))
          ~concentration:(A.float_ positional.(2))
          ((A.value arr) positional.(0));
      ]
  | "ops.segment_max" ->
      [
        Ojax.Ops.Segment.segment_max
          ?num_segments:(A.opt options "num_segments" A.int_)
          ?indices_are_sorted:(A.opt options "indices_are_sorted" A.bool_)
          ?unique_indices:(A.opt options "unique_indices" A.bool_)
          ?bucket_size:(A.opt options "bucket_size" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "ops.segment_min" ->
      [
        Ojax.Ops.Segment.segment_min
          ?num_segments:(A.opt options "num_segments" A.int_)
          ?indices_are_sorted:(A.opt options "indices_are_sorted" A.bool_)
          ?unique_indices:(A.opt options "unique_indices" A.bool_)
          ?bucket_size:(A.opt options "bucket_size" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "ops.segment_prod" ->
      [
        Ojax.Ops.Segment.segment_prod
          ?num_segments:(A.opt options "num_segments" A.int_)
          ?indices_are_sorted:(A.opt options "indices_are_sorted" A.bool_)
          ?unique_indices:(A.opt options "unique_indices" A.bool_)
          ?bucket_size:(A.opt options "bucket_size" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "ops.segment_sum" ->
      [
        Ojax.Ops.Segment.segment_sum
          ?num_segments:(A.opt options "num_segments" A.int_)
          ?indices_are_sorted:(A.opt options "indices_are_sorted" A.bool_)
          ?unique_indices:(A.opt options "unique_indices" A.bool_)
          ?bucket_size:(A.opt options "bucket_size" A.int_)
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_fft.dct" ->
      [
        Ojax.Scipy.Fft.dct
          ?type_:(A.opt options "type" A.int_)
          ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "sp_fft.dctn" ->
      [
        Ojax.Scipy.Fft.dctn
          ?type_:(A.opt options "type" A.int_)
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "sp_fft.idct" ->
      [
        Ojax.Scipy.Fft.idct
          ?type_:(A.opt options "type" A.int_)
          ?n:(A.opt options "n" A.int_)
          ?axis:(A.opt options "axis" A.int_)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "sp_fft.idctn" ->
      [
        Ojax.Scipy.Fft.idctn
          ?type_:(A.opt options "type" A.int_)
          ?s:(A.opt options "s" A.int_list)
          ?axes:(A.opt options "axes" A.int_list)
          ?norm:(A.opt options "norm" A.string_)
          ((A.value arr) positional.(0));
      ]
  | "sp_linalg.cholesky" ->
      [
        Ojax.Scipy.Linalg.cholesky
          ?lower:(A.opt options "lower" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "sp_linalg.convolution_matrix" ->
      [
        Ojax.Scipy.Linalg_special.convolution_matrix
          ?mode:(A.opt options "mode" A.string_)
          ((A.value arr) positional.(0))
          (A.int_ positional.(1));
      ]
  | "sp_linalg.dft" ->
      [
        Ojax.Scipy.Linalg_special.dft
          ?scale:(A.opt options "scale" A.string_)
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_ positional.(0));
      ]
  | "sp_linalg.eigh" ->
      Ojax.Scipy.Linalg.eigh
        ?b:(A.opt options "b" (A.value arr))
        ?lower:(A.opt options "lower" A.bool_)
        ?eigvals_only:(A.opt options "eigvals_only" A.bool_)
        ?eigvals:(A.opt options "eigvals" A.int_pair)
        ?type_:(A.opt options "type" A.int_)
        ((A.value arr) positional.(0))
  | "sp_linalg.eigh_tridiagonal" ->
      Ojax.Scipy.Linalg.eigh_tridiagonal
        ?eigvals_only:(A.opt options "eigvals_only" A.bool_)
        ?select:(A.opt options "select" A.string_)
        ?select_range:(A.opt options "select_range" A.float_pair)
        ?tol:(A.opt options "tol" A.float_)
        ?key:(A.opt options "key" (A.value arr))
        ((A.value arr) positional.(0))
        ((A.value arr) positional.(1))
  | "sp_linalg.expm" ->
      [
        Ojax.Scipy.Linalg.expm
          ?upper_triangular:(A.opt options "upper_triangular" A.bool_)
          ?max_squarings:(A.opt options "max_squarings" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "sp_linalg.expm_frechet" ->
      Ojax.Scipy.Linalg.expm_frechet
        ?method_:(A.opt options "method" A.string_)
        ?compute_expm:(A.opt options "compute_expm" A.bool_)
        ((A.value arr) positional.(0))
        ((A.value arr) positional.(1))
  | "sp_linalg.hadamard" ->
      [
        Ojax.Scipy.Linalg_special.hadamard
          ?dtype:(A.opt options "dtype" A.dtype)
          (A.int_ positional.(0));
      ]
  | "sp_linalg.hankel" ->
      [
        Ojax.Scipy.Linalg_special.hankel
          ?r:(A.opt options "r" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_linalg.helmert" ->
      [
        Ojax.Scipy.Linalg_special.helmert
          ?full:(A.opt options "full" A.bool_)
          (A.int_ positional.(0));
      ]
  | "sp_linalg.hessenberg" ->
      Ojax.Scipy.Linalg.hessenberg
        ?calc_q:(A.opt options "calc_q" A.bool_)
        ((A.value arr) positional.(0))
  | "sp_linalg.lu" ->
      Ojax.Scipy.Linalg.lu
        ?permute_l:(A.opt options "permute_l" A.bool_)
        ((A.value arr) positional.(0))
  | "sp_linalg.polar" ->
      Ojax.Scipy.Linalg.polar
        ?side:(A.opt options "side" A.string_)
        ?method_:(A.opt options "method" A.string_)
        ?eps:(A.opt options "eps" A.float_)
        ?max_iterations:(A.opt options "max_iterations" A.int_)
        ((A.value arr) positional.(0))
  | "sp_linalg.qr" ->
      Ojax.Scipy.Linalg.qr
        ?mode:(A.opt options "mode" A.string_)
        ?pivoting:(A.opt options "pivoting" A.bool_)
        ((A.value arr) positional.(0))
  | "sp_linalg.qr_multiply" ->
      Ojax.Scipy.Linalg.qr_multiply
        ?mode:(A.opt options "mode" A.string_)
        ?pivoting:(A.opt options "pivoting" A.bool_)
        ?conjugate:(A.opt options "conjugate" A.bool_)
        ((A.value arr) positional.(0))
        ((A.value arr) positional.(1))
  | "sp_linalg.schur" ->
      Ojax.Scipy.Linalg.schur
        ?output:(A.opt options "output" A.string_)
        ((A.value arr) positional.(0))
  | "sp_linalg.solve" ->
      [
        Ojax.Scipy.Linalg.solve
          ?lower:(A.opt options "lower" A.bool_)
          ?assume_a:(A.opt options "assume_a" A.string_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_linalg.solve_sylvester" ->
      [
        Ojax.Scipy.Linalg_special.solve_sylvester
          ?method_:(A.opt options "method" A.string_)
          ?tol:(A.opt options "tol" A.float_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_linalg.solve_triangular" ->
      [
        Ojax.Scipy.Linalg.solve_triangular
          ?trans:(A.opt options "trans" A.trans)
          ?lower:(A.opt options "lower" A.bool_)
          ?unit_diagonal:(A.opt options "unit_diagonal" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_linalg.sqrtm" ->
      [
        Ojax.Scipy.Linalg.sqrtm
          ?blocksize:(A.opt options "blocksize" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "sp_linalg.svd" ->
      Ojax.Scipy.Linalg.svd
        ?full_matrices:(A.opt options "full_matrices" A.bool_)
        ?compute_uv:(A.opt options "compute_uv" A.bool_)
        ?lapack_driver:(A.opt options "lapack_driver" A.string_)
        ((A.value arr) positional.(0))
  | "sp_linalg.toeplitz" ->
      [
        Ojax.Scipy.Linalg.toeplitz
          ?r:(A.opt options "r" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_signal.convolve" ->
      [
        Ojax.Scipy.Signal.convolve
          ?mode:(A.opt options "mode" A.string_)
          ?method_:(A.opt options "method" A.string_)
          ?precision:(A.opt options "precision" A.precision)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_signal.convolve2d" ->
      [
        Ojax.Scipy.Signal.convolve2d
          ?mode:(A.opt options "mode" A.string_)
          ?boundary:(A.opt options "boundary" A.string_)
          ?fillvalue:(A.opt options "fillvalue" A.float_)
          ?precision:(A.opt options "precision" A.precision)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_signal.correlate" ->
      [
        Ojax.Scipy.Signal.correlate
          ?mode:(A.opt options "mode" A.string_)
          ?method_:(A.opt options "method" A.string_)
          ?precision:(A.opt options "precision" A.precision)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_signal.correlate2d" ->
      [
        Ojax.Scipy.Signal.correlate2d
          ?mode:(A.opt options "mode" A.string_)
          ?boundary:(A.opt options "boundary" A.string_)
          ?fillvalue:(A.opt options "fillvalue" A.float_)
          ?precision:(A.opt options "precision" A.precision)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_signal.csd" ->
      let a, b =
        Ojax.Scipy.Signal.csd
          ?fs:(A.opt options "fs" A.float_)
          ?window:(A.opt options "window" (A.value arr))
          ?nperseg:(A.opt options "nperseg" A.int_)
          ?noverlap:(A.opt options "noverlap" A.int_)
          ?nfft:(A.opt options "nfft" A.int_)
          ?detrend:(A.opt_keep options "detrend" A.string_option)
          ?return_onesided:(A.opt options "return_onesided" A.bool_)
          ?scaling:(A.opt options "scaling" A.string_)
          ?axis:(A.opt options "axis" A.int_)
          ?average:(A.opt options "average" A.string_)
          ((A.value arr) positional.(0))
          ((A.value_option arr) positional.(1))
      in
      [ a; b ]
  | "sp_signal.detrend" ->
      [
        Ojax.Scipy.Signal.detrend
          ?axis:(A.opt options "axis" A.int_)
          ?type_:(A.opt options "type" A.string_)
          ?bp:(A.opt options "bp" A.int_list)
          ((A.value arr) positional.(0));
      ]
  | "sp_signal.fftconvolve" ->
      [
        Ojax.Scipy.Signal.fftconvolve
          ?mode:(A.opt options "mode" A.string_)
          ?axes:(A.opt options "axes" A.int_list)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_signal.istft" ->
      let a, b =
        Ojax.Scipy.Signal.istft
          ?fs:(A.opt options "fs" A.float_)
          ?window:(A.opt options "window" (A.value arr))
          ?nperseg:(A.opt options "nperseg" A.int_)
          ?noverlap:(A.opt options "noverlap" A.int_)
          ?nfft:(A.opt options "nfft" A.int_)
          ?input_onesided:(A.opt options "input_onesided" A.bool_)
          ?boundary:(A.opt options "boundary" A.bool_)
          ?time_axis:(A.opt options "time_axis" A.int_)
          ?freq_axis:(A.opt options "freq_axis" A.int_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "sp_signal.stft" ->
      let a, b, c =
        Ojax.Scipy.Signal.stft
          ?fs:(A.opt options "fs" A.float_)
          ?window:(A.opt options "window" (A.value arr))
          ?nperseg:(A.opt options "nperseg" A.int_)
          ?noverlap:(A.opt options "noverlap" A.int_)
          ?nfft:(A.opt options "nfft" A.int_)
          ?detrend:(A.opt options "detrend" A.string_)
          ?return_onesided:(A.opt options "return_onesided" A.bool_)
          ?boundary:(A.opt_keep options "boundary" A.string_option)
          ?padded:(A.opt options "padded" A.bool_)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0))
      in
      [ a; b; c ]
  | "sp_signal.welch" ->
      let a, b =
        Ojax.Scipy.Signal.welch
          ?fs:(A.opt options "fs" A.float_)
          ?window:(A.opt options "window" (A.value arr))
          ?nperseg:(A.opt options "nperseg" A.int_)
          ?noverlap:(A.opt options "noverlap" A.int_)
          ?nfft:(A.opt options "nfft" A.int_)
          ?detrend:(A.opt_keep options "detrend" A.string_option)
          ?return_onesided:(A.opt options "return_onesided" A.bool_)
          ?scaling:(A.opt options "scaling" A.string_)
          ?axis:(A.opt options "axis" A.int_)
          ?average:(A.opt options "average" A.string_)
          ((A.value arr) positional.(0))
      in
      [ a; b ]
  | "sp_special.comb" ->
      [
        Ojax.Scipy.Special.comb
          ?repetition:(A.opt options "repetition" A.bool_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_special.factorial" ->
      [
        Ojax.Scipy.Special.factorial
          ?exact:(A.opt options "exact" A.bool_)
          ((A.value arr) positional.(0));
      ]
  | "sp_special.log_ndtr" ->
      [
        Ojax.Scipy.Special.log_ndtr
          ?series_order:(A.opt options "series_order" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "sp_special.log_softmax" ->
      [
        Ojax.Scipy.Special.log_softmax
          ?axis:(A.opt options "axis" A.axis_arg)
          ((A.value arr) positional.(0));
      ]
  | "sp_special.logsumexp" ->
      A.refuse options [ "return_sign" ];
      [
        Ojax.Scipy.Special.logsumexp
          ?axis:(A.opt options "axis" A.int_array)
          ?b:(A.opt options "b" (A.value arr))
          ?keepdims:(A.opt options "keepdims" A.bool_)
          ?where:(A.opt options "where" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_special.softmax" ->
      [
        Ojax.Scipy.Special.softmax
          ?axis:(A.opt options "axis" A.axis_arg)
          ((A.value arr) positional.(0));
      ]
  | "sp_special.sph_harm_y" ->
      [
        Ojax.Scipy.Special.sph_harm_y
          ?diff_n:(A.opt options "diff_n" A.int_)
          ?n_max:(A.opt options "n_max" A.int_)
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2))
          ((A.value arr) positional.(3));
      ]
  | "sp_special.zeta" ->
      [
        Ojax.Scipy.Special.zeta
          ?q:(A.opt options "q" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_integrate.trapezoid" ->
      [
        Ojax.Scipy.Integrate.trapezoid
          ?x:(A.opt options "x" (A.value arr))
          ?dx:(A.opt options "dx" A.float_)
          ?axis:(A.opt options "axis" A.int_)
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.cdf" ->
      [
        Ojax.Scipy.Stats.Norm.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.isf" ->
      [
        Ojax.Scipy.Stats.Norm.isf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.logcdf" ->
      [
        Ojax.Scipy.Stats.Norm.logcdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.logpdf" ->
      [
        Ojax.Scipy.Stats.Norm.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.logsf" ->
      [
        Ojax.Scipy.Stats.Norm.logsf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.pdf" ->
      [
        Ojax.Scipy.Stats.Norm.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.ppf" ->
      [
        Ojax.Scipy.Stats.Norm.ppf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_norm.sf" ->
      [
        Ojax.Scipy.Stats.Norm.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_t.logpdf" ->
      [
        Ojax.Scipy.Stats.T.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_t.pdf" ->
      [
        Ojax.Scipy.Stats.T.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.cdf" ->
      [
        Ojax.Scipy.Stats.Gamma.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.logcdf" ->
      [
        Ojax.Scipy.Stats.Gamma.logcdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.logpdf" ->
      [
        Ojax.Scipy.Stats.Gamma.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.logsf" ->
      [
        Ojax.Scipy.Stats.Gamma.logsf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.pdf" ->
      [
        Ojax.Scipy.Stats.Gamma.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_gamma.sf" ->
      [
        Ojax.Scipy.Stats.Gamma.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1));
      ]
  | "sp_stats_beta.cdf" ->
      [
        Ojax.Scipy.Stats.Beta.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_beta.logcdf" ->
      [
        Ojax.Scipy.Stats.Beta.logcdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_beta.logpdf" ->
      [
        Ojax.Scipy.Stats.Beta.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_beta.logsf" ->
      [
        Ojax.Scipy.Stats.Beta.logsf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_beta.pdf" ->
      [
        Ojax.Scipy.Stats.Beta.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_beta.sf" ->
      [
        Ojax.Scipy.Stats.Beta.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0))
          ((A.value arr) positional.(1))
          ((A.value arr) positional.(2));
      ]
  | "sp_stats_expon.cdf" ->
      [
        Ojax.Scipy.Stats.Expon.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.logcdf" ->
      [
        Ojax.Scipy.Stats.Expon.logcdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.logpdf" ->
      [
        Ojax.Scipy.Stats.Expon.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.logsf" ->
      [
        Ojax.Scipy.Stats.Expon.logsf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.pdf" ->
      [
        Ojax.Scipy.Stats.Expon.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.ppf" ->
      [
        Ojax.Scipy.Stats.Expon.ppf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_expon.sf" ->
      [
        Ojax.Scipy.Stats.Expon.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.cdf" ->
      [
        Ojax.Scipy.Stats.Cauchy.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.isf" ->
      [
        Ojax.Scipy.Stats.Cauchy.isf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.logcdf" ->
      [
        Ojax.Scipy.Stats.Cauchy.logcdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.logpdf" ->
      [
        Ojax.Scipy.Stats.Cauchy.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.logsf" ->
      [
        Ojax.Scipy.Stats.Cauchy.logsf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.pdf" ->
      [
        Ojax.Scipy.Stats.Cauchy.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.ppf" ->
      [
        Ojax.Scipy.Stats.Cauchy.ppf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_cauchy.sf" ->
      [
        Ojax.Scipy.Stats.Cauchy.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_laplace.cdf" ->
      [
        Ojax.Scipy.Stats.Laplace.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_laplace.logpdf" ->
      [
        Ojax.Scipy.Stats.Laplace.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_laplace.pdf" ->
      [
        Ojax.Scipy.Stats.Laplace.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.cdf" ->
      [
        Ojax.Scipy.Stats.Logistic.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.isf" ->
      [
        Ojax.Scipy.Stats.Logistic.isf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.logpdf" ->
      [
        Ojax.Scipy.Stats.Logistic.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.pdf" ->
      [
        Ojax.Scipy.Stats.Logistic.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.ppf" ->
      [
        Ojax.Scipy.Stats.Logistic.ppf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_logistic.sf" ->
      [
        Ojax.Scipy.Stats.Logistic.sf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_uniform.cdf" ->
      [
        Ojax.Scipy.Stats.Uniform.cdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_uniform.logpdf" ->
      [
        Ojax.Scipy.Stats.Uniform.logpdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_uniform.pdf" ->
      [
        Ojax.Scipy.Stats.Uniform.pdf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | "sp_stats_uniform.ppf" ->
      [
        Ojax.Scipy.Stats.Uniform.ppf
          ?loc:(A.opt options "loc" (A.value arr))
          ?scale:(A.opt options "scale" (A.value arr))
          ((A.value arr) positional.(0));
      ]
  | other -> A.no ("function:" ^ other)
