open Types

let tuple = Validate.python_tuple

let trailing ~name ~ranks shapes =
  let rec go i ranks shapes =
    match (ranks, shapes) with
    | rank :: ranks, shape :: shapes ->
        let n = Array.length shape in
        if n < rank then
          Py_exn.type_error
            (Printf.sprintf
               "Input %d to %s must have rank at least %d, but got shape=%s" i
               name rank (tuple shape));
        (Array.sub shape 0 (n - rank), Array.sub shape (n - rank) rank)
        :: go (i + 1) ranks shapes
    | _ -> []
  in
  let parts = go 0 ranks shapes in
  (match parts with
  | (first, _) :: rest
    when List.exists (fun (b, _) -> Array.length b <> Array.length first) rest
    ->
      Py_exn.type_error
        (Printf.sprintf
           "All inputs to %s must have the same number of batch dimensions, \
            but got [%s] batch dimensions for the inputs."
           name
           (String.concat ", "
              (List.map (fun (b, _) -> string_of_int (Array.length b)) parts)))
  | _ -> ());
  List.map snd parts

let last a = a.(Array.length a - 1)

let square name s =
  if s.(0) <> s.(1) then
    Py_exn.value_error
      (Printf.sprintf "The input to %s must be a square matrix. Got shape %s."
         name (tuple s))

let rule ~name ~ranks shapes f = f (trailing ~name ~ranks (shapes ()))

let concatenate_rule dimension shapes =
  let listed () = String.concat ", " (List.map tuple shapes) in
  match shapes with
  | [] -> Py_exn.type_error "concatenate expects at least one operand, got 0."
  | first :: _ ->
      let rank = Array.length first in
      if List.exists (fun s -> Array.length s <> rank) shapes then
        Py_exn.type_error
          (Printf.sprintf
             "Cannot concatenate arrays with different numbers of dimensions: \
              got %s."
             (listed ()));
      if dimension < 0 || dimension >= rank then
        Py_exn.type_error
          (Printf.sprintf
             "concatenate dimension out of bounds: dimension %d for shapes %s."
             dimension (listed ()));
      let others s =
        List.filteri (fun i _ -> i <> dimension) (Array.to_list s)
      in
      if List.exists (fun s -> others s <> others first) shapes then
        Py_exn.type_error
          (Printf.sprintf
             "Cannot concatenate arrays with shapes that differ in dimensions \
              other than the one being concatenated: concatenating along \
              dimension %d for shapes %s."
             dimension (listed ()))

let check prim shapes =
  match prim with
  | Concatenate dimension -> concatenate_rule dimension (shapes ())
  | Cholesky ->
      rule ~name:"cholesky" ~ranks:[ 2 ] shapes (function
        | s :: _ -> square "cholesky" s
        | [] -> ())
  | Cholesky_update ->
      rule ~name:"cholesky_update" ~ranks:[ 2; 1 ] shapes (function
        | r :: w :: _ ->
            if r.(0) <> r.(1) || w.(0) <> r.(1) then
              Py_exn.value_error
                (Printf.sprintf
                   "Rank-1 update to Cholesky decomposition takes a square \
                    matrix and a vector of the same size as input. Got shapes \
                    %s and %s instead"
                   (tuple r) (tuple w))
        | _ -> ())
  | Eig _ ->
      rule ~name:"eig" ~ranks:[ 2 ] shapes (function
        | s :: _ -> square "eig" s
        | [] -> ())
  | Eigh _ ->
      rule ~name:"eigh" ~ranks:[ 2 ] shapes (function
        | s :: _ ->
            if s.(0) <> last s then
              Py_exn.value_error
                ("Argument to symmetric eigendecomposition must have shape \
                  [..., n, n], got shape " ^ tuple s)
        | [] -> ())
  | Hessenberg ->
      rule ~name:"hessenberg" ~ranks:[ 2 ] shapes (function
        | s :: _ ->
            if s.(0) <> last s then
              Py_exn.value_error
                ("Argument to Hessenberg reduction must have shape [..., n, \
                  n], got shape " ^ tuple s)
        | [] -> ())
  | Householder_product ->
      rule ~name:"householder_product" ~ranks:[ 2; 1 ] shapes (function
        | a :: taus :: _ ->
            if a.(0) < a.(1) then
              Py_exn.value_error
                ("The first argument to householder_product must have at least \
                  as many rows as columns, got shape " ^ tuple a);
            if taus.(0) > min a.(0) a.(1) then
              Py_exn.value_error
                "The second argument to householder_product must not have more \
                 rows than the minimum of the first argument's rows and \
                 columns."
        | _ -> ())
  | Ormqr { ormqr_left; _ } ->
      rule ~name:"ormqr" ~ranks:[ 2; 1; 2 ] shapes (function
        | a :: _ :: c :: _ ->
            if ormqr_left && c.(0) <> a.(0) then
              Py_exn.value_error
                (Printf.sprintf
                   "ormqr with left=True expects c to have the same number of \
                    rows as the Householder matrix a. Got a shape %s and c \
                    shape %s."
                   (tuple a) (tuple c));
            if (not ormqr_left) && c.(1) <> a.(0) then
              Py_exn.value_error
                (Printf.sprintf
                   "ormqr with left=False expects c to have the same number of \
                    columns as the Householder matrix a has rows. Got a shape \
                    %s and c shape %s."
                   (tuple a) (tuple c))
        | _ -> ())
  | Lu -> rule ~name:"lu" ~ranks:[ 2 ] shapes ignore
  | Lu_pivots_to_permutation permutation_size ->
      rule ~name:"lu_pivots_to_permutation" ~ranks:[ 1 ] shapes (function
        | s :: _ ->
            if not (permutation_size >= s.(0)) then
              Py_exn.value_error
                (Printf.sprintf
                   "Output permutation size %d has to exceed the trailing \
                    dimension of the pivots. Got pivots size %d"
                   permutation_size s.(0))
        | [] -> ())
  | Geqrf -> rule ~name:"geqrf" ~ranks:[ 2 ] shapes ignore
  | Geqp3 -> rule ~name:"geqp3" ~ranks:[ 2; 1 ] shapes ignore
  | Qr _ | Qr_pivoted _ -> rule ~name:"qr" ~ranks:[ 2 ] shapes ignore
  | Schur _ ->
      rule ~name:"schur" ~ranks:[ 2 ] shapes (function
        | s :: _ -> square "schur" s
        | [] -> ())
  | Svd _ -> rule ~name:"svd" ~ranks:[ 2 ] shapes ignore
  | Symmetric_product _ ->
      rule ~name:"symmetric_product" ~ranks:[ 2; 2 ] shapes (function
        | a :: c :: _ ->
            if a.(0) <> c.(1) || c.(0) <> c.(1) then
              Py_exn.value_error
                (Printf.sprintf
                   "symmetric_update expects a rectangular matrix of shape (m, \
                    n) and a square matrix of shape (n, n). Got shapes %s and \
                    %s."
                   (tuple a) (tuple c))
        | _ -> ())
  | Triangular_solve { left_side; _ } ->
      rule ~name:"triangular_solve" ~ranks:[ 2; 2 ] shapes (function
        | a :: b :: _ ->
            if a.(0) <> a.(1) then
              Py_exn.value_error
                ("The first input to triangular_solve must be a square matrix. \
                  Got shape " ^ tuple a ^ ".");
            if a.(1) <> b.(if left_side then 0 else 1) then
              Py_exn.value_error
                (Printf.sprintf
                   "Incompatible shapes for arguments to triangular_solve: %s \
                    and %s."
                   (tuple a) (tuple b))
        | _ -> ())
  | Tridiagonal _ ->
      rule ~name:"tridiagonal" ~ranks:[ 2 ] shapes (function
        | s :: _ ->
            if s.(0) <> s.(1) || s.(1) = 0 then
              Py_exn.value_error
                ("The input to tridiagonal must be a square matrix. Got shape "
               ^ tuple s ^ ".")
        | [] -> ())
  | Tridiagonal_solve _ ->
      rule ~name:"tridiagonal_solve" ~ranks:[ 1; 1; 1; 2 ] shapes (function
        | dl :: d :: du :: b :: _ ->
            if dl <> d || dl <> du then
              Py_exn.type_error
                "tridiagonal_solve requires that all diagonal arguments have \
                 the same shape.";
            if dl.(0) <> b.(0) then
              Py_exn.type_error
                "tridiagonal_solve requires that the leading ndim-1 dimensions \
                 of b equal the dimensions of the diagonal arguments."
        | _ -> ())
  | _ -> ()
