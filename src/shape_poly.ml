exception
  Inconclusive_dimension_operation = Errors.Inconclusive_dimension_operation

exception Unexpected_dim_var of string

let help_msg =
  "\n\
   This error arises for comparison operations with shapes that\n\
   are non-constant, and the result of the operation cannot be represented as\n\
   a boolean value for all values of the symbolic dimensions involved.\n\n\
   Please see \
   https://docs.jax.dev/en/latest/export/shape_poly.html#comparison-of-symbolic-dimensions-is-partially-supported\n\
   for more details.\n"

let inconclusive message = Inconclusive_dimension_operation (message ^ help_msg)

type comparator = Eq | Geq
type precision = For_geq0_or_leq0 | For_geq0_or_lt0 | Best

type factor = { desc : factor_desc; fsize : int }
and factor_desc = Var of string | Op of string * expr list
and term = { factors : (factor * int) list; tsize : int }
and expr = { terms : (term * int) list; scope : scope; esize : int }

and scope = {
  sid : int;
  mutable rules : (term * (expr * int)) list;
  mutable explicit : sym_constraint list;
  mutable initialized : bool;
  mutable bounds_cache : (expr * (float * float * precision)) list;
  mutable decision_initial : decision option;
}

and dim = Const of int | Sym of expr

and sym_constraint = {
  cmp : comparator;
  debug_str : string;
  e1 : dim;
  e2 : dim;
  diff : dim;
}

and decision = {
  dscope : scope;
  mutable processed : term list;
  mutable term_bounds : (term * (float * float)) list;
  mutable expr_constraints : (term * (comparator * int * expr) list) list;
}

type spec = { shape : dim array; dtype : Dtype.t }

let floordiv a b =
  if b = 0 then raise Division_by_zero
  else
    let q = a / b and r = a mod b in
    if r <> 0 && r < 0 <> (b < 0) then q - 1 else q

let pymod a b = a - (b * floordiv a b)

let prec_rank = function
  | For_geq0_or_leq0 -> 0
  | For_geq0_or_lt0 -> 1
  | Best -> 2

let bounds_sufficient prec lb ub =
  match prec with
  | For_geq0_or_leq0 -> lb >= 0.0 || ub <= 0.0
  | For_geq0_or_lt0 -> lb >= 0.0 || ub < 0.0
  | Best -> false

let rec cmp_sequence elem l1 l2 =
  match (l1, l2) with
  | [], [] -> 0
  | [], _ -> -1
  | _, [] -> 1
  | a :: ra, b :: rb ->
      let c = elem a b in
      if c <> 0 then c else cmp_sequence elem ra rb

let rec factor_cmp a b =
  let c = compare a.fsize b.fsize in
  if c <> 0 then c
  else
    match (a.desc, b.desc) with
    | Var x, Var y -> compare x y
    | Var _, Op _ -> -1
    | Op _, Var _ -> 1
    | Op (o1, l1), Op (o2, l2) ->
        let c = compare o1 o2 in
        if c <> 0 then c else cmp_sequence expr_cmp l1 l2

and term_cmp a b =
  let c = compare a.tsize b.tsize in
  if c <> 0 then c
  else
    cmp_sequence
      (fun (f1, e1) (f2, e2) ->
        let c = factor_cmp f1 f2 in
        if c <> 0 then c else compare e1 e2)
      a.factors b.factors

and expr_cmp a b =
  let c = compare a.esize b.esize in
  if c <> 0 then c
  else
    cmp_sequence
      (fun (t1, k1) (t2, k2) ->
        let c = term_cmp t1 t2 in
        if c <> 0 then c else compare k1 k2)
      a.terms b.terms

let term_size factors =
  List.fold_left (fun acc (f, e) -> acc + 1 + (e * f.fsize)) 0 factors

let make_term factors = { factors; tsize = term_size factors }
let term_one = make_term []
let is_constant_term t = t.factors = []

let expr_size terms =
  List.fold_left (fun acc (t, k) -> acc + 1 + (abs k * t.tsize)) 0 terms

let make_expr terms scope =
  let terms = if terms = [] then [ (term_one, 0) ] else terms in
  { terms; scope; esize = expr_size terms }

let factor_var v = { desc = Var v; fsize = 1 }

let factor_op op operands =
  {
    desc = Op (op, operands);
    fsize = List.fold_left (fun acc o -> acc + o.esize) 1 operands;
  }

let linear_pairs cmp l1 f1 l2 f2 =
  let rec go l1 l2 acc =
    match (l1, l2) with
    | (m1, c1) :: r1, (m2, c2) :: r2 ->
        let c = cmp m1 m2 in
        if c < 0 then go l1 r2 ((m2, c2 * f2) :: acc)
        else if c > 0 then go r1 l2 ((m1, c1 * f1) :: acc)
        else
          let k = (c1 * f1) + (c2 * f2) in
          if k = 0 then go r1 r2 acc else go r1 r2 ((m1, k) :: acc)
    | rest, [] ->
        List.rev_append acc
          (List.filter_map
             (fun (m, c) -> if c <> 0 then Some (m, c * f1) else None)
             rest)
    | [], rest ->
        List.rev_append acc
          (List.filter_map
             (fun (m, c) -> if c <> 0 then Some (m, c * f2) else None)
             rest)
  in
  go l1 l2 []

let term_mul a b = make_term (linear_pairs factor_cmp a.factors 1 b.factors 1)

let rec factor_to_string f =
  match f.desc with
  | Var v -> v
  | Op (op, operands) ->
      Printf.sprintf "%s(%s)" op
        (String.concat ", " (List.map expr_to_string operands))

and term_to_string t =
  let sorted =
    List.sort
      (fun (f1, e1) (f2, e2) ->
        let c = factor_cmp f1 f2 in
        if c <> 0 then c else compare e1 e2)
      t.factors
  in
  String.concat "*"
    (List.map
       (fun (f, e) ->
         if e <> 1 then Printf.sprintf "%s^%d" (factor_to_string f) e
         else factor_to_string f)
       sorted)

and expr_to_string e =
  let one (t, k) =
    let abs_k = abs k in
    let sign = if k > 0 then "+" else "-" in
    if is_constant_term t then
      if abs_k <> 0 then Printf.sprintf "%s %d" sign abs_k else "0"
    else if abs_k = 1 then Printf.sprintf "%s %s" sign (term_to_string t)
    else Printf.sprintf "%s %d*%s" sign abs_k (term_to_string t)
  in
  let res = String.concat " " (List.map one e.terms) in
  if String.length res >= 2 && String.sub res 0 2 = "+ " then
    String.sub res 2 (String.length res - 2)
  else res

let dim_to_string = function
  | Const k -> string_of_int k
  | Sym e -> expr_to_string e

let assoc_term t l =
  List.find_map (fun (k, v) -> if term_cmp k t = 0 then Some v else None) l

let term_divide a b =
  let factors = linear_pairs factor_cmp a.factors 1 b.factors (-1) in
  List.iter
    (fun (_, e) ->
      if e <= 0 then
        raise
          (inconclusive
             (Printf.sprintf "Cannot divide %s by %s." (term_to_string a)
                (term_to_string b))))
    factors;
  make_term factors

let add_coeff coeffs t k =
  if k = 0 then coeffs
  else
    let rec go = function
      | [] -> [ (t, k) ]
      | (t2, k2) :: rest when term_cmp t t2 = 0 -> (t2, k2 + k) :: rest
      | p :: rest -> p :: go rest
    in
    go coeffs

let coeff_to_sorted coeffs =
  List.sort
    (fun (a, _) (b, _) -> term_cmp b a)
    (List.filter (fun (_, k) -> k <> 0) coeffs)

let normalize_term t t_k scope =
  if scope.rules = [] then []
  else
    match assoc_term t scope.rules with
    | Some (after, k_after) when t_k mod k_after = 0 ->
        (t, -t_k)
        :: List.map
             (fun (t2, c2) -> (t2, c2 * floordiv t_k k_after))
             after.terms
    | _ ->
        if List.length t.factors <= 1 then []
        else
          let rec scan = function
            | [] -> []
            | (f, e) :: rest -> (
                match assoc_term (make_term [ (f, e) ]) scope.rules with
                | Some (after, k_after) when t_k mod k_after = 0 ->
                    let without = term_divide t (make_term [ (f, e) ]) in
                    (t, -t_k)
                    :: List.map
                         (fun (t2, c2) ->
                           (term_mul t2 without, c2 * floordiv t_k k_after))
                         after.terms
                | _ -> scan rest)
          in
          scan t.factors

let normalize_sorted terms scope =
  let rec first_update = function
    | [] -> None
    | (t, k) :: rest -> (
        match normalize_term t k scope with
        | [] -> first_update rest
        | updates -> Some updates)
  in
  let terms =
    match first_update terms with
    | None -> terms
    | Some updates ->
        coeff_to_sorted
          (List.fold_left (fun acc (t, k) -> add_coeff acc t k) terms updates)
  in
  match terms with
  | [] -> Const 0
  | (t, k) :: _ when is_constant_term t -> Const k
  | _ -> Sym (make_expr terms scope)

let from_term t k scope = normalize_sorted [ (t, k) ] scope
let from_var v scope = from_term (make_term [ (factor_var v, 1) ]) 1 scope

let scope_to_string scope =
  let base = Printf.sprintf "%d" scope.sid in
  if scope.explicit = [] then base
  else
    base ^ " with constraints:"
    ^ String.concat "" (List.map (fun c -> "\n  " ^ c.debug_str) scope.explicit)

let check_same_scope ?(when_ = "") ?(self_descr = " ")
    ?(other_descr = "unknown") scope (other : expr) =
  if scope != other.scope then
    invalid_arg
      (Printf.sprintf
         "Invalid mixing of symbolic scopes %s.\n\
          Expected %sscope %s\n\
          and found for '%s' (%s) scope %s\n\
          See \
          https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints."
         when_ self_descr (scope_to_string scope) (expr_to_string other)
         other_descr
         (scope_to_string other.scope))

let ensure_poly d op scope =
  match d with
  | Sym e ->
      check_same_scope scope e ~when_:("for operation " ^ op);
      e
  | Const k -> make_expr [ (term_one, k) ] scope

let from_operation op a b scope =
  let f = factor_op op [ ensure_poly a op scope; ensure_poly b op scope ] in
  from_term (make_term [ (f, 1) ]) 1 scope

let linear_combination e1 k1 e2 k2 scope =
  (match (e1, e2) with
  | Sym a, Sym b -> check_same_scope a.scope b ~when_:"for linear combination"
  | _ -> ());
  match (e1, e2) with
  | Const a, Const b -> Const ((a * k1) + (b * k2))
  | _ ->
      let t1 =
        match e1 with Sym e -> e.terms | Const c -> [ (term_one, c) ]
      in
      let t2 =
        match e2 with
        | Sym e -> e.terms
        | Const 0 -> []
        | Const c -> [ (term_one, c) ]
      in
      normalize_sorted (linear_pairs term_cmp t1 k1 t2 k2) scope

let scope_of a b =
  match (a, b) with Sym e, _ | _, Sym e -> Some e.scope | _ -> None

let add a b =
  match (a, b) with
  | Const x, Const y -> Const (x + y)
  | Sym _, Const 0 -> a
  | Const 0, Sym _ -> b
  | Sym e, _ -> linear_combination a 1 b 1 e.scope
  | Const _, Sym e -> linear_combination b 1 a 1 e.scope

let sub a b =
  match (a, b) with
  | Const x, Const y -> Const (x - y)
  | Sym _, Const 0 -> a
  | Sym e, _ -> linear_combination a 1 b (-1) e.scope
  | Const _, Sym e -> linear_combination b (-1) a 1 e.scope

let neg = function
  | Const x -> Const (-x)
  | Sym e as a -> linear_combination a (-1) (Const 0) 0 e.scope

let mul a b =
  match (a, b) with
  | Const x, Const y -> Const (x * y)
  | Sym _, Const 1 -> a
  | Const 1, Sym _ -> b
  | Sym _, Const 0 | Const 0, Sym _ -> Const 0
  | Sym e, Const k | Const k, Sym e ->
      linear_combination (Sym e) k (Const 0) 0 e.scope
  | Sym x, Sym y ->
      check_same_scope x.scope y ~when_:"for operation mul";
      let coeffs =
        List.fold_left
          (fun acc (m1, c1) ->
            List.fold_left
              (fun acc (m2, c2) -> add_coeff acc (term_mul m1 m2) (c1 * c2))
              acc y.terms)
          [] x.terms
      in
      normalize_sorted (coeff_to_sorted coeffs) x.scope

let rec pow d power =
  if power < 0 then
    invalid_arg
      (Printf.sprintf
         "Symbolic dimension cannot be raised to negative powers: '%s' ** '%d'"
         (dim_to_string d) power)
  else if power = 0 then Const 1
  else mul (pow d (power - 1)) d

let to_constant = function
  | Const k -> Some k
  | Sym e -> (
      match e.terms with
      | (t, k) :: _ when is_constant_term t -> Some k
      | _ -> None)

let is_symbolic_dim = function Const _ -> false | Sym _ -> true
let is_constant_dim d = not (is_symbolic_dim d)
let of_int k = Const k

let divmod_expr self divisor =
  let scope = self.scope in
  let fallback () =
    ( from_operation "floordiv" (Sym self) divisor scope,
      from_operation "mod" (Sym self) divisor scope )
  in
  try
    let rec loop dividend quotient =
      match to_constant dividend with
      | Some d -> (d, quotient)
      | None ->
          let mon, count =
            match dividend with
            | Sym e -> List.hd e.terms
            | Const _ -> assert false
          in
          let qterm, dcount =
            match divisor with
            | Sym d ->
                let dterm, dcount = List.hd d.terms in
                (term_divide mon dterm, dcount)
            | Const k -> (mon, k)
          in
          if pymod count dcount <> 0 then raise (inconclusive "");
          let q = from_term qterm (floordiv count dcount) scope in
          loop (sub dividend (mul q divisor)) (add quotient q)
    in
    let dividend, quotient = loop (Sym self) (Const 0) in
    match divisor with
    | Sym _ ->
        if dividend <> 0 then raise (inconclusive "");
        (quotient, Const 0)
    | Const k ->
        (add quotient (Const (floordiv dividend k)), Const (pymod dividend k))
  with Inconclusive_dimension_operation _ -> fallback ()

let divmod a b =
  match (a, b) with
  | Const x, Const y -> (Const (floordiv x y), Const (pymod x y))
  | Sym e, _ -> divmod_expr e b
  | Const _, Sym d -> divmod_expr (ensure_poly a "divmod" d.scope) b

let floordiv_dim a b = fst (divmod a b)
let mod_dim a b = snd (divmod a b)

let new_decision scope =
  { dscope = scope; processed = []; term_bounds = []; expr_constraints = [] }

let to_single_term e =
  let rec go n term = function
    | [] -> (
        match term with Some (k, t) -> Some (n, k, t) | None -> assert false)
    | (t, k) :: rest ->
        if is_constant_term t then go k term rest
        else if term = None then go n (Some (k, t)) rest
        else None
  in
  go 0 None e.terms

let unsatisfiable debug =
  invalid_arg (Printf.sprintf "Unsatisfiable constraint: %s" debug)

let replace_assoc t v l =
  let rec go = function
    | [] -> [ (t, v) ]
    | (k, _) :: rest when term_cmp k t = 0 -> (k, v) :: rest
    | p :: rest -> p :: go rest
  in
  go l

let add_to_state d cmp e debug =
  match to_single_term e with
  | Some (n, t_k, t) ->
      let lb, ub =
        Option.value
          (assoc_term t d.term_bounds)
          ~default:(neg_infinity, infinity)
      in
      let debug_s = Option.value debug ~default:"None" in
      let lb, ub =
        match cmp with
        | Eq ->
            if pymod n t_k <> 0 then unsatisfiable debug_s;
            let v = float_of_int (-floordiv n t_k) in
            (Float.max lb v, Float.min ub v)
        | Geq ->
            let bound = float_of_int (-n) /. float_of_int t_k in
            if t_k > 0 then (Float.max lb (Float.ceil bound), ub)
            else (lb, Float.min ub (Float.floor bound))
      in
      if lb > ub then unsatisfiable debug_s;
      d.term_bounds <- replace_assoc t (lb, ub) d.term_bounds
  | None ->
      let lead_t, lead_k = List.hd e.terms in
      let existing =
        Option.value (assoc_term lead_t d.expr_constraints) ~default:[]
      in
      let same (c1, k1, e1) (c2, k2, e2) =
        c1 = c2 && k1 = k2 && expr_cmp e1 e2 = 0
      in
      let entry = (cmp, lead_k, e) in
      if not (List.exists (same entry) existing) then
        d.expr_constraints <-
          replace_assoc lead_t (existing @ [ entry ]) d.expr_constraints

let sgn x = if x >= 0 then 1 else -1

let terms_of = function
  | Sym e -> e.terms
  | Const 0 -> []
  | Const k -> [ (term_one, k) ]

let combine_term_with_existing d t t_k ~only_smaller =
  let scope = d.dscope in
  let t_lb, t_ub =
    Option.value (assoc_term t d.term_bounds) ~default:(neg_infinity, infinity)
  in
  let term_expr k = Sym (make_expr [ (t, k) ] scope) in
  let from_bounds =
    if t_lb = t_ub then
      [ (Eq, sub (term_expr 1) (Const (int_of_float t_lb)), abs t_k, -sgn t_k) ]
    else
      (if t_lb > neg_infinity then
         [
           ( Geq,
             sub (term_expr 1) (Const (int_of_float t_lb)),
             abs t_k,
             -sgn t_k );
         ]
       else [])
      @
      if t_ub < infinity then
        [
          ( Geq,
            add (term_expr (-1)) (Const (int_of_float t_ub)),
            abs t_k,
            sgn t_k );
        ]
      else []
  in
  let groups =
    if only_smaller then
      [ Option.value (assoc_term t d.expr_constraints) ~default:[] ]
    else List.map snd d.expr_constraints
  in
  let from_constraints =
    List.concat_map
      (List.filter_map (fun (c_eq, _, c) ->
           match assoc_term t c.terms with
           | Some tc_k -> Some (c_eq, Sym c, abs t_k, -tc_k * sgn t_k)
           | None -> None))
      groups
  in
  from_bounds @ from_constraints

let combine_constraint_with_existing d eq e debug =
  let same (c1, e1) (c2, e2) = c1 = c2 && expr_cmp e1 e2 = 0 in
  let combos = ref [] in
  List.iter
    (fun (t, t_k) ->
      if not (is_constant_term t) then
        List.iter
          (fun (c_eq, c, c_s, t_s) ->
            if t_s > 0 || eq = Eq then
              let new_eq = if eq = Eq && c_eq = Eq then Eq else Geq in
              match linear_combination (Sym e) t_s c c_s e.scope with
              | new_e -> (
                  match to_constant new_e with
                  | Some const ->
                      if
                        (new_eq = Geq && const < 0)
                        || (new_eq = Eq && const <> 0)
                      then
                        invalid_arg
                          (Printf.sprintf "Unsatisfiable constraints: %s"
                             (match debug with
                             | Some s -> s
                             | None -> expr_to_string e ^ " >= 0"))
                  | None -> (
                      match new_e with
                      | Sym x ->
                          if not (List.exists (same (new_eq, x)) !combos) then
                            combos := !combos @ [ (new_eq, x) ]
                      | Const _ -> ())))
          (combine_term_with_existing d t t_k ~only_smaller:false))
    e.terms;
  !combos

type arg = F of float | D of dim

let arg_string = function
  | F f -> Printf.sprintf "%g" f
  | D d -> dim_to_string d

let combine_and_add d cmp e1 e2 debug =
  let skip =
    match (e1, e2, cmp) with
    | F f, _, _ when Float.is_nan f -> true
    | _, F f, _ when Float.is_nan f -> true
    | F f, _, Geq when f = infinity -> true
    | _, F f, Geq when f = neg_infinity -> true
    | _ -> false
  in
  if not skip then
    let to_dim = function F f -> Const (int_of_float f) | D x -> x in
    let e = sub (to_dim e1) (to_dim e2) in
    match to_constant e with
    | Some const ->
        if const < 0 then
          unsatisfiable
            (match debug with
            | Some s -> s
            | None -> arg_string e1 ^ " >= " ^ arg_string e2)
    | None -> (
        match e with
        | Sym x ->
            add_to_state d cmp x debug;
            List.iter
              (fun (c, a) -> add_to_state d c a None)
              (combine_constraint_with_existing d cmp x debug)
        | Const _ -> ())

let py_min = function
  | [] -> invalid_arg "py_min"
  | x :: rest -> List.fold_left (fun a b -> if b < a then b else a) x rest

let py_max = function
  | [] -> invalid_arg "py_max"
  | x :: rest -> List.fold_left (fun a b -> if b > a then b else a) x rest

let math_floor_with_inf a b =
  let opposite = a >= 0.0 <> (b >= 0.0) in
  if Float.is_finite b && Float.is_finite a then Float.floor (a /. b)
  else if Float.is_finite a then if opposite then -1.0 else 0.0
  else if opposite then neg_infinity
  else infinity

let rec bounds_for_sorted d terms prec =
  match terms with
  | [] -> (0.0, 0.0)
  | (t, t_k) :: _ when is_constant_term t -> (float_of_int t_k, float_of_int t_k)
  | (t, t_k) :: rest_terms -> (
      let lb = ref neg_infinity and ub = ref infinity in
      let done_early = ref None in
      List.iter
        (fun (c_eq, c, c_s, t_s) ->
          if !done_early = None then (
            let rest = linear_pairs term_cmp terms t_s (terms_of c) c_s in
            let rest_lb, rest_ub = bounds_for_sorted d rest Best in
            let ts = float_of_int t_s in
            if rest_ub < infinity then
              if t_s > 0 then ub := Float.min !ub (Float.floor (rest_ub /. ts))
              else lb := Float.max !lb (Float.ceil (rest_ub /. ts));
            if rest_lb > neg_infinity && c_eq = Eq then
              if t_s > 0 then lb := Float.max !lb (Float.ceil (rest_lb /. ts))
              else ub := Float.min !ub (Float.floor (rest_lb /. ts));
            if bounds_sufficient prec !lb !ub then done_early := Some (!lb, !ub)))
        (combine_term_with_existing d t t_k ~only_smaller:true);
      match !done_early with
      | Some r -> r
      | None -> (
          match t.factors with
          | [ ({ desc = Op ((("max" | "min") as op), [ op1; op2 ]); _ }, 1) ] ->
              let rest1 = linear_pairs term_cmp rest_terms 1 op1.terms t_k in
              let rest2 = linear_pairs term_cmp rest_terms 1 op2.terms t_k in
              let l1, u1 = bounds_for_sorted d rest1 Best in
              let l2, u2 = bounds_for_sorted d rest2 Best in
              let like_max = if op = "max" then t_k > 0 else t_k < 0 in
              if like_max then
                ( Float.max !lb (Float.max l1 l2),
                  Float.min !ub (Float.max u1 u2) )
              else
                ( Float.max !lb (Float.min l1 l2),
                  Float.min !ub (Float.min u1 u2) )
          | _ -> (!lb, !ub)))

let rec bounds d e prec ~add_implicit =
  match to_constant e with
  | Some k -> (float_of_int k, float_of_int k)
  | None -> (
      let x = match e with Sym x -> x | Const _ -> assert false in
      let scope = d.dscope in
      let cached =
        List.find_map
          (fun (k, v) -> if expr_cmp k x = 0 then Some v else None)
          scope.bounds_cache
      in
      match cached with
      | Some (lb, ub, prev)
        when bounds_sufficient prec lb ub || prec_rank prev >= prec_rank prec ->
          (lb, ub)
      | _ ->
          if add_implicit then add_implicit_expr d x;
          let lb, ub = bounds_for_sorted d x.terms prec in
          scope.bounds_cache <-
            (x, (lb, ub, prec))
            :: List.filter (fun (k, _) -> expr_cmp k x <> 0) scope.bounds_cache;
          (lb, ub))

and add_implicit_expr d e =
  List.iter
    (fun (t, _) -> if not (is_constant_term t) then add_implicit_term d t)
    e.terms

and add_implicit_dim d = function
  | Sym e -> add_implicit_expr d e
  | Const _ -> ()

and add_implicit_term d t =
  if not (List.exists (fun p -> term_cmp p t = 0) d.processed) then (
    d.processed <- t :: d.processed;
    let scope = d.dscope in
    let t_e = from_term t 1 scope in
    let bounds_best o = bounds d (Sym o) Best ~add_implicit:false in
    let geq a b = combine_and_add d Geq a b None in
    match t.factors with
    | [ (f, 1) ] -> (
        match f.desc with
        | Var _ -> geq (D t_e) (D (Const 1))
        | Op (op, operands) -> (
            List.iter (add_implicit_expr d) operands;
            let op1, op2 =
              match operands with
              | [ a; b ] -> (a, b)
              | _ -> invalid_arg "Shape_poly: operand arity"
            in
            match op with
            | "mod" ->
                let l, u =
                  bounds d (Sym op2) For_geq0_or_lt0 ~add_implicit:false
                in
                if l > 0.0 then (
                  geq (D t_e) (D (Const 0));
                  geq (D (sub (Sym op2) (Const 1))) (D t_e);
                  geq (F (u -. 1.0)) (D t_e))
                else if u < 0.0 then (
                  geq (D t_e) (D (add (Sym op2) (Const 1)));
                  geq (D t_e) (F (l +. 1.0));
                  geq (D (Const 0)) (D t_e))
            | "floordiv" ->
                let op1_l, op1_u = bounds_best op1 in
                let op2_l, op2_u = bounds_best op2 in
                if op2_l <= 0.0 && 0.0 <= op2_u then
                  raise
                    (inconclusive
                       (Printf.sprintf
                          "Possible division by 0 in division by %s"
                          (expr_to_string op2)));
                let cands =
                  [
                    math_floor_with_inf op1_l op2_l;
                    math_floor_with_inf op1_l op2_u;
                    math_floor_with_inf op1_u op2_l;
                    math_floor_with_inf op1_u op2_u;
                  ]
                in
                geq (D t_e) (F (py_min cands));
                geq (F (py_max cands)) (D t_e);
                if op2_l >= 0.0 then (
                  if op1_l >= 0.0 then geq (D t_e) (D (Const 0));
                  let mod_e = from_operation "mod" (Sym op1) (Sym op2) scope in
                  add_implicit_dim d mod_e;
                  let combined = add (mul (Sym op2) t_e) mod_e in
                  combine_and_add d Eq (D (Sym op1)) (D combined) None)
            | "max" ->
                let l1, u1 = bounds_best op1 in
                let l2, u2 = bounds_best op2 in
                geq (D t_e) (F (py_max [ l1; l2 ]));
                geq (F (py_max [ u1; u2 ])) (D t_e);
                geq (D t_e) (D (Sym op1));
                geq (D t_e) (D (Sym op2))
            | "min" ->
                let l1, u1 = bounds_best op1 in
                let l2, u2 = bounds_best op2 in
                geq (D t_e) (F (py_min [ l1; l2 ]));
                geq (F (py_min [ u1; u2 ])) (D t_e);
                geq (D (Sym op1)) (D t_e);
                geq (D (Sym op2)) (D t_e)
            | _ -> invalid_arg "Shape_poly: unknown operation"))
    | factors ->
        let per_factor =
          List.map
            (fun (f1, f1_exp) ->
              let f1_t = make_term [ (f1, 1) ] in
              let f1_e = from_term f1_t 1 scope in
              add_implicit_term d f1_t;
              let a_l, a_u = bounds d f1_e Best ~add_implicit:false in
              let p = float_of_int f1_exp in
              (Float.pow a_l p, Float.pow a_u p))
            factors
        in
        let cands =
          List.fold_left
            (fun acc (l, u) ->
              List.concat_map (fun x -> [ x *. l; x *. u ]) acc)
            [ 1.0 ] per_factor
        in
        geq (D t_e) (F (py_min cands));
        geq (F (py_max cands)) (D t_e))

let initialize d =
  List.iter
    (fun c ->
      (match c.diff with
      | Sym e when to_constant c.diff = None -> add_implicit_expr d e
      | _ -> ());
      combine_and_add d c.cmp (D c.diff) (D (Const 0)) (Some c.debug_str);
      d.dscope.bounds_cache <- [])
    d.dscope.explicit;
  d

let build_decision scope =
  if (not scope.initialized) || scope.explicit = [] then
    initialize (new_decision scope)
  else
    let base =
      match scope.decision_initial with
      | Some s -> s
      | None ->
          let s = initialize (new_decision scope) in
          scope.decision_initial <- Some s;
          s
    in
    { base with dscope = scope }

let bounds_decision e prec =
  match e with
  | Const k -> (float_of_int k, float_of_int k)
  | Sym x -> bounds (build_decision x.scope) e prec ~add_implicit:true

let max_expr a b ~reversed =
  let lb, ub = bounds_decision (sub a b) For_geq0_or_leq0 in
  if 0.0 <= lb then a
  else if ub <= 0.0 then b
  else
    let scope = match scope_of a b with Some s -> s | None -> assert false in
    if reversed then from_operation "max" b a scope
    else from_operation "max" a b scope

let min_expr a b ~reversed =
  let lb, ub = bounds_decision (sub a b) For_geq0_or_leq0 in
  if 0.0 <= lb then b
  else if ub <= 0.0 then a
  else
    let scope = match scope_of a b with Some s -> s | None -> assert false in
    if reversed then from_operation "min" b a scope
    else from_operation "min" a b scope

let max_dim a b =
  match (a, b) with
  | Const x, Const y -> Const (Stdlib.max x y)
  | Const _, Sym _ -> max_expr b a ~reversed:true
  | Sym _, _ -> max_expr a b ~reversed:false

let min_dim a b =
  match (a, b) with
  | Const x, Const y -> Const (Stdlib.min x y)
  | Const _, Sym _ -> min_expr b a ~reversed:true
  | Sym _, _ -> min_expr a b ~reversed:false

let geq_decision e1 e2 cmp_str =
  match (e1, e2) with
  | Const a, Const b -> a >= b
  | _ ->
      (match (e1, e2) with
      | Sym a, Sym b ->
          check_same_scope a.scope b ~when_:("when comparing " ^ cmp_str)
      | _ -> ());
      let scope =
        match scope_of e1 e2 with Some s -> s | None -> assert false
      in
      let lb, ub = bounds_decision (sub e1 e2) For_geq0_or_lt0 in
      if lb >= 0.0 then true
      else if ub < 0.0 then false
      else
        let describe =
          if scope.explicit <> [] then
            "\nUsing symbolic scope " ^ scope_to_string scope
          else ""
        in
        raise
          (inconclusive
             (Printf.sprintf
                "Symbolic dimension comparison %s is inconclusive.%s" cmp_str
                describe))

let ge a b =
  geq_decision a b
    (Printf.sprintf "'%s' >= '%s'" (dim_to_string a) (dim_to_string b))

let le a b =
  geq_decision b a
    (Printf.sprintf "'%s' <= '%s'" (dim_to_string a) (dim_to_string b))

let gt a b =
  not
    (geq_decision b a
       (Printf.sprintf "'%s' > '%s'" (dim_to_string a) (dim_to_string b)))

let lt a b =
  not
    (geq_decision a b
       (Printf.sprintf "'%s' < '%s'" (dim_to_string a) (dim_to_string b)))

let equal a b =
  match (a, b) with
  | Const x, Const y -> x = y
  | Sym x, Sym y when x.scope != y.scope -> false
  | _ -> ( match to_constant (sub a b) with Some 0 -> true | _ -> false)

let vars_of_dim d =
  let module S = Set.Make (String) in
  let rec of_expr e =
    List.fold_left
      (fun acc (t, _) ->
        List.fold_left
          (fun acc (f, _) -> S.union acc (of_factor f))
          acc t.factors)
      S.empty e.terms
  and of_factor f =
    match f.desc with
    | Var v -> S.singleton v
    | Op (_, operands) ->
        List.fold_left (fun acc o -> S.union acc (of_expr o)) S.empty operands
  in
  match d with Const _ -> [] | Sym e -> S.elements (of_expr e)

let rec evaluate_expr env e =
  List.fold_left
    (fun acc (t, k) -> acc + (evaluate_term env e.scope t * k))
    0 e.terms

and evaluate_term env scope t =
  List.fold_left
    (fun acc (f, e) ->
      let v = evaluate_factor env scope f in
      let rec p i = if i = 0 then 1 else v * p (i - 1) in
      acc * p e)
    1 t.factors

and evaluate_factor env scope f =
  match f.desc with
  | Var v -> (
      match List.assoc_opt v env with
      | Some x -> x
      | None -> (
          match from_var v scope with
          | Const k -> k
          | Sym n -> (
              match n.terms with
              | [ ({ factors = [ ({ desc = Var w; _ }, 1) ]; _ }, 1) ]
                when w = v ->
                  raise
                    (Unexpected_dim_var
                       (Printf.sprintf
                          "Encountered dimension variable '%s' that is not \
                           appearing in the shapes of the function arguments.\n\
                           The following dimension variables are appearing in \
                           the shapes of the function arguments: [%s].\n\
                           Please see \
                           https://docs.jax.dev/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes \
                           for more details."
                          v
                          (String.concat ", "
                             (List.map (fun (k, _) -> "'" ^ k ^ "'") env))))
              | _ -> evaluate_expr env n)))
  | Op (op, operands) -> (
      let values = List.map (evaluate_expr env) operands in
      match (op, values) with
      | "floordiv", [ a; b ] -> floordiv a b
      | "mod", [ a; b ] -> pymod a b
      | "max", [ a; b ] -> Stdlib.max a b
      | "min", [ a; b ] -> Stdlib.min a b
      | _ -> invalid_arg "Shape_poly: unknown operation")

let evaluate env = function Const k -> k | Sym e -> evaluate_expr env e

type token_kind =
  | Lpar
  | Rpar
  | Comma
  | Plus
  | Minus
  | Star
  | Circumflex
  | Ellipsis_tok
  | Name
  | Number
  | Other of string
  | Endmarker

type token = { kind : token_kind; text : string; col : int; line : string }

let token_name = function
  | Lpar -> "LPAR"
  | Rpar -> "RPAR"
  | Comma -> "COMMA"
  | Plus -> "PLUS"
  | Minus -> "MINUS"
  | Star -> "STAR"
  | Circumflex -> "CIRCUMFLEX"
  | Ellipsis_tok -> "ELLIPSIS"
  | Name -> "NAME"
  | Number -> "NUMBER"
  | Other s -> s
  | Endmarker -> "ENDMARKER"

let two_char_ops =
  [
    ("**", "DOUBLESTAR");
    ("//", "DOUBLESLASH");
    (">=", "GREATEREQUAL");
    ("<=", "LESSEQUAL");
    ("==", "EQEQUAL");
    ("!=", "NOTEQUAL");
  ]

let tokenize s =
  let n = String.length s in
  let is_digit c = c >= '0' && c <= '9' in
  let is_alpha c =
    (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'
  in
  let depth = ref 0 in
  let rec go i acc =
    if i >= n then
      let tail =
        if !depth > 0 then
          [
            { kind = Endmarker; text = ""; col = 0; line = "" };
            { kind = Other "NL"; text = ""; col = n; line = s };
          ]
        else [ { kind = Endmarker; text = ""; col = 0; line = "" } ]
      in
      List.rev (tail @ acc)
    else
      let c = s.[i] in
      let tok kind len =
        go (i + len)
          ({ kind; text = String.sub s i len; col = i; line = s } :: acc)
      in
      let span pred =
        let j = ref i in
        while !j < n && pred s.[!j] do
          incr j
        done;
        !j - i
      in
      if c = ' ' || c = '\t' || c = '\n' then go (i + 1) acc
      else if is_alpha c then
        tok Name (span (fun c -> is_alpha c || is_digit c))
      else if is_digit c then tok Number (span (fun c -> is_digit c || c = '.'))
      else if i + 3 <= n && String.sub s i 3 = "..." then tok Ellipsis_tok 3
      else if i + 2 <= n && List.mem_assoc (String.sub s i 2) two_char_ops then
        tok (Other (List.assoc (String.sub s i 2) two_char_ops)) 2
      else
        match c with
        | '(' ->
            incr depth;
            tok Lpar 1
        | ')' ->
            decr depth;
            tok Rpar 1
        | ',' -> tok Comma 1
        | '+' -> tok Plus 1
        | '-' -> tok Minus 1
        | '*' -> tok Star 1
        | '^' -> tok Circumflex 1
        | '/' -> tok (Other "SLASH") 1
        | '%' -> tok (Other "PERCENT") 1
        | '.' -> tok (Other "DOT") 1
        | ':' -> tok (Other "COLON") 1
        | '>' -> tok (Other "GREATER") 1
        | '<' -> tok (Other "LESS") 1
        | '=' -> tok (Other "EQUAL") 1
        | _ -> tok (Other "ERRORTOKEN") 1
  in
  go 0 []

let py_repr s = "'" ^ s ^ "'"

let like_repr like =
  let item = function None -> "None" | Some k -> string_of_int k in
  match Array.to_list like with
  | [ x ] -> "(" ^ item x ^ ",)"
  | l -> "(" ^ String.concat ", " (List.map item l) ^ ")"

type parser = {
  spec_repr : string;
  like : int option array option;
  pscope : scope;
  mutable toks : token list;
  mutable dims : dim list;
}

let parse_err p tok detail =
  let n = List.length p.dims in
  let tail =
    match tok with
    | Some t ->
        Printf.sprintf "Parsed '%s', remaining '%s'."
          (String.sub t.line 0 t.col)
          (String.sub t.line t.col (String.length t.line - t.col))
    | None -> ""
  in
  Invalid_argument
    (Printf.sprintf "syntax error in symbolic shape %s in dimension %d: %s. %s"
       p.spec_repr n detail tail)

let next_tok p =
  match p.toks with
  | t :: rest ->
      p.toks <- rest;
      t
  | [] -> raise (parse_err p None "unexpected end of string")

let expect p tok expected =
  if not (List.mem tok.kind expected) then
    raise
      (parse_err p (Some tok)
         (Printf.sprintf "expecting one of {%s} but found %s"
            (String.concat ", " (List.map token_name expected))
            (token_name tok.kind)))

let consume p tok expected =
  expect p tok [ expected ];
  next_tok p

let integer p tok =
  expect p tok [ Number ];
  match int_of_string_opt tok.text with
  | Some v when String.for_all (fun c -> c >= '0' && c <= '9') tok.text ->
      (v, next_tok p)
  | _ -> raise (parse_err p (Some tok) ("expecting integer, found " ^ tok.text))

let add_dim p expr tok =
  let like_text () =
    match p.like with Some l -> like_repr l | None -> "None"
  in
  let d =
    match expr with
    | Some d -> d
    | None ->
        raise
          (parse_err p (Some tok)
             ("unexpected placeholder for unknown dimension; like="
            ^ like_text ()))
  in
  (match (d, p.like) with
  | Const k, Some l ->
      let idx = List.length p.dims in
      if idx >= Array.length l then invalid_arg "tuple index out of range";
      if Some k <> l.(idx) then
        raise
          (parse_err p (Some tok)
             (Printf.sprintf "different size %d for known dimension; like=%s" k
                (like_text ())))
  | _ -> ());
  p.dims <- p.dims @ [ d ]

let follow_shape = [ Endmarker; Rpar ]
let follow_expr = Comma :: follow_shape
let follow_term = Plus :: Minus :: follow_expr

let rec parse_shape p tok =
  if tok.kind = Lpar then
    let tok = parse_shape p (next_tok p) in
    consume p tok Rpar
  else
    let rec loop tok =
      if List.mem tok.kind follow_shape then tok
      else
        let is_placeholder = tok.kind = Name && tok.text = "_" in
        if tok.kind = Ellipsis_tok || is_placeholder then (
          let like =
            match p.like with
            | Some l -> l
            | None ->
                raise
                  (parse_err p (Some tok)
                     "spec contains ... but no 'like' shape was given")
          in
          let parsed = List.length p.dims in
          let min_len =
            if tok.kind = Ellipsis_tok then parsed else parsed + 1
          in
          if Array.length like < min_len then
            raise
              (parse_err p (Some tok)
                 (Printf.sprintf
                    "cannot resolve placeholder '%s' because we parsed %d \
                     already and 'like' shape has only %d dimensions"
                    tok.text parsed (Array.length like)));
          if tok.kind = Ellipsis_tok then (
            for i = parsed to Array.length like - 1 do
              add_dim p (Option.map of_int like.(i)) tok
            done;
            next_tok p)
          else
            let e = like.(parsed) in
            let tok = next_tok p in
            after_dim (Option.map of_int e) tok)
        else
          let e, tok = parse_expr p tok in
          after_dim (Some e) tok
    and after_dim e tok =
      add_dim p e tok;
      if List.mem tok.kind follow_shape then tok else loop (consume p tok Comma)
    in
    loop tok

and parse_expr p tok =
  let negated, tok =
    if tok.kind = Minus then (true, next_tok p)
    else if tok.kind = Plus then (false, next_tok p)
    else (false, tok)
  in
  let rec loop negated acc tok =
    let t, tok = parse_term p tok in
    let signed = if negated then neg t else t in
    let acc = match acc with Some a -> add a signed | None -> signed in
    if List.mem tok.kind follow_expr then (acc, tok)
    else
      let negated = tok.kind = Minus in
      expect p tok [ Plus; Minus ];
      loop negated (Some acc) (next_tok p)
  in
  loop negated None tok

and parse_term p tok =
  let rec loop acc tok =
    let f, tok = parse_factor p tok in
    let f, tok =
      if tok.kind = Circumflex then (
        let tok = next_tok p in
        expect p tok [ Number ];
        let power, tok = integer p tok in
        (pow f power, tok))
      else (f, tok)
    in
    let acc = match acc with Some a -> mul a f | None -> f in
    if List.mem tok.kind follow_term then (acc, tok)
    else loop (Some acc) (consume p tok Star)
  in
  loop None tok

and parse_factor p tok =
  match tok.kind with
  | Name -> (
      match tok.text with
      | ("mod" | "floordiv" | "max" | "min") as op ->
          parse_binary p op (next_tok p)
      | v -> (from_var v p.pscope, next_tok p))
  | Minus ->
      let tok = next_tok p in
      expect p tok [ Number ];
      let v, tok = integer p tok in
      (Const (-v), tok)
  | Number ->
      let v, tok = integer p tok in
      (Const v, tok)
  | _ ->
      expect p tok [ Name; Minus; Number ];
      assert false

and parse_binary p op tok =
  let tok = consume p tok Lpar in
  let e1, tok = parse_expr p tok in
  let tok = consume p tok Comma in
  let e2, tok = parse_expr p tok in
  let tok = consume p tok Rpar in
  match op with
  | "max" -> (max_dim e1 e2, tok)
  | "min" -> (min_dim e1 e2, tok)
  | _ -> (from_operation op e1 e2 p.pscope, tok)

let run_parser spec like spec_repr scope =
  let p =
    { spec_repr; like; pscope = scope; toks = tokenize spec; dims = [] }
  in
  let tok = parse_shape p (next_tok p) in
  expect p tok [ Endmarker ];
  Array.of_list p.dims

let find_sub s sub =
  let n = String.length s and m = String.length sub in
  let rec go i =
    if i + m > n then None
    else if String.sub s i m = sub then Some i
    else go (i + 1)
  in
  go 0

let parse_single text scope =
  match run_parser text None (py_repr text) scope with
  | [| d |] -> d
  | [||] -> invalid_arg "not enough values to unpack (expected 1, got 0)"
  | _ -> invalid_arg "too many values to unpack (expected 1)"

let rec process_explicit scope (c : sym_constraint) =
  match to_constant c.diff with
  | Some const ->
      if (c.cmp = Eq && const <> 0) || (c.cmp = Geq && const < 0) then
        invalid_arg ("Unsatisfiable explicit constraint: " ^ c.debug_str)
  | None ->
      (if c.cmp = Eq then
         let invalid () =
           invalid_arg
             "Invalid equality constraint: {e1} == {e2}. The left-hand-side \
              must be of the form `term * coefficient`."
         in
         match c.e1 with
         | Const _ -> invalid ()
         | Sym e1 -> (
             match e1.terms with
             | [ (before, before_k) ] -> (
                 let after = ensure_poly c.e2 "parse_constraint" e1.scope in
                 if assoc_term before scope.rules <> None then
                   failwith
                     ("Found multiple equality constraints with the same \
                       left-hand-side: " ^ term_to_string before);
                 scope.rules <- scope.rules @ [ (before, (after, before_k)) ];
                 match (before_k, c.e2, before.factors) with
                 | 1, Const 0, [ ({ desc = Op ("mod", [ op1; op2 ]); _ }, 1) ]
                   -> (
                     match to_constant (Sym op2) with
                     | Some k2 ->
                         let fd = factor_op "floordiv" [ op1; op2 ] in
                         let k_times =
                           from_term (make_term [ (fd, 1) ]) k2 e1.scope
                         in
                         process_explicit scope
                           {
                             cmp = Eq;
                             e1 = k_times;
                             e2 = Sym op1;
                             diff = sub k_times (Sym op1);
                             debug_str =
                               Printf.sprintf "%s == %s" (dim_to_string k_times)
                                 (expr_to_string op1);
                           }
                     | None -> ())
                 | _ -> ())
             | _ -> invalid ()));
      scope.explicit <- scope.explicit @ [ c ]

let parse_constraint scope c_str =
  let cmp_pos, cmp, is_geq =
    match find_sub c_str "==" with
    | Some i -> (i, Eq, true)
    | None -> (
        match find_sub c_str ">=" with
        | Some i -> (i, Geq, true)
        | None -> (
            match find_sub c_str "<=" with
            | Some i -> (i, Geq, false)
            | None ->
                invalid_arg
                  "Constraint parsing error: must contain one of '==' or '>=' \
                   or '<='"))
  in
  let e1_str = String.sub c_str 0 cmp_pos in
  let e2_str =
    String.sub c_str (cmp_pos + 2) (String.length c_str - cmp_pos - 2)
  in
  let e1 = parse_single e1_str scope in
  let e2 = parse_single e2_str scope in
  let e1, e2 = if cmp = Geq && not is_geq then (e2, e1) else (e1, e2) in
  process_explicit scope { debug_str = c_str; cmp; e1; e2; diff = sub e1 e2 }

let scope_counter = ref 0

let symbolic_scope ?(constraints = []) () =
  incr scope_counter;
  let scope =
    {
      sid = !scope_counter;
      rules = [];
      explicit = [];
      initialized = false;
      bounds_cache = [];
      decision_initial = None;
    }
  in
  List.iter
    (fun c ->
      parse_constraint scope c;
      scope.bounds_cache <- [])
    constraints;
  scope.initialized <- true;
  scope

let resolve_scope ?(constraints = []) ?scope ~message () =
  match scope with
  | None -> symbolic_scope ~constraints ()
  | Some s ->
      if constraints <> [] then invalid_arg message;
      s

let symbolic_shape ?constraints ?scope ?like shape_spec =
  let scope =
    resolve_scope ?constraints ?scope
      ~message:"Cannot specify both a `scope` and `constraints`." ()
  in
  let spec, spec_repr =
    match shape_spec with None -> ("...", "None") | Some s -> (s, py_repr s)
  in
  run_parser spec like spec_repr scope

let symbolic_args_specs ?constraints ?scope args shapes_specs =
  let scope =
    resolve_scope ?constraints ?scope
      ~message:"Cannot use both `scope` and `constraints`" ()
  in
  let shapes_specs =
    match (args, shapes_specs) with
    | Tree_util.Tuple _, Tree_util.List l -> Tree_util.Tuple l
    | _ -> shapes_specs
  in
  let specs =
    try Tree_util.tree_broadcast shapes_specs args
    with Invalid_argument m | Failure m ->
      invalid_arg
        ("pytree structure error in export.symbolic_args_specs shapes_specs: "
       ^ m)
  in
  Tree_util.tree_map2
    (fun spec (a : Types.aval) ->
      {
        shape =
          symbolic_shape ~scope spec ~like:(Array.map (fun d -> Some d) a.shape);
        dtype = a.dtype;
      })
    specs args

let all_dim_vars specs =
  let module S = Set.Make (String) in
  S.elements
    (List.fold_left
       (fun acc s ->
         Array.fold_left
           (fun acc d -> S.union acc (S.of_list (vars_of_dim d)))
           acc s.shape)
       S.empty specs)

let shape_to_string shape =
  match Array.to_list shape with
  | [ d ] -> "(" ^ dim_to_string d ^ ",)"
  | l -> "(" ^ String.concat ", " (List.map dim_to_string l) ^ ")"

let int_shape_to_string shape =
  shape_to_string (Array.map (fun k -> Const k) shape)

let trailer_errors =
  ". Please see \
   https://docs.jax.dev/en/latest/export/shape_poly.html#shape-assertion-errors \
   for more details."

let solve_dim_vars ?arg_names specs (shapes : int array list) =
  let names =
    match arg_names with
    | Some n -> n
    | None -> List.mapi (fun i _ -> Printf.sprintf "args[%d]" i) specs
  in
  let triples = List.combine (List.combine names specs) shapes in
  let poly_specs =
    List.filter_map
      (fun ((name, s), _) ->
        if Array.exists is_symbolic_dim s.shape then
          Some (name ^ ".shape = " ^ shape_to_string s.shape)
        else None)
      triples
  in
  let poly_err =
    " Using the following polymorphic shapes specifications: "
    ^ String.concat "," poly_specs
    ^ "."
  in
  let eqns =
    List.concat_map
      (fun ((name, s), actual) ->
        List.concat
          (List.mapi
             (fun i d ->
               match d with
               | Sym e ->
                   [ (e, Printf.sprintf "%s.shape[%d]" name i, actual.(i)) ]
               | Const _ -> [])
             (Array.to_list s.shape)))
      triples
  in
  let env = ref [] in
  let solution = Buffer.create 64 in
  Buffer.add_string solution " Obtained dimension variables: ";
  let failures = ref [] in
  let add_check ok message =
    if not ok then failures := !failures @ [ message ]
  in
  let process (e, dim_name, actual) =
    let rec scan terms var value =
      match terms with
      | [] -> Some (var, value)
      | (t, k) :: rest -> (
          match evaluate_term !env e.scope t with
          | v -> scan rest var (value - (v * k))
          | exception Unexpected_dim_var _ -> (
              match (t.factors, var) with
              | [ ({ desc = Var v; _ }, 1) ], None ->
                  scan rest (Some (v, k)) value
              | _ -> None))
    in
    match scan e.terms None actual with
    | None -> false
    | Some (Some (var, var_k), dim_value) ->
        let var_value =
          if var_k = 1 then dim_value
          else
            let q = floordiv dim_value var_k and r = pymod dim_value var_k in
            add_check (r = 0)
              (Printf.sprintf
                 "Input shapes do not match the polymorphic shapes \
                  specification. Division had remainder %d when computing the \
                  value of '%s'.%s%s%s"
                 r var poly_err (Buffer.contents solution) trailer_errors);
            q
        in
        env := !env @ [ (var, var_value) ];
        Buffer.add_string solution
          (Printf.sprintf
             "'%s' = %d from specification '%s' for dimension %s (= %d), " var
             var_value (expr_to_string e) dim_name actual);
        add_check (var_value >= 1)
          (Printf.sprintf
             "Input shapes do not match the polymorphic shapes specification. \
              Expected value >= 1 for dimension variable '%s'.%s%s%s"
             var poly_err (Buffer.contents solution) trailer_errors);
        true
    | Some (None, _) ->
        let spec_value = evaluate_expr !env e in
        add_check (actual = spec_value)
          (Printf.sprintf
             "Input shapes do not match the polymorphic shapes specification. \
              Found inconsistency between dimension size %s (= %d) and the \
              specification '%s' (= %d).%s%s%s"
             dim_name actual (expr_to_string e) spec_value poly_err
             (Buffer.contents solution) trailer_errors);
        true
  in
  let scope = match eqns with (e, _, _) :: _ -> Some e.scope | [] -> None in
  let rec solve eqns =
    let nr = List.length eqns in
    let remaining = List.filter (fun eqn -> not (process eqn)) eqns in
    if remaining = [] then ()
    else if List.length remaining >= nr then
      let module S = Set.Make (String) in
      let vars =
        List.fold_left
          (fun acc (e, _, _) -> S.union acc (S.of_list (vars_of_dim (Sym e))))
          S.empty remaining
      in
      let unsolved = S.elements (S.diff vars (S.of_list (List.map fst !env))) in
      invalid_arg
        (Printf.sprintf
           "Cannot solve for values of dimension variables {%s}. We can only \
            solve linear uni-variate constraints.%s Unprocessed \
            specifications: %s. Please see \
            https://docs.jax.dev/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes \
            for more details."
           (String.concat ", " (List.map py_repr unsolved))
           poly_err
           (String.concat ", "
              (List.map
                 (fun (e, name, _) ->
                   Printf.sprintf "'%s' for dimension size %s"
                     (expr_to_string e) name)
                 remaining)))
    else solve remaining
  in
  solve eqns;
  (match scope with
  | Some scope when !env <> [] ->
      List.iter
        (fun (c : sym_constraint) ->
          let value = evaluate !env c.diff in
          let ok = if c.cmp = Eq then value = 0 else value >= 0 in
          add_check ok
            (Printf.sprintf
               "Input shapes do not match the symbolic shape constraint %s. \
                Expected '%s' to be %s to 0, but found %d. %s%s%s"
               c.debug_str (dim_to_string c.diff)
               (if c.cmp = Geq then "greater or equal" else "equal")
               value poly_err (Buffer.contents solution) trailer_errors))
        scope.explicit
  | _ -> ());
  (match !failures with m :: _ -> invalid_arg m | [] -> ());
  !env

let scope_of_dim = function Sym e -> Some e.scope | Const _ -> None

let check_scopes ~when_ specs =
  let current = ref None in
  List.iteri
    (fun arg_idx s ->
      Array.iter
        (fun d ->
          match (scope_of_dim d, d) with
          | Some scope, Sym e -> (
              let descr = Printf.sprintf "args[%d]" arg_idx in
              match !current with
              | None -> current := Some (scope, descr)
              | Some (first, first_descr) ->
                  check_same_scope first e ~when_
                    ~self_descr:
                      (Printf.sprintf "current (from %s) " first_descr)
                    ~other_descr:descr)
          | _ -> ())
        s.shape)
    specs
