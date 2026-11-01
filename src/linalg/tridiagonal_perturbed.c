#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "lapack_shim.h"

static float complex divide_c(float complex numerator, float complex divisor) {
  float a = crealf(numerator), b = cimagf(numerator);
  float c = crealf(divisor), d = cimagf(divisor);
  float re, im;
  if (fabsf(c) < fabsf(d)) {
    float ratio = c / d;
    float denominator = fmaf(c, ratio, d);
    re = fmaf(a, ratio, b) / denominator;
    im = fmaf(b, ratio, -a) / denominator;
  } else {
    float ratio = d / c;
    float denominator = fmaf(d, ratio, c);
    re = fmaf(b, ratio, a) / denominator;
    im = fmaf(-a, ratio, b) / denominator;
  }
  if (isnan(re) && isnan(im)) {
    if (c == 0.0f && d == 0.0f && (!isnan(a) || !isnan(b))) {
      re = copysignf(HUGE_VALF, c) * a;
      im = copysignf(HUGE_VALF, c) * b;
    } else if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {
      a = copysignf(isinf(a) ? 1.0f : 0.0f, a);
      b = copysignf(isinf(b) ? 1.0f : 0.0f, b);
      re = HUGE_VALF * (a * c + b * d);
      im = HUGE_VALF * (b * c - a * d);
    } else if ((isinf(c) || isinf(d)) && isfinite(a) && isfinite(b)) {
      c = copysignf(isinf(c) ? 1.0f : 0.0f, c);
      d = copysignf(isinf(d) ? 1.0f : 0.0f, d);
      re = 0.0f * (a * c + b * d);
      im = 0.0f * (b * c - a * d);
    }
  }
  float complex out;
  __real__ out = re;
  __imag__ out = im;
  return out;
}

#define DIVIDE_S(a, b) ((a) / (b))
#define DIVIDE_D(a, b) ((a) / (b))
#define DIVIDE_Z(a, b) ((a) / (b))

#define DEFINE_SOLVER(NAME, SCALAR, REAL, ABS, RE, COPYSIGN, FMA_SUB, DIVIDE,  \
                      TINY, HUGE_REAL, EPS)                                    \
  static void NAME##_perturb_pivot(REAL perturb, SCALAR *pivot, SCALAR *row,   \
                                   int k_rhs) {                                \
    const REAL one = 1;                                                        \
    const REAL small = one / HUGE_REAL;                                        \
    const REAL safemin = small < TINY ? TINY : small * (one + EPS);            \
    const REAL bignum = one / safemin;                                         \
    REAL abs_pivot = ABS(*pivot);                                              \
    if (abs_pivot >= one) return;                                              \
    REAL magnitude = fabs((double)perturb) > (double)safemin                   \
                         ? (REAL)fabs((double)perturb)                         \
                         : safemin;                                            \
    perturb = COPYSIGN(magnitude, RE(*pivot));                                 \
    int stop = 0;                                                              \
    REAL max_factor = ABS(row[0]);                                             \
    for (int j = 1; j < k_rhs; j++) {                                          \
      REAL next = ABS(row[j]);                                                 \
      if (next > max_factor) max_factor = next;                                \
    }                                                                          \
    while (abs_pivot < one && !stop) {                                         \
      if (abs_pivot < safemin) {                                               \
        if (abs_pivot == (REAL)0 || max_factor * safemin > abs_pivot) {        \
          *pivot += perturb;                                                   \
          perturb *= (REAL)2;                                                  \
        } else {                                                               \
          *pivot *= bignum;                                                    \
          for (int j = 0; j < k_rhs; j++) row[j] *= bignum;                    \
          stop = 1;                                                            \
        }                                                                      \
      } else if (max_factor > abs_pivot * bignum) {                            \
        *pivot += perturb;                                                     \
        perturb *= (REAL)2;                                                    \
      } else {                                                                 \
        stop = 1;                                                              \
      }                                                                        \
      abs_pivot = ABS(*pivot);                                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void NAME(int n, int k_rhs, const SCALAR *sub, const SCALAR *diag,    \
                   const SCALAR *super, const SCALAR *rhs, SCALAR *x) {        \
    if (n == 0) return;                                                        \
    SCALAR *row = calloc((size_t)(k_rhs > 0 ? k_rhs : 1), sizeof(SCALAR));     \
    if (n == 1) {                                                              \
      SCALAR p = diag[0];                                                      \
      REAL perturb = EPS * ABS(p);                                             \
      for (int j = 0; j < k_rhs; j++) row[j] = rhs[j];                         \
      NAME##_perturb_pivot(perturb, &p, row, k_rhs);                           \
      for (int j = 0; j < k_rhs; j++) x[j] = DIVIDE(row[j], p);                     \
      free(row);                                                               \
      return;                                                                  \
    }                                                                          \
    SCALAR *u = calloc((size_t)n * 3, sizeof(SCALAR));                         \
    SCALAR *u0 = u, *u1 = u + n, *u2 = u + 2 * n;                              \
    u0[0] = diag[0];                                                           \
    u1[0] = super[0];                                                          \
    REAL max_abs_u = ABS(u0[0]) < ABS(u1[0]) ? ABS(u1[0]) : ABS(u0[0]);        \
    REAL scale1 = ABS(u0[0]) + ABS(u1[0]);                                     \
    for (int j = 0; j < k_rhs; j++) x[j * n] = rhs[j * n];                     \
    for (int k = 0; k < n - 1; ++k) {                                          \
      u0[k + 1] = diag[k + 1];                                                 \
      REAL scale2 = ABS(sub[k + 1]) + ABS(u0[k + 1]);                          \
      if (k < n - 2) scale2 += ABS(super[k + 1]);                              \
      if (sub[k + 1] == (SCALAR)0) {                                           \
        scale1 = scale2;                                                       \
        u1[k + 1] = super[k + 1];                                              \
        u2[k] = 0;                                                             \
        for (int j = 0; j < k_rhs; j++) x[k + 1 + j * n] = rhs[k + 1 + j * n]; \
      } else {                                                                 \
        const REAL piv1 =                                                      \
            u0[k] == (SCALAR)0 ? (REAL)0 : ABS(u0[k]) / scale1;                \
        const REAL piv2 = ABS(sub[k + 1]) / scale2;                            \
        if (piv2 <= piv1) {                                                    \
          scale1 = scale2;                                                     \
          const SCALAR factor = DIVIDE(sub[k + 1], u0[k]);                       \
          u0[k + 1] = FMA_SUB(u0[k + 1], factor, u1[k]);                       \
          u1[k + 1] = super[k + 1];                                            \
          u2[k] = 0;                                                           \
          for (int j = 0; j < k_rhs; j++) {                                    \
            SCALAR product = factor * x[k + j * n];                            \
            x[k + 1 + j * n] = rhs[k + 1 + j * n] - product;                   \
          }                                                                    \
        } else {                                                               \
          const SCALAR factor = DIVIDE(u0[k], sub[k + 1]);                       \
          const SCALAR utmp = u1[k];                                           \
          u0[k] = sub[k + 1];                                                  \
          u0[k + 1] = FMA_SUB(utmp, factor, diag[k + 1]);                      \
          u1[k] = diag[k + 1];                                                 \
          if (k < n - 2) {                                                     \
            u2[k] = super[k + 1];                                              \
            u1[k + 1] = -factor * super[k + 1];                                \
          }                                                                    \
          for (int j = 0; j < k_rhs; j++) {                                    \
            SCALAR product = factor * rhs[k + 1 + j * n];                      \
            x[k + 1 + j * n] = x[k + j * n] - product;                         \
            x[k + j * n] = rhs[k + 1 + j * n];                                 \
          }                                                                    \
          scale1 = ABS(u0[k + 1]) + ABS(u1[k + 1]);                            \
        }                                                                      \
      }                                                                        \
      if (k < n - 2) {                                                         \
        if (max_abs_u < ABS(u0[k])) max_abs_u = ABS(u0[k]);                    \
        if (max_abs_u < ABS(u1[k])) max_abs_u = ABS(u1[k]);                    \
        if (max_abs_u < ABS(u2[k])) max_abs_u = ABS(u2[k]);                    \
      }                                                                        \
    }                                                                          \
    if (max_abs_u < ABS(u0[n - 1])) max_abs_u = ABS(u0[n - 1]);                \
    REAL perturb = EPS * max_abs_u;                                            \
    SCALAR p = u0[n - 1];                                                      \
    for (int j = 0; j < k_rhs; j++) row[j] = x[n - 1 + j * n];                 \
    NAME##_perturb_pivot(perturb, &p, row, k_rhs);                             \
    SCALAR inverse = DIVIDE((SCALAR)1, p);                                       \
    for (int j = 0; j < k_rhs; j++) x[n - 1 + j * n] = row[j] * inverse;       \
    p = u0[n - 2];                                                             \
    for (int j = 0; j < k_rhs; j++) {                                          \
      SCALAR product = u1[n - 2] * x[n - 1 + j * n];                           \
      row[j] = x[n - 2 + j * n] - product;                                     \
    }                                                                          \
    NAME##_perturb_pivot(COPYSIGN(perturb, RE(p)), &p, row, k_rhs);            \
    inverse = DIVIDE((SCALAR)1, p);                                                 \
    for (int j = 0; j < k_rhs; j++) x[n - 2 + j * n] = row[j] * inverse;       \
    for (int k = n - 3; k >= 0; --k) {                                         \
      p = u0[k];                                                               \
      for (int j = 0; j < k_rhs; j++) {                                        \
        SCALAR first = u1[k] * x[k + 1 + j * n];                               \
        SCALAR difference = x[k + j * n] - first;                              \
        SCALAR second = u2[k] * x[k + 2 + j * n];                              \
        row[j] = difference - second;                                          \
      }                                                                        \
      NAME##_perturb_pivot(COPYSIGN(perturb, RE(p)), &p, row, k_rhs);          \
      inverse = DIVIDE((SCALAR)1, p);                                               \
      for (int j = 0; j < k_rhs; j++) x[k + j * n] = row[j] * inverse;         \
    }                                                                          \
    free(u);                                                                   \
    free(row);                                                                 \
  }

#define IDENTITY(v) (v)
#define FUSED_F(a, f, b) fmaf(-(f), (b), (a))
#define FUSED_D(a, f, b) fma(-(f), (b), (a))
#define PLAIN(a, f, b) ((a) - (f) * (b))

DEFINE_SOLVER(solve_s, float, float, fabsf, IDENTITY, copysignf, FUSED_F,
              DIVIDE_S, FLT_MIN, FLT_MAX, FLT_EPSILON)
DEFINE_SOLVER(solve_d, double, double, fabs, IDENTITY, copysign, FUSED_D,
              DIVIDE_D, DBL_MIN, DBL_MAX, DBL_EPSILON)
DEFINE_SOLVER(solve_c, float complex, float, cabsf, crealf, copysignf, PLAIN,
              divide_c, FLT_MIN, FLT_MAX, FLT_EPSILON)
DEFINE_SOLVER(solve_z, double complex, double, cabs, creal, copysign, PLAIN,
              DIVIDE_Z, DBL_MIN, DBL_MAX, DBL_EPSILON)

void ojax_tridiagonal_perturbed(char prec, int n, int k_rhs, const void *dl,
                                const void *d, const void *du, const void *b,
                                void *x) {
  switch (prec) {
    case 's':
      solve_s(n, k_rhs, dl, d, du, b, x);
      break;
    case 'd':
      solve_d(n, k_rhs, dl, d, du, b, x);
      break;
    case 'c':
      solve_c(n, k_rhs, dl, d, du, b, x);
      break;
    default:
      solve_z(n, k_rhs, dl, d, du, b, x);
      break;
  }
}

CAMLprim value ojax_tridiagonal_perturbed_host(value v_sizes, value v_complex,
                                               value v_dl, value v_d,
                                               value v_du, value v_b,
                                               value v_x) {
  CAMLparam5(v_sizes, v_complex, v_dl, v_d, v_du);
  CAMLxparam2(v_b, v_x);
  int single = (Caml_ba_array_val(v_b)->flags & CAML_BA_KIND_MASK) ==
               CAML_BA_FLOAT32;
  char prec = Bool_val(v_complex) ? (single ? 'c' : 'z') : (single ? 's' : 'd');
  ojax_tridiagonal_perturbed(prec, Int_val(Field(v_sizes, 0)),
                             Int_val(Field(v_sizes, 1)),
                             Caml_ba_data_val(v_dl), Caml_ba_data_val(v_d),
                             Caml_ba_data_val(v_du), Caml_ba_data_val(v_b),
                             Caml_ba_data_val(v_x));
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_tridiagonal_perturbed_host_bc(value *argv, int argc) {
  (void)argc;
  return ojax_tridiagonal_perturbed_host(argv[0], argv[1], argv[2], argv[3],
                                         argv[4], argv[5], argv[6]);
}
