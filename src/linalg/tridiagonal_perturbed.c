#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "lapack_shim.h"

#if defined(__aarch64__)
#define BINARY_OP(T, NAME, OP, FALLBACK, R)                      \
  static inline T NAME(T a, T b) {                               \
    T r;                                                         \
    __asm__(OP " %" R "0, %" R "1, %" R "2"                      \
            : "=w"(r)                                            \
            : "w"(a), "w"(b));                                   \
    return r;                                                    \
  }
#define TERNARY_OP(T, NAME, OP, FALLBACK, R)                     \
  static inline T NAME(T n, T m, T a) {                          \
    T r;                                                         \
    __asm__(OP " %" R "0, %" R "1, %" R "2, %" R "3"             \
            : "=w"(r)                                            \
            : "w"(n), "w"(m), "w"(a));                           \
    return r;                                                    \
  }
#else
#define BINARY_OP(T, NAME, OP, FALLBACK, R) \
  static inline T NAME(T a, T b) { return FALLBACK; }
#define TERNARY_OP(T, NAME, OP, FALLBACK, R) \
  static inline T NAME(T n, T m, T a) { return FALLBACK; }
#endif

#define DEFINE_OPS(S, T, R, FMA)                                  \
  BINARY_OP(T, S##_mul, "fmul", a * b, R)                         \
  BINARY_OP(T, S##_add, "fadd", a + b, R)                         \
  BINARY_OP(T, S##_sub, "fsub", a - b, R)                         \
  BINARY_OP(T, S##_div, "fdiv", a / b, R)                         \
  BINARY_OP(T, S##_nmul, "fnmul", -(a * b), R)                    \
  TERNARY_OP(T, S##_madd, "fmadd", FMA(n, m, a), R)               \
  TERNARY_OP(T, S##_msub, "fmsub", FMA(-n, m, a), R)              \
  TERNARY_OP(T, S##_nmsub, "fnmsub", FMA(n, m, -a), R)

DEFINE_OPS(s, float, "s", fmaf)
DEFINE_OPS(d, double, "d", fma)

#define DEFINE_REAL_SOLVER(S, T, FABS, COPYSIGN, TINY, EPS, PACKET)            \
  static T S##_max_factor(const T *row, int k_rhs) {                           \
    int start = 1;                                                             \
    T best = FABS(row[0]);                                                     \
    if (k_rhs >= PACKET) {                                                     \
      start = k_rhs & ~(PACKET - 1);                                           \
      for (int j = 1; j < start; j++) {                                        \
        T next = FABS(row[j]);                                                 \
        if (isnan(best) || isnan(next))                                        \
          best = (T)NAN;                                                       \
        else if (best < next)                                                  \
          best = next;                                                         \
      }                                                                        \
    }                                                                          \
    for (int j = start; j < k_rhs; j++) {                                      \
      T next = FABS(row[j]);                                                   \
      if (best < next) best = next;                                            \
    }                                                                          \
    return best;                                                               \
  }                                                                            \
                                                                               \
  static void S##_perturb_pivot(T perturb, T *pivot, T *row, int k_rhs) {      \
    const T one = 1;                                                           \
    const T safemin = TINY;                                                    \
    const T bignum = one / safemin;                                            \
    T abs_pivot = FABS(*pivot);                                                \
    if (abs_pivot >= one) return;                                              \
    T magnitude = FABS(perturb);                                               \
    perturb = COPYSIGN(magnitude < safemin ? safemin : magnitude, *pivot);     \
    int stop = 0;                                                              \
    const T max_factor = S##_max_factor(row, k_rhs);                           \
    while (abs_pivot < one && !stop) {                                         \
      if (abs_pivot < safemin) {                                               \
        if (abs_pivot == (T)0 || S##_mul(max_factor, safemin) > abs_pivot) {   \
          *pivot = S##_add(perturb, *pivot);                                   \
          perturb = S##_add(perturb, perturb);                                 \
        } else {                                                               \
          *pivot = S##_mul(*pivot, bignum);                                    \
          for (int j = 0; j < k_rhs; j++) row[j] = S##_mul(row[j], bignum);    \
          stop = 1;                                                            \
        }                                                                      \
      } else if (max_factor > S##_mul(abs_pivot, bignum)) {                    \
        *pivot = S##_add(perturb, *pivot);                                     \
        perturb = S##_add(perturb, perturb);                                   \
      } else {                                                                 \
        stop = 1;                                                              \
      }                                                                        \
      abs_pivot = FABS(*pivot);                                                \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void solve_##S(int n, int k_rhs, const T *sub, const T *diag,         \
                        const T *super, const T *rhs, T *x) {                  \
    if (n == 0) return;                                                        \
    T *row = calloc((size_t)(k_rhs > 0 ? k_rhs : 1), sizeof(T));               \
    if (n == 1) {                                                              \
      T p = diag[0];                                                           \
      T perturb = S##_mul(FABS(p), EPS);                                       \
      for (int j = 0; j < k_rhs; j++) row[j] = rhs[j];                         \
      S##_perturb_pivot(perturb, &p, row, k_rhs);                              \
      for (int j = 0; j < k_rhs; j++) x[j] = S##_div(row[j], p);               \
      free(row);                                                               \
      return;                                                                  \
    }                                                                          \
    T *u = calloc((size_t)n * 3, sizeof(T));                                   \
    T *u0 = u, *u1 = u + n, *u2 = u + 2 * n;                                   \
    u0[0] = diag[0];                                                           \
    u1[0] = super[0];                                                          \
    T max_abs_u = FABS(u0[0]) < FABS(u1[0]) ? FABS(u1[0]) : FABS(u0[0]);       \
    T scale1 = FABS(u0[0]) + FABS(u1[0]);                                      \
    for (int j = 0; j < k_rhs; j++) x[j * n] = rhs[j * n];                     \
    for (int k = 0; k < n - 1; ++k) {                                          \
      u0[k + 1] = diag[k + 1];                                                 \
      T scale2 = FABS(sub[k + 1]) + FABS(u0[k + 1]);                           \
      if (k < n - 2) scale2 += FABS(super[k + 1]);                             \
      if (sub[k + 1] == (T)0) {                                                \
        scale1 = scale2;                                                       \
        u1[k + 1] = super[k + 1];                                              \
        u2[k] = 0;                                                             \
        for (int j = 0; j < k_rhs; j++) x[k + 1 + j * n] = rhs[k + 1 + j * n]; \
      } else {                                                                 \
        const T piv1 = u0[k] == (T)0 ? (T)0 : FABS(u0[k]) / scale1;            \
        const T piv2 = FABS(sub[k + 1]) / scale2;                              \
        if (piv2 <= piv1) {                                                    \
          scale1 = scale2;                                                     \
          const T factor = S##_div(sub[k + 1], u0[k]);                         \
          u0[k + 1] = S##_msub(factor, u1[k], diag[k + 1]);                    \
          u1[k + 1] = super[k + 1];                                            \
          u2[k] = 0;                                                           \
          for (int j = 0; j < k_rhs; j++)                                      \
            x[k + 1 + j * n] = S##_sub(rhs[k + 1 + j * n],                     \
                                       S##_mul(factor, x[k + j * n]));         \
        } else {                                                               \
          const T factor = S##_div(u0[k], sub[k + 1]);                         \
          const T utmp = u1[k];                                                \
          u0[k] = sub[k + 1];                                                  \
          u0[k + 1] = S##_msub(factor, diag[k + 1], utmp);                     \
          u1[k] = diag[k + 1];                                                 \
          if (k < n - 2) {                                                     \
            u2[k] = super[k + 1];                                              \
            u1[k + 1] = S##_mul(super[k + 1], -factor);                        \
          }                                                                    \
          for (int j = 0; j < k_rhs; j++) {                                    \
            x[k + 1 + j * n] = S##_sub(x[k + j * n],                           \
                                       S##_mul(factor, rhs[k + 1 + j * n]));   \
            x[k + j * n] = rhs[k + 1 + j * n];                                 \
          }                                                                    \
          scale1 = FABS(u0[k + 1]) + FABS(u1[k + 1]);                          \
        }                                                                      \
      }                                                                        \
      if (k < n - 2) {                                                         \
        if (max_abs_u < FABS(u0[k])) max_abs_u = FABS(u0[k]);                  \
        if (max_abs_u < FABS(u1[k])) max_abs_u = FABS(u1[k]);                  \
        if (max_abs_u < FABS(u2[k])) max_abs_u = FABS(u2[k]);                  \
      }                                                                        \
    }                                                                          \
    if (max_abs_u < FABS(u0[n - 1])) max_abs_u = FABS(u0[n - 1]);              \
    T perturb = S##_mul(max_abs_u, EPS);                                       \
    T p = u0[n - 1];                                                           \
    for (int j = 0; j < k_rhs; j++) row[j] = x[n - 1 + j * n];                 \
    S##_perturb_pivot(perturb, &p, row, k_rhs);                                \
    T inverse = S##_div((T)1, p);                                              \
    for (int j = 0; j < k_rhs; j++)                                            \
      x[n - 1 + j * n] = S##_mul(inverse, row[j]);                             \
    p = u0[n - 2];                                                             \
    for (int j = 0; j < k_rhs; j++)                                            \
      row[j] = S##_sub(x[n - 2 + j * n], S##_mul(u1[n - 2], x[n - 1 + j * n]));\
    S##_perturb_pivot(COPYSIGN(perturb, p), &p, row, k_rhs);                   \
    inverse = S##_div((T)1, p);                                                \
    for (int j = 0; j < k_rhs; j++)                                            \
      x[n - 2 + j * n] = S##_mul(inverse, row[j]);                             \
    for (int k = n - 3; k >= 0; --k) {                                         \
      p = u0[k];                                                               \
      for (int j = 0; j < k_rhs; j++) {                                        \
        T difference =                                                         \
            S##_sub(x[k + j * n], S##_mul(u1[k], x[k + 1 + j * n]));           \
        row[j] = S##_sub(difference, S##_mul(u2[k], x[k + 2 + j * n]));        \
      }                                                                        \
      S##_perturb_pivot(COPYSIGN(perturb, p), &p, row, k_rhs);                 \
      inverse = S##_div((T)1, p);                                              \
      for (int j = 0; j < k_rhs; j++)                                          \
        x[k + j * n] = S##_mul(inverse, row[j]);                               \
    }                                                                          \
    free(u);                                                                   \
    free(row);                                                                 \
  }

DEFINE_REAL_SOLVER(s, float, fabsf, copysignf, FLT_MIN, FLT_EPSILON, 4)
DEFINE_REAL_SOLVER(d, double, fabs, copysign, DBL_MIN, DBL_EPSILON, 2)

#define DEFINE_COMPLEX_SOLVER(C, S, R, CABS, FABS, COPYSIGN, TINY, EPS)  \
  typedef struct {                                                             \
    R re, im;                                                                  \
  } C##_t;                                                                     \
                                                                               \
  static inline C##_t C##_make(R re, R im) {                                   \
    C##_t out = {re, im};                                                      \
    return out;                                                                \
  }                                                                            \
                                                                               \
  static inline R C##_abs(C##_t v) {                                           \
    return CABS(__builtin_complex(v.re, v.im));                                \
  }                                                                            \
                                                                               \
  static inline int C##_is_zero(C##_t v) {                                     \
    return v.re == (R)0 && v.im == (R)0;                                       \
  }                                                                            \
                                                                               \
  static inline R C##_box(int on, R v) {                                       \
    return COPYSIGN(on ? (R)1 : (R)0, v);                                      \
  }                                                                            \
                                                                               \
  static inline R C##_unnan(R v) { return isnan(v) ? COPYSIGN((R)0, v) : v; }  \
                                                                               \
  static C##_t C##_rt_mul(R a, R b, R c, R d) {                                \
    R ad = S##_mul(a, d), bc = S##_mul(c, b);                                  \
    R bd = S##_mul(b, d), ac = S##_mul(a, c);                                  \
    R im = S##_add(ad, bc), re = S##_sub(ac, bd);                              \
    if (!(isnan(re) && isnan(im))) return C##_make(re, im);                    \
    int recalc = 0;                                                            \
    if (isinf(a) || isinf(b)) {                                                \
      a = C##_box(isinf(a), a);                                                \
      b = C##_box(isinf(b), b);                                                \
      c = C##_unnan(c);                                                        \
      d = C##_unnan(d);                                                        \
      recalc = 1;                                                              \
    }                                                                          \
    if (isinf(c) || isinf(d)) {                                                \
      c = C##_box(isinf(c), c);                                                \
      d = C##_box(isinf(d), d);                                                \
      a = C##_unnan(a);                                                        \
      b = C##_unnan(b);                                                        \
      recalc = 1;                                                              \
    }                                                                          \
    if (!recalc && (isinf(ac) || isinf(bd) || isinf(ad) || isinf(bc))) {       \
      a = C##_unnan(a);                                                        \
      b = C##_unnan(b);                                                        \
      c = C##_unnan(c);                                                        \
      d = C##_unnan(d);                                                        \
      recalc = 1;                                                              \
    }                                                                          \
    if (!recalc) return C##_make(re, im);                                      \
    R big = (R)INFINITY;                                                       \
    R next_re = S##_nmsub(a, c, S##_mul(b, d));                                \
    R next_im = S##_madd(a, d, S##_mul(b, c));                                 \
    return C##_make(S##_mul(next_re, big), S##_mul(next_im, big));             \
  }                                                                            \
                                                                               \
  static C##_t C##_rt_div(R a, R b, R c, R d) {                                \
    R re, im;                                                                  \
    if (FABS(c) < FABS(d)) {                                                   \
      R ratio = S##_div(c, d);                                                 \
      R denominator = S##_madd(c, ratio, d);                                   \
      R top_re = S##_madd(ratio, a, b);                                        \
      R top_im = S##_nmsub(ratio, b, a);                                       \
      re = S##_div(top_re, denominator);                                       \
      im = S##_div(top_im, denominator);                                       \
    } else {                                                                   \
      R ratio = S##_div(d, c);                                                 \
      R denominator = S##_madd(d, ratio, c);                                   \
      R top_re = S##_madd(b, ratio, a);                                        \
      R top_im = S##_msub(a, ratio, b);                                        \
      re = S##_div(top_re, denominator);                                       \
      im = S##_div(top_im, denominator);                                       \
    }                                                                          \
    if (!(isnan(re) && isnan(im))) return C##_make(re, im);                    \
    if (c == (R)0 && d == (R)0 && (!isnan(a) || !isnan(b))) {                  \
      R signed_inf = COPYSIGN((R)INFINITY, c);                                 \
      return C##_make(S##_mul(a, signed_inf), S##_mul(b, signed_inf));         \
    }                                                                          \
    if ((isinf(a) || isinf(b)) && isfinite(c) && isfinite(d)) {                \
      R a1 = C##_box(isinf(a), a), b1 = C##_box(isinf(b), b);                  \
      R big = (R)INFINITY;                                                     \
      return C##_make(S##_mul(S##_madd(c, a1, S##_mul(d, b1)), big),           \
                      S##_mul(S##_nmsub(c, b1, S##_mul(d, a1)), big));         \
    }                                                                          \
    if ((isinf(c) || isinf(d)) && isfinite(a) && isfinite(b)) {                \
      R c1 = C##_box(isinf(c), c), d1 = C##_box(isinf(d), d);                  \
      R zero = (R)0;                                                           \
      return C##_make(S##_mul(S##_madd(a, c1, S##_mul(b, d1)), zero),          \
                      S##_mul(S##_nmsub(b, c1, S##_mul(a, d1)), zero));        \
    }                                                                          \
    return C##_make(re, im);                                                   \
  }                                                                            \
                                                                               \
  static inline C##_t C##_finish(R re, R im, R a, R b, R c, R d) {             \
    if (isnan(re) && isnan(im)) return C##_rt_mul(a, b, c, d);                 \
    return C##_make(re, im);                                                   \
  }                                                                            \
                                                                               \
  static inline C##_t C##_mul_left(C##_t f, C##_t v) {                         \
    return C##_finish(                                                         \
        S##_sub(S##_mul(f.re, v.re), S##_mul(f.im, v.im)),                     \
        S##_add(S##_mul(f.im, v.re), S##_mul(f.re, v.im)), f.re, f.im, v.re,   \
        v.im);                                                                 \
  }                                                                            \
                                                                               \
  static inline C##_t C##_mul_lanes(C##_t f, C##_t v) {                        \
    return C##_finish(                                                         \
        S##_sub(S##_mul(v.re, f.re), S##_mul(v.im, f.im)),                     \
        S##_add(S##_mul(v.im, f.re), S##_mul(v.re, f.im)), f.re, f.im, v.re,   \
        v.im);                                                                 \
  }                                                                            \
                                                                               \
  static inline C##_t C##_mul_inverse(C##_t inverse, C##_t v) {                \
    return C##_finish(                                                         \
        S##_sub(S##_mul(inverse.re, v.re), S##_mul(inverse.im, v.im)),         \
        S##_add(S##_mul(inverse.im, v.re), S##_mul(inverse.re, v.im)), v.re,   \
        v.im, inverse.re, inverse.im);                                         \
  }                                                                            \
                                                                               \
  static inline C##_t C##_mul_negated(C##_t f, C##_t v) {                      \
    return C##_finish(                                                         \
        S##_sub(S##_mul(f.im, v.im), S##_mul(f.re, v.re)),                     \
        S##_sub(S##_nmul(f.re, v.im), S##_mul(f.im, v.re)), -f.re, -f.im,      \
        v.re, v.im);                                                           \
  }                                                                            \
                                                                               \
  static inline C##_t C##_sub(C##_t a, C##_t b) {                              \
    return C##_make(S##_sub(a.re, b.re), S##_sub(a.im, b.im));                 \
  }                                                                            \
                                                                               \
  static void C##_perturb_pivot(R perturb, C##_t *pivot, C##_t *row,           \
                                int k_rhs) {                                   \
    const R one = 1;                                                           \
    const R safemin = TINY;                                                    \
    const R bignum = one / safemin;                                            \
    R abs_pivot = C##_abs(*pivot);                                             \
    if (abs_pivot >= one) return;                                              \
    R magnitude = FABS(perturb);                                               \
    perturb = COPYSIGN(magnitude < safemin ? safemin : magnitude, pivot->re);  \
    int stop = 0;                                                              \
    R max_factor = C##_abs(row[0]);                                            \
    for (int j = 1; j < k_rhs; j++) {                                          \
      R next = C##_abs(row[j]);                                                \
      if (max_factor < next) max_factor = next;                                \
    }                                                                          \
    while (abs_pivot < one && !stop) {                                         \
      if (abs_pivot < safemin) {                                               \
        if (abs_pivot == (R)0 || S##_mul(max_factor, safemin) > abs_pivot) {   \
          pivot->re = S##_add(perturb, pivot->re);                             \
          perturb = S##_add(perturb, perturb);                                 \
        } else {                                                               \
          pivot->re = S##_mul(pivot->re, bignum);                              \
          pivot->im = S##_mul(pivot->im, bignum);                              \
          for (int j = 0; j < k_rhs; j++) {                                    \
            row[j].re = S##_mul(row[j].re, bignum);                            \
            row[j].im = S##_mul(row[j].im, bignum);                            \
          }                                                                    \
          stop = 1;                                                            \
        }                                                                      \
      } else if (max_factor > S##_mul(abs_pivot, bignum)) {                    \
        pivot->re = S##_add(perturb, pivot->re);                               \
        perturb = S##_add(perturb, perturb);                                   \
      } else {                                                                 \
        stop = 1;                                                              \
      }                                                                        \
      abs_pivot = C##_abs(*pivot);                                             \
    }                                                                          \
  }                                                                            \
                                                                               \
  static void solve_##C(int n, int k_rhs, const C##_t *sub,                    \
                        const C##_t *diag, const C##_t *super,                 \
                        const C##_t *rhs, C##_t *x) {                          \
    if (n == 0) return;                                                        \
    C##_t *row = calloc((size_t)(k_rhs > 0 ? k_rhs : 1), sizeof(C##_t));       \
    if (n == 1) {                                                              \
      C##_t p = diag[0];                                                       \
      R perturb = S##_mul(C##_abs(p), EPS);                                    \
      for (int j = 0; j < k_rhs; j++) row[j] = rhs[j];                         \
      C##_perturb_pivot(perturb, &p, row, k_rhs);                              \
      for (int j = 0; j < k_rhs; j++)                                          \
        x[j] = C##_rt_div(row[j].re, row[j].im, p.re, p.im);                   \
      free(row);                                                               \
      return;                                                                  \
    }                                                                          \
    C##_t *u = calloc((size_t)n * 3, sizeof(C##_t));                           \
    C##_t *u0 = u, *u1 = u + n, *u2 = u + 2 * n;                               \
    u0[0] = diag[0];                                                           \
    u1[0] = super[0];                                                          \
    R max_abs_u =                                                              \
        C##_abs(u0[0]) < C##_abs(u1[0]) ? C##_abs(u1[0]) : C##_abs(u0[0]);     \
    R scale1 = C##_abs(u0[0]) + C##_abs(u1[0]);                                \
    for (int j = 0; j < k_rhs; j++) x[j * n] = rhs[j * n];                     \
    for (int k = 0; k < n - 1; ++k) {                                          \
      u0[k + 1] = diag[k + 1];                                                 \
      R scale2 = C##_abs(sub[k + 1]) + C##_abs(u0[k + 1]);                     \
      if (k < n - 2) scale2 += C##_abs(super[k + 1]);                          \
      if (C##_is_zero(sub[k + 1])) {                                           \
        scale1 = scale2;                                                       \
        u1[k + 1] = super[k + 1];                                              \
        u2[k] = C##_make(0, 0);                                                \
        for (int j = 0; j < k_rhs; j++) x[k + 1 + j * n] = rhs[k + 1 + j * n]; \
      } else {                                                                 \
        const R piv1 = C##_is_zero(u0[k]) ? (R)0 : C##_abs(u0[k]) / scale1;    \
        const R piv2 = C##_abs(sub[k + 1]) / scale2;                           \
        if (piv2 <= piv1) {                                                    \
          scale1 = scale2;                                                     \
          const C##_t factor =                                                 \
              C##_rt_div(sub[k + 1].re, sub[k + 1].im, u0[k].re, u0[k].im);    \
          u0[k + 1] = C##_sub(diag[k + 1], C##_mul_left(factor, u1[k]));       \
          u1[k + 1] = super[k + 1];                                            \
          u2[k] = C##_make(0, 0);                                              \
          for (int j = 0; j < k_rhs; j++)                                      \
            x[k + 1 + j * n] = C##_sub(rhs[k + 1 + j * n],                     \
                                       C##_mul_lanes(factor, x[k + j * n]));   \
        } else {                                                               \
          const C##_t factor =                                                 \
              C##_rt_div(u0[k].re, u0[k].im, sub[k + 1].re, sub[k + 1].im);    \
          const C##_t utmp = u1[k];                                            \
          u0[k] = sub[k + 1];                                                  \
          u0[k + 1] = C##_sub(utmp, C##_mul_left(factor, diag[k + 1]));        \
          u1[k] = diag[k + 1];                                                 \
          if (k < n - 2) {                                                     \
            u2[k] = super[k + 1];                                              \
            u1[k + 1] = C##_mul_negated(factor, super[k + 1]);                 \
          }                                                                    \
          for (int j = 0; j < k_rhs; j++) {                                    \
            x[k + 1 + j * n] =                                                 \
                C##_sub(x[k + j * n],                                          \
                        C##_mul_lanes(factor, rhs[k + 1 + j * n]));            \
            x[k + j * n] = rhs[k + 1 + j * n];                                 \
          }                                                                    \
          scale1 = C##_abs(u0[k + 1]) + C##_abs(u1[k + 1]);                    \
        }                                                                      \
      }                                                                        \
      if (k < n - 2) {                                                         \
        if (max_abs_u < C##_abs(u0[k])) max_abs_u = C##_abs(u0[k]);            \
        if (max_abs_u < C##_abs(u1[k])) max_abs_u = C##_abs(u1[k]);            \
        if (max_abs_u < C##_abs(u2[k])) max_abs_u = C##_abs(u2[k]);            \
      }                                                                        \
    }                                                                          \
    if (max_abs_u < C##_abs(u0[n - 1])) max_abs_u = C##_abs(u0[n - 1]);        \
    R perturb = S##_mul(max_abs_u, EPS);                                       \
    C##_t p = u0[n - 1];                                                       \
    for (int j = 0; j < k_rhs; j++) row[j] = x[n - 1 + j * n];                 \
    C##_perturb_pivot(perturb, &p, row, k_rhs);                                \
    C##_t inverse = C##_rt_div((R)1, (R)0, p.re, p.im);                        \
    for (int j = 0; j < k_rhs; j++)                                            \
      x[n - 1 + j * n] = C##_mul_inverse(inverse, row[j]);                     \
    p = u0[n - 2];                                                             \
    for (int j = 0; j < k_rhs; j++)                                            \
      row[j] = C##_sub(x[n - 2 + j * n],                                       \
                       C##_mul_lanes(u1[n - 2], x[n - 1 + j * n]));            \
    C##_perturb_pivot(COPYSIGN(perturb, p.re), &p, row, k_rhs);                \
    inverse = C##_rt_div((R)1, (R)0, p.re, p.im);                              \
    for (int j = 0; j < k_rhs; j++)                                            \
      x[n - 2 + j * n] = C##_mul_inverse(inverse, row[j]);                     \
    for (int k = n - 3; k >= 0; --k) {                                         \
      p = u0[k];                                                               \
      for (int j = 0; j < k_rhs; j++) {                                        \
        C##_t first = C##_mul_left(u1[k], x[k + 1 + j * n]);                   \
        C##_t second = C##_mul_left(u2[k], x[k + 2 + j * n]);                  \
        row[j] = C##_sub(C##_sub(x[k + j * n], first), second);                \
      }                                                                        \
      C##_perturb_pivot(COPYSIGN(perturb, p.re), &p, row, k_rhs);              \
      inverse = C##_rt_div((R)1, (R)0, p.re, p.im);                            \
      for (int j = 0; j < k_rhs; j++)                                          \
        x[k + j * n] = C##_mul_inverse(inverse, row[j]);                       \
    }                                                                          \
    free(u);                                                                   \
    free(row);                                                                 \
  }

DEFINE_COMPLEX_SOLVER(c, s, float, cabsf, fabsf, copysignf,
                      FLT_MIN, FLT_EPSILON)
DEFINE_COMPLEX_SOLVER(z, d, double, cabs, fabs, copysign,
                      DBL_MIN, DBL_EPSILON)

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
