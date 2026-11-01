#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dlfcn.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "lapack_shim.h"

typedef int32_t LI;
typedef void *P;

#define L1 ((P)(uintptr_t)1)
#define OOM INT_MIN

static void *ojax_lapack_library(void) {
  static int tried = 0;
  static void *lib = NULL;
  if (tried) return lib;
  tried = 1;
#ifdef __APPLE__
  lib = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate",
               RTLD_NOW | RTLD_LOCAL);
#else
  const char *env = getenv("OJAX_LAPACK_LIB");
  const char *names[] = {env, "libopenblas.so.0", "liblapack.so.3"};
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && !lib; i++)
    if (names[i] && names[i][0]) lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
#endif
  return lib;
}

void *ojax_lapack_symbol(char prec, const char *base) {
  void *lib = ojax_lapack_library();
  if (!lib) return NULL;
  char name[64];
#ifdef __APPLE__
  snprintf(name, sizeof name, "%c%s$NEWLAPACK", prec, base);
  void *fn = dlsym(lib, name);
  if (fn) return fn;
#endif
  static const char *const forms[] = {"%c%s_", "scipy_%c%s_"};
  for (size_t i = 0; i < sizeof(forms) / sizeof(forms[0]); i++) {
    snprintf(name, sizeof name, forms[i], prec, base);
    void *found = dlsym(lib, name);
    if (found) return found;
  }
  return NULL;
}

static int ojax_lapack_is_ilp64(void) {
  void *lib = ojax_lapack_library();
  return lib && !ojax_lapack_symbol('d', "potrf") &&
         (dlsym(lib, "dpotrf64_") || dlsym(lib, "scipy_dpotrf64_"));
}

static int ojax_lapack_resolve(void) {
  return ojax_lapack_symbol('d', "potrf") != NULL;
}

static void *need(char prec, const char *base) {
  void *fn = ojax_lapack_symbol(prec, base);
  if (!fn) {
    char msg[96];
    snprintf(msg, sizeof msg,
             "ojax linalg: LAPACK library lacks %c%s", prec, base);
    caml_failwith(msg);
  }
  return fn;
}

#define IP(v, i) ((int)Int_val(Field((v), (i))))
#define D(v) ((void *)Caml_ba_data_val(v))
#define SINGLE(v) \
  ((Caml_ba_array_val(v)->flags & CAML_BA_KIND_MASK) == CAML_BA_FLOAT32)
#define PREC_R(v) (SINGLE(v) ? 's' : 'd')
#define PREC_C(v) (SINGLE(v) ? 'c' : 'z')
#define CALL(fn, n, ...) ((void (*)(F##n))(fn))(__VA_ARGS__)
#define F5 P, P, P, P, P
#define F6 F5, P
#define F7 F6, P
#define F8 F7, P
#define F9 F8, P
#define F10 F9, P
#define F11 F10, P
#define F12 F11, P
#define F13 F12, P
#define F14 F13, P
#define F15 F14, P
#define F16 F15, P
#define F17 F16, P
#define F18 F17, P
#define F19 F18, P

CAMLprim value ojax_lapack_available(value v_unit) {
  CAMLparam1(v_unit);
  CAMLreturn(Val_bool(ojax_lapack_resolve()));
}

CAMLprim value ojax_lapack_single_routines(value v_unit) {
  CAMLparam1(v_unit);
#ifdef __APPLE__
  CAMLreturn(Val_false);
#else
  void *lib = ojax_lapack_library();
  CAMLreturn(Val_bool(lib && dlsym(lib, "scipy_spotrf_") != NULL));
#endif
}

CAMLprim value ojax_lapack_abi_int_size(value v_unit) {
  CAMLparam1(v_unit);
  CAMLreturn(Val_int(ojax_lapack_resolve() ? (int)sizeof(LI) : 0));
}

CAMLprim value ojax_lapack_backend(value v_unit) {
  CAMLparam1(v_unit);
#ifdef __APPLE__
  CAMLreturn(caml_copy_string("accelerate"));
#else
  if (ojax_lapack_is_ilp64())
    CAMLreturn(caml_copy_string("unavailable: ILP64 symbols need 32-bit LAPACK"));
  CAMLreturn(
      caml_copy_string(ojax_lapack_resolve() ? "dlopen" : "unavailable"));
#endif
}

static size_t elt_size(char prec) {
  return prec == 's' ? 4 : (prec == 'z' ? 16 : 8);
}

LI ojax_lapack_query_size(char prec, const void *q) {
  LI n = (prec == 's' || prec == 'c') ? (LI)(*(const float *)q)
                                      : (LI)(*(const double *)q);
  return n < 1 ? 1 : n;
}

static void *work_alloc(char prec, LI n) {
  return malloc(elt_size(prec) * (size_t)(n < 1 ? 1 : n));
}

static void *rwork_alloc(char prec, size_t n) {
  return malloc((prec == 'c' ? 4 : 8) * (n < 1 ? 1 : n));
}

LI ojax_lapack_syevd_work(LI n) { return 1 + 6 * n + 2 * n * n; }
LI ojax_lapack_heevd_work(LI n) { return 2 * n + n * n; }
LI ojax_lapack_heevd_rwork(LI n) { return 1 + 5 * n + 2 * n * n; }
LI ojax_lapack_evd_iwork(LI n) { return 3 + 5 * n; }

int ojax_lapack_call_potrf(char prec, char uplo, LI n, void *a, LI lda) {
  void *fn = ojax_lapack_symbol(prec, "potrf");
  LI info = 0;
  CALL(fn, 6, &uplo, &n, a, &lda, &info, L1);
  return info;
}

int ojax_lapack_call_getrf(char prec, LI m, LI n, void *a, LI lda, LI *ipiv) {
  void *fn = ojax_lapack_symbol(prec, "getrf");
  LI info = 0;
  CALL(fn, 6, &m, &n, a, &lda, ipiv, &info);
  return info;
}

int ojax_lapack_call_geqrf(char prec, LI m, LI n, void *a, LI lda, void *tau) {
  void *fn = ojax_lapack_symbol(prec, "geqrf");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  CALL(fn, 8, &m, &n, a, &lda, tau, q, &lwork, &info);
  if (info != 0) return info;
  lwork = ojax_lapack_query_size(prec, q);
  void *work = work_alloc(prec, lwork);
  if (!work) return OOM;
  CALL(fn, 8, &m, &n, a, &lda, tau, work, &lwork, &info);
  free(work);
  return info;
}

int ojax_lapack_call_geqp3(char prec, LI m, LI n, void *a, LI lda, LI *jpvt,
                           void *tau) {
  void *fn = ojax_lapack_symbol(prec, "geqp3");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  int cplx = prec == 'c' || prec == 'z';
  void *rwork = cplx ? rwork_alloc(prec, 2 * (size_t)n + 1) : NULL;
  if (cplx && !rwork) return OOM;
  if (cplx)
    CALL(fn, 10, &m, &n, a, &lda, jpvt, tau, q, &lwork, rwork, &info);
  else
    CALL(fn, 9, &m, &n, a, &lda, jpvt, tau, q, &lwork, &info);
  if (info == 0) {
    lwork = ojax_lapack_query_size(prec, q);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(rwork);
      return OOM;
    }
    if (cplx)
      CALL(fn, 10, &m, &n, a, &lda, jpvt, tau, work, &lwork, rwork, &info);
    else
      CALL(fn, 9, &m, &n, a, &lda, jpvt, tau, work, &lwork, &info);
    free(work);
  }
  free(rwork);
  return info;
}

int ojax_lapack_call_orgqr(char prec, LI m, LI n, LI k, void *a, LI lda,
                           void *tau) {
  int cplx = prec == 'c' || prec == 'z';
  void *fn = ojax_lapack_symbol(prec, cplx ? "ungqr" : "orgqr");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  CALL(fn, 9, &m, &n, &k, a, &lda, tau, q, &lwork, &info);
  if (info != 0) return info;
  lwork = ojax_lapack_query_size(prec, q);
  void *work = work_alloc(prec, lwork);
  if (!work) return OOM;
  CALL(fn, 9, &m, &n, &k, a, &lda, tau, work, &lwork, &info);
  free(work);
  return info;
}

int ojax_lapack_call_ormqr(char prec, char side, char trans, LI m, LI n, LI k,
                           void *a, LI lda, void *tau, void *c, LI ldc) {
  int cplx = prec == 'c' || prec == 'z';
  void *fn = ojax_lapack_symbol(prec, cplx ? "unmqr" : "ormqr");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  CALL(fn, 15, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, q, &lwork,
       &info, L1, L1);
  if (info != 0) return info;
  lwork = ojax_lapack_query_size(prec, q);
  void *work = work_alloc(prec, lwork);
  if (!work) return OOM;
  CALL(fn, 15, &side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork,
       &info, L1, L1);
  free(work);
  return info;
}

int ojax_lapack_call_gesdd(char prec, char jobz, LI m, LI n, void *a, LI lda,
                           void *s, void *u, LI ldu, void *vt, LI ldvt) {
  void *fn = ojax_lapack_symbol(prec, "gesdd");
  int cplx = prec == 'c' || prec == 'z';
  size_t mn = (size_t)(m < n ? m : n), mx = (size_t)(m < n ? n : m);
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  LI *iwork = malloc(sizeof(LI) * (8 * mn + 1));
  void *rwork = cplx ? rwork_alloc(prec, 5 * mn * mn + 7 * mn + 2 * mx * mn + 16)
                     : NULL;
  if (!iwork || (cplx && !rwork)) {
    free(iwork);
    free(rwork);
    return OOM;
  }
  if (cplx)
    CALL(fn, 16, &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, q, &lwork,
         rwork, iwork, &info, L1);
  else
    CALL(fn, 15, &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, q, &lwork,
         iwork, &info, L1);
  if (info == 0) {
    lwork = ojax_lapack_query_size(prec, q);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(iwork);
      free(rwork);
      return OOM;
    }
    if (cplx)
      CALL(fn, 16, &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           rwork, iwork, &info, L1);
    else
      CALL(fn, 15, &jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
           iwork, &info, L1);
    free(work);
  }
  free(iwork);
  free(rwork);
  return info;
}

int ojax_lapack_call_gesvd(char prec, char job, LI m, LI n, void *a, LI lda,
                           void *s, void *u, LI ldu, void *vt, LI ldvt) {
  void *fn = ojax_lapack_symbol(prec, "gesvd");
  int cplx = prec == 'c' || prec == 'z';
  size_t mn = (size_t)(m < n ? m : n);
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  void *rwork = cplx ? rwork_alloc(prec, 5 * mn + 16) : NULL;
  if (cplx && !rwork) return OOM;
  if (cplx)
    CALL(fn, 17, &job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, q, &lwork,
         rwork, &info, L1, L1);
  else
    CALL(fn, 16, &job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, q, &lwork,
         &info, L1, L1);
  if (info == 0) {
    lwork = ojax_lapack_query_size(prec, q);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(rwork);
      return OOM;
    }
    if (cplx)
      CALL(fn, 17, &job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
           &lwork, rwork, &info, L1, L1);
    else
      CALL(fn, 16, &job, &job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work,
           &lwork, &info, L1, L1);
    free(work);
  }
  free(rwork);
  return info;
}

int ojax_lapack_call_syevd(char prec, char jobz, char uplo, LI n, void *a,
                           LI lda, void *w) {
  int cplx = prec == 'c' || prec == 'z';
  void *fn = ojax_lapack_symbol(prec, cplx ? "heevd" : "syevd");
  LI info = 0;
  LI liwork = ojax_lapack_evd_iwork(n);
  LI *iwork = malloc(sizeof(LI) * (size_t)liwork);
  if (!iwork) return OOM;
  if (cplx) {
    LI lwork = ojax_lapack_heevd_work(n), lrwork = ojax_lapack_heevd_rwork(n);
    void *work = work_alloc(prec, lwork);
    void *rwork = rwork_alloc(prec, (size_t)lrwork);
    if (!work || !rwork) {
      free(work);
      free(rwork);
      free(iwork);
      return OOM;
    }
    CALL(fn, 15, &jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &lrwork,
         iwork, &liwork, &info, L1, L1);
    free(work);
    free(rwork);
  } else {
    LI lwork = ojax_lapack_syevd_work(n);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(iwork);
      return OOM;
    }
    CALL(fn, 13, &jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork,
         &info, L1, L1);
    free(work);
  }
  free(iwork);
  return info;
}

static int all_finite(char prec, const void *a, size_t count) {
  for (size_t i = 0; i < count; i++) {
    double v = (prec == 's' || prec == 'c') ? (double)((const float *)a)[i]
                                            : ((const double *)a)[i];
    if (!isfinite(v)) return 0;
  }
  return 1;
}

int ojax_lapack_call_geev(char prec, char jobvl, char jobvr, LI n, void *a,
                          LI lda, void *w, void *vl, LI ldvl, void *vr,
                          LI ldvr) {
  void *fn = ojax_lapack_symbol(prec, "geev");
  int cplx = prec == 'c' || prec == 'z';
  if (!all_finite(prec, a, (size_t)n * (size_t)n * (cplx ? 2 : 1))) return -4;
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  void *rwork = cplx ? rwork_alloc(prec, 2 * (size_t)n + 1) : NULL;
  if (cplx && !rwork) return OOM;
  char *wi = (char *)w + (size_t)n * (prec == 's' ? 4 : 8);
  if (cplx)
    CALL(fn, 16, &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, q,
         &lwork, rwork, &info, L1, L1);
  else
    CALL(fn, 16, &jobvl, &jobvr, &n, a, &lda, w, wi, vl, &ldvl, vr, &ldvr, q,
         &lwork, &info, L1, L1);
  if (info == 0) {
    lwork = ojax_lapack_query_size(prec, q);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(rwork);
      return OOM;
    }
    if (cplx)
      CALL(fn, 16, &jobvl, &jobvr, &n, a, &lda, w, vl, &ldvl, vr, &ldvr, work,
           &lwork, rwork, &info, L1, L1);
    else
      CALL(fn, 16, &jobvl, &jobvr, &n, a, &lda, w, wi, vl, &ldvl, vr, &ldvr,
           work, &lwork, &info, L1, L1);
    free(work);
  }
  free(rwork);
  return info;
}

int ojax_lapack_call_gees(char prec, char jobvs, LI n, void *a, LI lda, LI *sdim,
                          void *w, void *vs, LI ldvs) {
  void *fn = ojax_lapack_symbol(prec, "gees");
  int cplx = prec == 'c' || prec == 'z';
  char sort = 'N';
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  void *rwork = cplx ? rwork_alloc(prec, (size_t)n + 1) : NULL;
  if (cplx && !rwork) return OOM;
  char *wi = (char *)w + (size_t)n * (prec == 's' ? 4 : 8);
  if (cplx)
    CALL(fn, 17, &jobvs, &sort, NULL, &n, a, &lda, sdim, w, vs, &ldvs, q,
         &lwork, rwork, NULL, &info, L1, L1);
  else
    CALL(fn, 17, &jobvs, &sort, NULL, &n, a, &lda, sdim, w, wi, vs, &ldvs, q,
         &lwork, NULL, &info, L1, L1);
  if (info == 0) {
    lwork = ojax_lapack_query_size(prec, q);
    void *work = work_alloc(prec, lwork);
    if (!work) {
      free(rwork);
      return OOM;
    }
    if (cplx)
      CALL(fn, 17, &jobvs, &sort, NULL, &n, a, &lda, sdim, w, vs, &ldvs, work,
           &lwork, rwork, NULL, &info, L1, L1);
    else
      CALL(fn, 17, &jobvs, &sort, NULL, &n, a, &lda, sdim, w, wi, vs, &ldvs,
           work, &lwork, NULL, &info, L1, L1);
    free(work);
  }
  free(rwork);
  return info;
}

int ojax_lapack_call_gehrd(char prec, LI n, LI ilo, LI ihi, void *a, LI lda,
                           void *tau) {
  void *fn = ojax_lapack_symbol(prec, "gehrd");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  CALL(fn, 9, &n, &ilo, &ihi, a, &lda, tau, q, &lwork, &info);
  if (info != 0) return info;
  lwork = ojax_lapack_query_size(prec, q);
  void *work = work_alloc(prec, lwork);
  if (!work) return OOM;
  CALL(fn, 9, &n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
  free(work);
  return info;
}

int ojax_lapack_call_sytrd(char prec, char uplo, LI n, void *a, LI lda, void *d,
                           void *e, void *tau) {
  int cplx = prec == 'c' || prec == 'z';
  void *fn = ojax_lapack_symbol(prec, cplx ? "hetrd" : "sytrd");
  LI info = 0, lwork = -1;
  double q[2] = {0.0, 0.0};
  CALL(fn, 11, &uplo, &n, a, &lda, d, e, tau, q, &lwork, &info, L1);
  if (info != 0) return info;
  lwork = ojax_lapack_query_size(prec, q);
  void *work = work_alloc(prec, lwork);
  if (!work) return OOM;
  CALL(fn, 11, &uplo, &n, a, &lda, d, e, tau, work, &lwork, &info, L1);
  free(work);
  return info;
}

void ojax_lapack_call_trsm(char prec, char side, char uplo, char trans,
                           char diag, LI m, LI n, void *a, LI lda, void *b,
                           LI ldb) {
  void *fn = ojax_lapack_symbol(prec, "trsm");
  float s1[2] = {1.0f, 0.0f};
  double d1[2] = {1.0, 0.0};
  void *alpha = (prec == 's' || prec == 'c') ? (void *)s1 : (void *)d1;
  CALL(fn, 15, &side, &uplo, &trans, &diag, &m, &n, alpha, a, &lda, b, &ldb, L1,
       L1, L1, L1);
}

int ojax_lapack_call_gtsv(char prec, LI n, LI nrhs, void *dl, void *d, void *du,
                          void *b, LI ldb) {
  void *fn = ojax_lapack_symbol(prec, "gtsv");
  LI info = 0;
  CALL(fn, 8, &n, &nrhs, dl, d, du, b, &ldb, &info);
  return info;
}

#define FINISH(name)                                                     \
  caml_acquire_runtime_system();                                         \
  if (info == OOM) caml_failwith("ojax linalg: " name " workspace alloc failed");

#define NEED_CHECK(prec, base) (void)need((prec), (base))

CAMLprim value ojax_lapack_potrf(value v_ip, value v_cplx, value v_a) {
  CAMLparam3(v_ip, v_cplx, v_a);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "potrf");
  char uplo = (char)IP(v_ip, 0);
  LI n = IP(v_ip, 1), lda = IP(v_ip, 2);
  void *a = D(v_a);
  caml_release_runtime_system();
  int info = ojax_lapack_call_potrf(prec, uplo, n, a, lda);
  FINISH("potrf");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_getrf(value v_ip, value v_cplx, value v_a,
                                 value v_ipiv) {
  CAMLparam4(v_ip, v_cplx, v_a, v_ipiv);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "getrf");
  LI m = IP(v_ip, 0), n = IP(v_ip, 1), lda = IP(v_ip, 2);
  void *a = D(v_a);
  LI *ipiv = D(v_ipiv);
  caml_release_runtime_system();
  int info = ojax_lapack_call_getrf(prec, m, n, a, lda, ipiv);
  FINISH("getrf");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_geqrf(value v_ip, value v_cplx, value v_a,
                                 value v_tau) {
  CAMLparam4(v_ip, v_cplx, v_a, v_tau);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "geqrf");
  LI m = IP(v_ip, 0), n = IP(v_ip, 1), lda = IP(v_ip, 2);
  void *a = D(v_a), *tau = D(v_tau);
  caml_release_runtime_system();
  int info = ojax_lapack_call_geqrf(prec, m, n, a, lda, tau);
  FINISH("geqrf");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_geqp3(value v_ip, value v_cplx, value v_a,
                                 value v_jpvt, value v_tau) {
  CAMLparam5(v_ip, v_cplx, v_a, v_jpvt, v_tau);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "geqp3");
  LI m = IP(v_ip, 0), n = IP(v_ip, 1), lda = IP(v_ip, 2);
  void *a = D(v_a), *tau = D(v_tau);
  LI *jpvt = D(v_jpvt);
  caml_release_runtime_system();
  int info = ojax_lapack_call_geqp3(prec, m, n, a, lda, jpvt, tau);
  FINISH("geqp3");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_orgqr(value v_ip, value v_cplx, value v_a,
                                 value v_tau) {
  CAMLparam4(v_ip, v_cplx, v_a, v_tau);
  int cplx = Bool_val(v_cplx);
  char prec = cplx ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, cplx ? "ungqr" : "orgqr");
  LI m = IP(v_ip, 0), n = IP(v_ip, 1), k = IP(v_ip, 2), lda = IP(v_ip, 3);
  void *a = D(v_a), *tau = D(v_tau);
  caml_release_runtime_system();
  int info = ojax_lapack_call_orgqr(prec, m, n, k, a, lda, tau);
  FINISH("orgqr");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_ormqr(value v_ip, value v_cplx, value v_a,
                                 value v_tau, value v_c) {
  CAMLparam5(v_ip, v_cplx, v_a, v_tau, v_c);
  int cplx = Bool_val(v_cplx);
  char prec = cplx ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, cplx ? "unmqr" : "ormqr");
  char side = (char)IP(v_ip, 0), trans = (char)IP(v_ip, 1);
  LI m = IP(v_ip, 2), n = IP(v_ip, 3), k = IP(v_ip, 4), lda = IP(v_ip, 5),
     ldc = IP(v_ip, 6);
  void *a = D(v_a), *tau = D(v_tau), *c = D(v_c);
  caml_release_runtime_system();
  int info =
      ojax_lapack_call_ormqr(prec, side, trans, m, n, k, a, lda, tau, c, ldc);
  FINISH("ormqr");
  CAMLreturn(Val_int(info));
}

static value svd_stub(int qr, value v_ip, value v_cplx, value v_a, value v_s,
                      value v_u, value v_vt) {
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, qr ? "gesvd" : "gesdd");
  char job = (char)IP(v_ip, 0);
  LI m = IP(v_ip, 1), n = IP(v_ip, 2), lda = IP(v_ip, 3), ldu = IP(v_ip, 4),
     ldvt = IP(v_ip, 5);
  void *a = D(v_a), *s = D(v_s), *u = D(v_u), *vt = D(v_vt);
  caml_release_runtime_system();
  int info = qr ? ojax_lapack_call_gesvd(prec, job, m, n, a, lda, s, u, ldu, vt,
                                         ldvt)
                : ojax_lapack_call_gesdd(prec, job, m, n, a, lda, s, u, ldu, vt,
                                         ldvt);
  FINISH("svd");
  return Val_int(info);
}

CAMLprim value ojax_lapack_gesdd(value v_ip, value v_cplx, value v_a, value v_s,
                                 value v_u, value v_vt) {
  CAMLparam5(v_ip, v_cplx, v_a, v_s, v_u);
  CAMLxparam1(v_vt);
  CAMLreturn(svd_stub(0, v_ip, v_cplx, v_a, v_s, v_u, v_vt));
}

CAMLprim value ojax_lapack_gesdd_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_gesdd(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value ojax_lapack_gesvd(value v_ip, value v_cplx, value v_a, value v_s,
                                 value v_u, value v_vt) {
  CAMLparam5(v_ip, v_cplx, v_a, v_s, v_u);
  CAMLxparam1(v_vt);
  CAMLreturn(svd_stub(1, v_ip, v_cplx, v_a, v_s, v_u, v_vt));
}

CAMLprim value ojax_lapack_gesvd_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_gesvd(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value ojax_lapack_syevd(value v_ip, value v_cplx, value v_a,
                                 value v_w) {
  CAMLparam4(v_ip, v_cplx, v_a, v_w);
  int cplx = Bool_val(v_cplx);
  char prec = cplx ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, cplx ? "heevd" : "syevd");
  char jobz = (char)IP(v_ip, 0), uplo = (char)IP(v_ip, 1);
  LI n = IP(v_ip, 2), lda = IP(v_ip, 3);
  void *a = D(v_a), *w = D(v_w);
  caml_release_runtime_system();
  int info = ojax_lapack_call_syevd(prec, jobz, uplo, n, a, lda, w);
  FINISH("syevd");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_geev(value v_ip, value v_cplx, value v_a, value v_w,
                                value v_vl, value v_vr) {
  CAMLparam5(v_ip, v_cplx, v_a, v_w, v_vl);
  CAMLxparam1(v_vr);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "geev");
  char jobvl = (char)IP(v_ip, 0), jobvr = (char)IP(v_ip, 1);
  LI n = IP(v_ip, 2), lda = IP(v_ip, 3), ldvl = IP(v_ip, 4), ldvr = IP(v_ip, 5);
  void *a = D(v_a), *w = D(v_w), *vl = D(v_vl), *vr = D(v_vr);
  caml_release_runtime_system();
  int info = ojax_lapack_call_geev(prec, jobvl, jobvr, n, a, lda, w, vl, ldvl,
                                   vr, ldvr);
  FINISH("geev");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_geev_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_geev(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value ojax_lapack_gees(value v_ip, value v_cplx, value v_a, value v_w,
                                value v_vs, value v_sdim) {
  CAMLparam5(v_ip, v_cplx, v_a, v_w, v_vs);
  CAMLxparam1(v_sdim);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "gees");
  char jobvs = (char)IP(v_ip, 0);
  LI n = IP(v_ip, 1), lda = IP(v_ip, 2), ldvs = IP(v_ip, 3);
  void *a = D(v_a), *w = D(v_w), *vs = D(v_vs);
  LI *sdim = D(v_sdim);
  caml_release_runtime_system();
  int info = ojax_lapack_call_gees(prec, jobvs, n, a, lda, sdim, w, vs, ldvs);
  FINISH("gees");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_gees_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_gees(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value ojax_lapack_gehrd(value v_ip, value v_cplx, value v_a,
                                 value v_tau) {
  CAMLparam4(v_ip, v_cplx, v_a, v_tau);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "gehrd");
  LI n = IP(v_ip, 0), ilo = IP(v_ip, 1), ihi = IP(v_ip, 2), lda = IP(v_ip, 3);
  void *a = D(v_a), *tau = D(v_tau);
  caml_release_runtime_system();
  int info = ojax_lapack_call_gehrd(prec, n, ilo, ihi, a, lda, tau);
  FINISH("gehrd");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_sytrd(value v_ip, value v_cplx, value v_a, value v_d,
                                 value v_e, value v_tau) {
  CAMLparam5(v_ip, v_cplx, v_a, v_d, v_e);
  CAMLxparam1(v_tau);
  int cplx = Bool_val(v_cplx);
  char prec = cplx ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, cplx ? "hetrd" : "sytrd");
  char uplo = (char)IP(v_ip, 0);
  LI n = IP(v_ip, 1), lda = IP(v_ip, 2);
  void *a = D(v_a), *d = D(v_d), *e = D(v_e), *tau = D(v_tau);
  caml_release_runtime_system();
  int info = ojax_lapack_call_sytrd(prec, uplo, n, a, lda, d, e, tau);
  FINISH("sytrd");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_sytrd_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_sytrd(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

CAMLprim value ojax_lapack_trtrs(value v_ip, value v_cplx, value v_a,
                                 value v_b) {
  CAMLparam4(v_ip, v_cplx, v_a, v_b);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  void *fn = need(prec, "trtrs");
  char uplo = (char)IP(v_ip, 0), trans = (char)IP(v_ip, 1),
       diag = (char)IP(v_ip, 2);
  LI n = IP(v_ip, 3), nrhs = IP(v_ip, 4), lda = IP(v_ip, 5), ldb = IP(v_ip, 6),
     info = 0;
  void *a = D(v_a), *b = D(v_b);
  caml_release_runtime_system();
  CALL(fn, 13, &uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info, L1, L1,
       L1);
  caml_acquire_runtime_system();
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_gecon(value v_ip, value v_a, value v_anorm,
                                 value v_rcond) {
  CAMLparam4(v_ip, v_a, v_anorm, v_rcond);
  char prec = PREC_R(v_a);
  void *fn = need(prec, "gecon");
  char norm = (char)IP(v_ip, 0);
  LI n = IP(v_ip, 1), lda = IP(v_ip, 2), info = 0;
  void *a = D(v_a), *rcond = D(v_rcond);
  double danorm = Double_val(v_anorm);
  float sanorm = (float)danorm;
  void *anorm = prec == 's' ? (void *)&sanorm : (void *)&danorm;
  size_t nw = n < 1 ? 1 : (size_t)n;
  void *work = malloc(elt_size(prec) * 4 * nw);
  LI *iwork = malloc(sizeof(LI) * nw);
  if (!work || !iwork) {
    free(work);
    free(iwork);
    caml_failwith("ojax linalg: gecon workspace alloc failed");
  }
  caml_release_runtime_system();
  CALL(fn, 10, &norm, &n, a, &lda, anorm, rcond, work, iwork, &info, L1);
  caml_acquire_runtime_system();
  free(work);
  free(iwork);
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_trsm(value v_ip, value v_cplx, value v_a,
                                value v_b) {
  CAMLparam4(v_ip, v_cplx, v_a, v_b);
  char prec = Bool_val(v_cplx) ? PREC_C(v_a) : PREC_R(v_a);
  NEED_CHECK(prec, "trsm");
  char side = (char)IP(v_ip, 0), uplo = (char)IP(v_ip, 1),
       trans = (char)IP(v_ip, 2), diag = (char)IP(v_ip, 3);
  LI m = IP(v_ip, 4), n = IP(v_ip, 5), lda = IP(v_ip, 6), ldb = IP(v_ip, 7);
  void *a = D(v_a), *b = D(v_b);
  caml_release_runtime_system();
  ojax_lapack_call_trsm(prec, side, uplo, trans, diag, m, n, a, lda, b, ldb);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_lapack_gtsv(value v_ip, value v_cplx, value v_dl, value v_d,
                                value v_du, value v_b) {
  CAMLparam5(v_ip, v_cplx, v_dl, v_d, v_du);
  CAMLxparam1(v_b);
  char prec = Bool_val(v_cplx) ? PREC_C(v_d) : PREC_R(v_d);
  NEED_CHECK(prec, "gtsv");
  LI n = IP(v_ip, 0), nrhs = IP(v_ip, 1), ldb = IP(v_ip, 2);
  void *dl = D(v_dl), *d = D(v_d), *du = D(v_du), *b = D(v_b);
  caml_release_runtime_system();
  int info = ojax_lapack_call_gtsv(prec, n, nrhs, dl, d, du, b, ldb);
  FINISH("gtsv");
  CAMLreturn(Val_int(info));
}

CAMLprim value ojax_lapack_gtsv_bc(value *argv, int argc) {
  (void)argc;
  return ojax_lapack_gtsv(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}
