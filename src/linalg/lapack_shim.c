#include <stdint.h>
#include <stdlib.h>

#ifdef __APPLE__
#define OJAX_HAVE_ACCELERATE 1
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
_Static_assert(sizeof(__LAPACK_int) == sizeof(int32_t),
               "ojax linalg seam requires 32-bit LAPACK int");
#else
#define OJAX_HAVE_ACCELERATE 0
#endif

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#define IP(v, i) ((int)Int_val(Field((v), (i))))
#define D(v) ((double *)Caml_ba_data_val(v))
#define I32(v) ((int32_t *)Caml_ba_data_val(v))

CAMLprim value ojax_lapack_available(value v_unit) {
  CAMLparam1(v_unit);
  CAMLreturn(Val_bool(OJAX_HAVE_ACCELERATE));
}

CAMLprim value ojax_lapack_abi_int_size(value v_unit) {
  CAMLparam1(v_unit);
#if OJAX_HAVE_ACCELERATE
  CAMLreturn(Val_int((int)sizeof(__LAPACK_int)));
#else
  CAMLreturn(Val_int(0));
#endif
}

#if OJAX_HAVE_ACCELERATE

CAMLprim value ojax_lapack_potrf(value v_ip, value v_a) {
  CAMLparam2(v_ip, v_a);
  char uplo = (char)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  double *a = D(v_a);
  __LAPACK_int info = 0;
  caml_release_runtime_system();
  dpotrf_(&uplo, &n, a, &lda, &info);
  caml_acquire_runtime_system();
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_getrf(value v_ip, value v_a, value v_ipiv) {
  CAMLparam3(v_ip, v_a, v_ipiv);
  __LAPACK_int m = (__LAPACK_int)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  double *a = D(v_a);
  __LAPACK_int *ipiv = (__LAPACK_int *)I32(v_ipiv);
  __LAPACK_int info = 0;
  caml_release_runtime_system();
  dgetrf_(&m, &n, a, &lda, ipiv, &info);
  caml_acquire_runtime_system();
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_geqrf(value v_ip, value v_a, value v_tau) {
  CAMLparam3(v_ip, v_a, v_tau);
  __LAPACK_int m = (__LAPACK_int)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  double *a = D(v_a);
  double *tau = D(v_tau);
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dgeqrf_(&m, &n, a, &lda, tau, &wkopt, &lwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: geqrf workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_orgqr(value v_ip, value v_a, value v_tau) {
  CAMLparam3(v_ip, v_a, v_tau);
  __LAPACK_int m = (__LAPACK_int)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int k = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 3);
  double *a = D(v_a);
  double *tau = D(v_tau);
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dorgqr_(&m, &n, &k, a, &lda, tau, &wkopt, &lwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: orgqr workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_gesdd(value v_ip, value v_a, value v_s, value v_u,
                                 value v_vt) {
  CAMLparam5(v_ip, v_a, v_s, v_u, v_vt);
  char jobz = (char)IP(v_ip, 0);
  __LAPACK_int m = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 3);
  __LAPACK_int ldu = (__LAPACK_int)IP(v_ip, 4);
  __LAPACK_int ldvt = (__LAPACK_int)IP(v_ip, 5);
  double *a = D(v_a);
  double *s = D(v_s);
  double *u = D(v_u);
  double *vt = D(v_vt);
  __LAPACK_int mn = m < n ? m : n;
  if (mn < 1) mn = 1;
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  __LAPACK_int *iwork = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  iwork = (__LAPACK_int *)malloc(sizeof(__LAPACK_int) * (size_t)(8 * mn));
  if (iwork == NULL)
    alloc_failed = 1;
  else {
    dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &wkopt, &lwork, iwork,
            &info);
    if (info == 0) {
      lwork = (__LAPACK_int)wkopt;
      if (lwork < 1) lwork = 1;
      work = (double *)malloc(sizeof(double) * (size_t)lwork);
      if (work == NULL)
        alloc_failed = 1;
      else {
        dgesdd_(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork,
                iwork, &info);
        free(work);
      }
    }
    free(iwork);
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: gesdd workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_syevd(value v_ip, value v_a, value v_w) {
  CAMLparam3(v_ip, v_a, v_w);
  char jobz = (char)IP(v_ip, 0);
  char uplo = (char)IP(v_ip, 1);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 3);
  double *a = D(v_a);
  double *w = D(v_w);
  __LAPACK_int info = 0, lwork = -1, liwork = -1, iwkopt = 0;
  double wkopt = 0.0;
  double *work = NULL;
  __LAPACK_int *iwork = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dsyevd_(&jobz, &uplo, &n, a, &lda, w, &wkopt, &lwork, &iwkopt, &liwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    liwork = iwkopt;
    if (liwork < 1) liwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    iwork = (__LAPACK_int *)malloc(sizeof(__LAPACK_int) * (size_t)liwork);
    if (work == NULL || iwork == NULL)
      alloc_failed = 1;
    else
      dsyevd_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, iwork, &liwork,
              &info);
    free(work);
    free(iwork);
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: syevd workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_geev(value v_ip, value v_a, value v_wri, value v_vl,
                                value v_vr) {
  CAMLparam5(v_ip, v_a, v_wri, v_vl, v_vr);
  char jobvl = (char)IP(v_ip, 0);
  char jobvr = (char)IP(v_ip, 1);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 3);
  __LAPACK_int ldvl = (__LAPACK_int)IP(v_ip, 4);
  __LAPACK_int ldvr = (__LAPACK_int)IP(v_ip, 5);
  double *a = D(v_a);
  double *wri = D(v_wri);
  double *wr = wri;
  double *wi = wri + n;
  double *vl = D(v_vl);
  double *vr = D(v_vr);
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, &wkopt,
         &lwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work,
             &lwork, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: geev workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_gees(value v_ip, value v_a, value v_wri, value v_vs,
                                value v_sdim) {
  CAMLparam5(v_ip, v_a, v_wri, v_vs, v_sdim);
  char jobvs = (char)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int ldvs = (__LAPACK_int)IP(v_ip, 3);
  double *a = D(v_a);
  double *wri = D(v_wri);
  double *wr = wri;
  double *wi = wri + n;
  double *vs = D(v_vs);
  __LAPACK_int *sdim = (__LAPACK_int *)I32(v_sdim);
  char sort = 'N';
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dgees_(&jobvs, &sort, NULL, &n, a, &lda, sdim, wr, wi, vs, &ldvs, &wkopt,
         &lwork, NULL, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dgees_(&jobvs, &sort, NULL, &n, a, &lda, sdim, wr, wi, vs, &ldvs, work,
             &lwork, NULL, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: gees workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_gehrd(value v_ip, value v_a, value v_tau) {
  CAMLparam3(v_ip, v_a, v_tau);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 0);
  __LAPACK_int ilo = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int ihi = (__LAPACK_int)IP(v_ip, 2);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 3);
  double *a = D(v_a);
  double *tau = D(v_tau);
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dgehrd_(&n, &ilo, &ihi, a, &lda, tau, &wkopt, &lwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dgehrd_(&n, &ilo, &ihi, a, &lda, tau, work, &lwork, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: gehrd workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_sytrd(value v_ip, value v_a, value v_d, value v_e,
                                 value v_tau) {
  CAMLparam5(v_ip, v_a, v_d, v_e, v_tau);
  char uplo = (char)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  double *a = D(v_a);
  double *d = D(v_d);
  double *e = D(v_e);
  double *tau = D(v_tau);
  __LAPACK_int info = 0, lwork = -1;
  double wkopt = 0.0;
  double *work = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  dsytrd_(&uplo, &n, a, &lda, d, e, tau, &wkopt, &lwork, &info);
  if (info == 0) {
    lwork = (__LAPACK_int)wkopt;
    if (lwork < 1) lwork = 1;
    work = (double *)malloc(sizeof(double) * (size_t)lwork);
    if (work == NULL)
      alloc_failed = 1;
    else {
      dsytrd_(&uplo, &n, a, &lda, d, e, tau, work, &lwork, &info);
      free(work);
    }
  }
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: sytrd workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_trtrs(value v_ip, value v_a, value v_b) {
  CAMLparam3(v_ip, v_a, v_b);
  char uplo = (char)IP(v_ip, 0);
  char trans = (char)IP(v_ip, 1);
  char diag = (char)IP(v_ip, 2);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 3);
  __LAPACK_int nrhs = (__LAPACK_int)IP(v_ip, 4);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 5);
  __LAPACK_int ldb = (__LAPACK_int)IP(v_ip, 6);
  double *a = D(v_a);
  double *b = D(v_b);
  __LAPACK_int info = 0;
  caml_release_runtime_system();
  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info);
  caml_acquire_runtime_system();
  CAMLreturn(Val_int((int)info));
}

CAMLprim value ojax_lapack_gecon(value v_ip, value v_a, value v_anorm,
                                 value v_rcond) {
  CAMLparam4(v_ip, v_a, v_anorm, v_rcond);
  char norm = (char)IP(v_ip, 0);
  __LAPACK_int n = (__LAPACK_int)IP(v_ip, 1);
  __LAPACK_int lda = (__LAPACK_int)IP(v_ip, 2);
  double *a = D(v_a);
  double anorm = Double_val(v_anorm);
  double *rcond = D(v_rcond);
  __LAPACK_int nw = n < 1 ? 1 : n;
  __LAPACK_int info = 0;
  double *work = NULL;
  __LAPACK_int *iwork = NULL;
  int alloc_failed = 0;
  caml_release_runtime_system();
  work = (double *)malloc(sizeof(double) * (size_t)(4 * nw));
  iwork = (__LAPACK_int *)malloc(sizeof(__LAPACK_int) * (size_t)nw);
  if (work == NULL || iwork == NULL)
    alloc_failed = 1;
  else
    dgecon_(&norm, &n, a, &lda, &anorm, rcond, work, iwork, &info);
  free(work);
  free(iwork);
  caml_acquire_runtime_system();
  if (alloc_failed) caml_failwith("ojax linalg: gecon workspace alloc failed");
  CAMLreturn(Val_int((int)info));
}

#else

static value ojax_lapack_unavail(void) {
  caml_failwith("ojax linalg: Accelerate unavailable on this platform");
  return Val_unit;
}

CAMLprim value ojax_lapack_potrf(value a, value b) {
  (void)a;
  (void)b;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_getrf(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_geqrf(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_orgqr(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_gesdd(value a, value b, value c, value d, value e) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_syevd(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_geev(value a, value b, value c, value d, value e) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_gees(value a, value b, value c, value d, value e) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_gehrd(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_sytrd(value a, value b, value c, value d, value e) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_trtrs(value a, value b, value c) {
  (void)a;
  (void)b;
  (void)c;
  return ojax_lapack_unavail();
}
CAMLprim value ojax_lapack_gecon(value a, value b, value c, value d) {
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  return ojax_lapack_unavail();
}

#endif
