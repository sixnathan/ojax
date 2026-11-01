#ifndef OJAX_LAPACK_SHIM_H
#define OJAX_LAPACK_SHIM_H

#include <stdint.h>

void *ojax_lapack_symbol(char prec, const char *base);
void ojax_tridiagonal_perturbed(char prec, int n, int k_rhs, const void *dl,
                                const void *d, const void *du, const void *b,
                                void *x);
int32_t ojax_lapack_query_size(char prec, const void *q);
int32_t ojax_lapack_syevd_work(int32_t n);
int32_t ojax_lapack_heevd_work(int32_t n);
int32_t ojax_lapack_heevd_rwork(int32_t n);
int32_t ojax_lapack_evd_iwork(int32_t n);
int ojax_lapack_call_potrf(char prec, char uplo, int32_t n, void *a,
                           int32_t lda);
int ojax_lapack_call_getrf(char prec, int32_t m, int32_t n, void *a,
                           int32_t lda, int32_t *ipiv);
int ojax_lapack_call_geqrf(char prec, int32_t m, int32_t n, void *a,
                           int32_t lda, void *tau);
int ojax_lapack_call_geqp3(char prec, int32_t m, int32_t n, void *a,
                           int32_t lda, int32_t *jpvt, void *tau);
int ojax_lapack_call_orgqr(char prec, int32_t m, int32_t n, int32_t k, void *a,
                           int32_t lda, void *tau);
int ojax_lapack_call_ormqr(char prec, char side, char trans, int32_t m,
                           int32_t n, int32_t k, void *a, int32_t lda,
                           void *tau, void *c, int32_t ldc);
int ojax_lapack_call_gesdd(char prec, char jobz, int32_t m, int32_t n, void *a,
                           int32_t lda, void *s, void *u, int32_t ldu, void *vt,
                           int32_t ldvt);
int ojax_lapack_call_gesvd(char prec, char job, int32_t m, int32_t n, void *a,
                           int32_t lda, void *s, void *u, int32_t ldu, void *vt,
                           int32_t ldvt);
int ojax_lapack_call_syevd(char prec, char jobz, char uplo, int32_t n, void *a,
                           int32_t lda, void *w);
int ojax_lapack_call_geev(char prec, char jobvl, char jobvr, int32_t n, void *a,
                          int32_t lda, void *w, void *vl, int32_t ldvl,
                          void *vr, int32_t ldvr);
int ojax_lapack_call_gees(char prec, char jobvs, int32_t n, void *a,
                          int32_t lda, int32_t *sdim, void *w, void *vs,
                          int32_t ldvs);
int ojax_lapack_call_gehrd(char prec, int32_t n, int32_t ilo, int32_t ihi,
                           void *a, int32_t lda, void *tau);
int ojax_lapack_call_sytrd(char prec, char uplo, int32_t n, void *a,
                           int32_t lda, void *d, void *e, void *tau);
void ojax_lapack_call_trsm(char prec, char side, char uplo, char trans,
                           char diag, int32_t m, int32_t n, void *a,
                           int32_t lda, void *b, int32_t ldb);
int ojax_lapack_call_gtsv(char prec, int32_t n, int32_t nrhs, void *dl, void *d,
                          void *du, void *b, int32_t ldb);

#endif
