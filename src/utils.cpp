// A little library of low-level array manipulations and timers.
// For its embryonic self-test see ../test/testutils.cpp, which only tests
// the next235 for now.


// ------------ complex array utils

T TEMPLATE(relerrtwonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a, TEMPLATE(CPX,T)* b)
// ||a-b||_2 / ||a||_2
{
  T err = 0.0, nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    nrm += real(conj(a[m])*a[m]);
    TEMPLATE(CPX,T) diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err/nrm);
}
T TEMPLATE(errtwonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a, TEMPLATE(CPX,T)* b)
// ||a-b||_2
{
  T err = 0.0;   // compute error 2-norm
  for (BIGINT m=0; m<n; ++m) {
    TEMPLATE(CPX,T) diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err);
}
T TEMPLATE(twonorm,T)(BIGINT n, TEMPLATE(CPX,T)* a)
// ||a||_2
{
  T nrm = 0.0;
  for (BIGINT m=0; m<n; ++m)
    nrm += real(conj(a[m])*a[m]);
  return sqrt(nrm);
}
T TEMPLATE(infnorm,T)(BIGINT n, TEMPLATE(CPX,T)* a)
// ||a||_infty
{
  T nrm = 0.0;
  for (BIGINT m=0; m<n; ++m) {
    T aa = real(conj(a[m])*a[m]);
    if (aa>nrm) nrm = aa;
  }
  return sqrt(nrm);
}

void TEMPLATE(arrayrange,T)(BIGINT n, T* a, T *lo, T *hi)
// With a a length-n array, writes out min(a) to lo and max(a) to hi,
// so that all a values lie in [lo,hi].
{
  *lo = INFINITY; *hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    if (a[m]<*lo) *lo = a[m];
    if (a[m]>*hi) *hi = a[m];
  }
}

void TEMPLATE(indexedarrayrange,T)(BIGINT n, BIGINT* i, T* a, T *lo, T *hi)
// With i a list of n indices, and a an array of length max(i), writes out
// min(a(i)) to lo and max(a(i)) to hi, so that all a(i) values lie in [lo,hi].
{
  *lo = INFINITY; *hi = -INFINITY;
  for (BIGINT m=0; m<n; ++m) {
    T A=a[i[m]];
    if (A<*lo) *lo = A;
    if (A>*hi) *hi = A;
  }
}

void TEMPLATE(arraywidcen,T)(BIGINT n, T* a, T *w, T *c)
// Writes out w = half-width and c = center of an interval enclosing all a[n]'s
// Only chooses a nonzero center if this increases w by less than fraction
// ARRAYWIDCEN_GROWFRAC defined in defs.h.
// This prevents rephasings which don't grow nf by much. 6/8/17
{
  T lo,hi;
  TEMPLATE(arrayrange,T)(n,a,&lo,&hi);
  *w = (hi-lo)/2;
  *c = (hi+lo)/2;
  if (FABS(*c)<ARRAYWIDCEN_GROWFRAC*(*w)) {
    *w += FABS(*c);
    *c = 0.0;
  }
}
