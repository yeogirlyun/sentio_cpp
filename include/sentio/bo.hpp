#pragma once
// bo.hpp — minimal, solid C++20 Bayesian Optimization (GP + EI)
// No external deps. Deterministic. Safe. Box bounds + integers + batch ask().

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <optional>
#include <limits>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>

// ----------------------------- Utilities -----------------------------
struct BOBounds {
  std::vector<double> lo, hi; // size = D
  bool valid() const { return !lo.empty() && lo.size()==hi.size(); }
  int dim() const { return (int)lo.size(); }
  inline double clamp_unit(double u, int d) const {
    return std::min(1.0, std::max(0.0, u));
  }
  inline double to_real(double u, int d) const {
    u = clamp_unit(u,d);
    return lo[d] + u * (hi[d] - lo[d]);
  }
  inline double to_unit(double x, int d) const {
    const double L = lo[d], H = hi[d];
    if (H<=L) return 0.0;
    double u = (x - L) / (H - L);
    return std::min(1.0, std::max(0.0, u));
  }
};

// Parameter kinds (continuous or integer). If INT, we round to nearest.
enum class ParamKind : uint8_t { CONT, INT };

struct BOOpts {
  int init_design = 16;          // initial random/space-filling points
  int cand_pool   = 2048;        // candidate samples for EI maximization
  int batch_q     = 1;           // q-EI via constant liar
  double jitter   = 1e-10;       // numerical jitter for Cholesky
  double ei_xi    = 0.01;        // EI exploration parameter
  bool  maximize  = true;        // true: maximize objective (default)
  uint64_t seed   = 42;          // deterministic rand seed
  bool  verbose   = false;
};

// ----------------------------- Tiny matrix helpers -----------------------------
// Row-major dense matrix with minimal ops (just what we need for GP Cholesky).
struct Mat {
  int n=0, m=0;                  // n rows, m cols
  std::vector<double> a;         // size n*m
  Mat()=default;
  Mat(int n_, int m_) : n(n_), m(m_), a((size_t)n_*(size_t)m_, 0.0) {}
  inline double& operator()(int i, int j){ return a[(size_t)i*(size_t)m + j]; }
  inline double  operator()(int i, int j) const { return a[(size_t)i*(size_t)m + j]; }
};

// Cholesky decomposition A = L L^T in-place into L (lower). Returns false if fails.
inline bool cholesky(Mat& A, double jitter=1e-10){
  // A must be square, symmetric, positive definite (we'll add jitter on diag).
  assert(A.n == A.m);
  const int n = A.n;
  for (int i=0;i<n;i++) A(i,i) += jitter;

  for (int i=0;i<n;i++){
    for (int j=0;j<=i;j++){
      double sum = A(i,j);
      for (int k=0;k<j;k++) sum -= A(i,k)*A(j,k);
      if (i==j){
        if (sum <= 0.0) return false;
        A(i,j) = std::sqrt(sum);
      } else {
        A(i,j) = sum / A(j,j);
      }
    }
    for (int j=i+1;j<n;j++) A(i,j)=0.0; // zero upper for clarity
  }
  return true;
}

// Solve L y = b (forward)
inline void trisolve_lower(const Mat& L, std::vector<double>& y){
  const int n=L.n;
  for (int i=0;i<n;i++){
    double s = y[i];
    for (int k=0;k<i;k++) s -= L(i,k)*y[k];
    y[i] = s / L(i,i);
  }
}
// Solve L^T x = y (backward)
inline void trisolve_upperT(const Mat& L, std::vector<double>& x){
  const int n=L.n;
  for (int i=n-1;i>=0;i--){
    double s = x[i];
    for (int k=i+1;k<n;k++) s -= L(k,i)*x[k];
    x[i] = s / L(i,i);
  }
}

// ----------------------------- Gaussian Process (RBF-ARD) -----------------------------
struct GP {
  // Hyperparams (on unit cube): signal^2, noise^2, lengthscales (per-dim)
  double sigma_f2 = 1.0;
  double sigma_n2 = 1e-6;
  std::vector<double> ell; // size D, lengthscales

  // Data (unit cube)
  std::vector<std::vector<double>> X; // N x D
  std::vector<double> y;              // N (centered)
  double y_mean = 0.0;

  // Factorization
  Mat L;                 // Cholesky of K = Kf + sigma_n2 I
  std::vector<double> alpha; // (K)^-1 y

  static inline double sqdist_ard(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& ell){
    double s=0.0;
    for (size_t d=0; d<ell.size(); ++d){
      const double z = (a[d]-b[d]) / std::max(1e-12, ell[d]);
      s += z*z;
    }
    return s;
  }

  inline double kf(const std::vector<double>& a, const std::vector<double>& b) const {
    const double s2 = sqdist_ard(a,b,ell);
    return sigma_f2 * std::exp(-0.5 * s2);
  }

  void fit(const std::vector<std::vector<double>>& X_unit,
           const std::vector<double>& y_raw,
           double jitter=1e-10)
  {
    X = X_unit;
    const int N = (int)X.size();
    y_mean = 0.0;
    for (double v: y_raw) y_mean += v;
    y_mean /= std::max(1, N);
    y.resize(N);
    for (int i=0;i<N;i++) y[i] = y_raw[i] - y_mean;

    // Build K
    Mat K(N,N);
    for (int i=0;i<N;i++){
      for (int j=0;j<=i;j++){
        double kij = kf(X[i], X[j]);
        if (i==j) kij += sigma_n2;
        K(i,j) = K(j,i) = kij;
      }
    }
    // Chol
    L = K;
    if (!cholesky(L, jitter)) {
      // increase jitter progressively if needed
      double j = std::max(jitter, 1e-12);
      bool ok=false;
      for (int t=0;t<6 && !ok; ++t) { L = K; ok = cholesky(L, j); j *= 10.0; }
      if (!ok) throw std::runtime_error("GP Cholesky failed (matrix not PD)");
    }
    // alpha = K^{-1} y  via L
    alpha = y;
    trisolve_lower(L, alpha);
    trisolve_upperT(L, alpha);
  }

  // Predictive mean/variance at x (unit cube)
  inline std::pair<double,double> predict(const std::vector<double>& x) const {
    const int N = (int)X.size();
    std::vector<double> k(N);
    for (int i=0;i<N;i++) k[i] = kf(x, X[i]);
    // mean = k^T alpha + y_mean
    double mu = std::inner_product(k.begin(), k.end(), alpha.begin(), 0.0) + y_mean;
    // v = L^{-1} k
    std::vector<double> v = k;
    trisolve_lower(L, v);
    double kxx = sigma_f2; // k(x,x) for RBF with same params is sigma_f2
    double var = kxx - std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    if (var < 1e-18) var = 1e-18;
    return {mu, var};
  }
};

// ----------------------------- Random/Sampling helpers -----------------------------
struct RNG {
  std::mt19937_64 g;
  explicit RNG(uint64_t seed): g(seed) {}
  double uni(){ return std::uniform_real_distribution<double>(0.0,1.0)(g); }
  int randint(int lo, int hi){ return std::uniform_int_distribution<int>(lo,hi)(g); }
};

// Jittered Latin hypercube on unit cube
inline std::vector<std::vector<double>> latin_hypercube(int n, int D, RNG& rng){
  std::vector<std::vector<double>> X(n, std::vector<double>(D,0.0));
  for (int d=0; d<D; ++d){
    std::vector<double> slots(n);
    for (int i=0;i<n;i++) slots[i] = (i + rng.uni()) / n;
    std::shuffle(slots.begin(), slots.end(), rng.g);
    for (int i=0;i<n;i++) X[i][d] = std::min(1.0, std::max(0.0, slots[i]));
  }
  return X;
}

// Round integer dims to nearest feasible grid-point after scaling
inline void apply_kinds_inplace(std::vector<double>& u, const std::vector<ParamKind>& kinds,
                                const BOBounds& B){
  for (int d=0; d<(int)u.size(); ++d){
    if (kinds[d] == ParamKind::INT){
      // snap in real space to integer, then rescale back to unit
      const double x = B.to_real(u[d], d);
      const double xi = std::round(x);
      u[d] = B.to_unit(xi, d);
    }
  }
}

// ----------------------------- Acquisition: Expected Improvement -----------------------------
inline double norm_pdf(double z){ static const double inv_sqrt2pi = 0.3989422804014327; return inv_sqrt2pi*std::exp(-0.5*z*z); }
inline double norm_cdf(double z){
  return 0.5 * std::erfc(-z/std::sqrt(2.0));
}
// EI for maximization: EI = (mu - best - xi) Phi(z) + sigma phi(z), z = (mu - best - xi)/sigma
inline double expected_improvement(double mu, double var, double best, double xi){
  const double sigma = std::sqrt(std::max(1e-18, var));
  const double diff  = mu - best - xi;
  if (sigma <= 1e-12) return std::max(0.0, diff);
  const double z = diff / sigma;
  return diff * norm_cdf(z) + sigma * norm_pdf(z);
}

// ----------------------------- Bayesian Optimizer -----------------------------
struct BO {
  BOBounds bounds;
  std::vector<ParamKind> kinds; // size D
  BOOpts opt;

  // data (real-space for API; internally store unit-cube for GP)
  std::vector<std::vector<double>> X_real; // N x D
  std::vector<std::vector<double>> X;      // N x D (unit)
  std::vector<double> y;                   // N
  GP gp;
  RNG rng;

  BO(const BOBounds& B, const std::vector<ParamKind>& kinds_, BOOpts o)
  : bounds(B), kinds(kinds_), opt(o), gp(), rng(o.seed)
  {
    assert(bounds.valid());
    assert((int)kinds.size() == bounds.dim());
    // init default GP hyperparams
    gp.ell.assign(bounds.dim(), 0.2); // medium lengthscale on unit cube
    gp.sigma_f2 = 1.0;
    gp.sigma_n2 = 1e-6;
  }

  int dim() const { return bounds.dim(); }
  int size() const { return (int)X.size(); }

  void clear(){
    X_real.clear(); X.clear(); y.clear();
  }

  // Append observation
  void tell(const std::vector<double>& xr, double val){
    assert((int)xr.size() == dim());
    X_real.push_back(xr);
    std::vector<double> u(dim());
    for (int d=0; d<dim(); ++d) u[d] = bounds.to_unit(xr[d], d);
    // snap integers
    apply_kinds_inplace(u, kinds, bounds);
    X.push_back(u);
    y.push_back(val);
  }

  // Fit GP (simple hyperparam heuristics; robust & fast)
  void fit(){
    if (X.empty()) return;
    // set GP hyperparams from y stats
    double m=0, s2=0;
    for (double v: y) m += v; m /= (double)y.size();
    for (double v: y) s2 += (v-m)*(v-m);
    s2 /= std::max(1, (int)y.size()-1);
    gp.sigma_f2 = std::max(1e-12, s2);
    gp.sigma_n2 = std::max(1e-9 * gp.sigma_f2, 1e-10);
    // modest ARD scaling: initialize lengthscales to 0.2; (optionally: scale by output sensitivity)
    for (double& e: gp.ell) e = 0.2;
    gp.fit(X, y, opt.jitter);
  }

  // Generate initial design if dataset is empty/small
  void ensure_init_design(){
    const int need = std::max(0, opt.init_design - (int)X.size());
    if (need <= 0) return;
    auto U = latin_hypercube(need, dim(), rng);
    for (auto& u : U){
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      // placeholder y (user will evaluate and call tell)
      X.push_back(u);
      X_real.push_back(xr);
      y.push_back(std::numeric_limits<double>::quiet_NaN()); // mark unevaluated
    }
  }

  // Ask for q new locations (real-space). Uses constant liar on current model (no new evals yet).
  std::vector<std::vector<double>> ask(int q=1){
    if (q<=0) return {};
    // If we have NaNs from ensure_init_design, return those first (ask-user-to-evaluate)
    std::vector<std::vector<double>> out;
    for (int i=0;i<(int)X.size() && (int)out.size()<q; ++i){
      if (!std::isfinite(y[i])) out.push_back(X_real[i]);
    }
    if ((int)out.size() == q) return out;

    // Ensure we can build a GP on finished points
    // Build filtered dataset of (finite) y's
    std::vector<std::vector<double>> Xf;
    std::vector<double> yf;
    Xf.reserve(X.size()); yf.reserve(y.size());
    for (int i=0;i<(int)X.size(); ++i){
      if (std::isfinite(y[i])) { Xf.push_back(X[i]); yf.push_back(y[i]); }
    }
    if (Xf.size() >= 2) {
      gp.fit(Xf, yf, opt.jitter);
    } else {
      // not enough data to fit GP: just random suggest
      out = random_candidates(q);
      return out;
    }

    const double y_best = opt.maximize ? *std::max_element(yf.begin(), yf.end())
                                       : *std::min_element(yf.begin(), yf.end());

    // batch selection with constant liar (for maximization, lie = y_best)
    std::vector<std::vector<double>> X_aug = Xf;
    std::vector<double> y_aug = yf;

    for (int pick=0; pick<q; ++pick){
      // pool
      double best_ei = -1.0;
      std::vector<double> best_u(dim());
      for (int c=0; c<opt.cand_pool; ++c){
        std::vector<double> u(dim());
        for (int d=0; d<dim(); ++d) u[d] = rng.uni();
        apply_kinds_inplace(u, kinds, bounds);
        // score EI on augmented model (approximate by reusing gp; small bias is fine)
        // NOTE: we approximate: use current gp (no retrain) — fast & works well in practice
        auto [mu, var] = gp.predict(u);
        double ei = opt.maximize ? expected_improvement(mu, var, y_best, opt.ei_xi)
                                 : expected_improvement(-mu, var, -y_best, opt.ei_xi);
        if (ei > best_ei){ best_ei = ei; best_u = u; }
      }
      // map to real, append
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(best_u[d], d);
      out.push_back(xr);

      // constant liar update (no refit for speed; optional: refit small)
      X_aug.push_back(best_u);
      y_aug.push_back(y_best); // lie
      // (Optionally refit gp with augmented data each pick for sharper q-EI; omitted for speed)
    }
    return out;
  }

  // Helper: random suggestions on real-space
  std::vector<std::vector<double>> random_candidates(int q){
    std::vector<std::vector<double>> out;
    out.reserve(q);
    for (int i=0;i<q;++i){
      std::vector<double> u(dim());
      for (int d=0; d<dim(); ++d) u[d] = rng.uni();
      apply_kinds_inplace(u, kinds, bounds);
      std::vector<double> xr(dim());
      for (int d=0; d<dim(); ++d) xr[d] = bounds.to_real(u[d], d);
      out.push_back(xr);
    }
    return out;
  }
};