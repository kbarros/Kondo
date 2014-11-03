#include <iomanip>
#include <cassert>
#include "iostream_util.h"
#include "fastkpm.h"


using namespace fkpm;

template <typename T>
double exact_energy(arma::Mat<T> const& H, double kB_T, double mu) {
    return electronic_grand_energy(arma::real(arma::eig_gen(H)), kB_T, mu);
}

void testExpansionCoeffs() {
    int M = 10;
    Vec<int> Mqs { 1*M, 10*M, 100*M, 1000*M };
    EnergyScale es { -1, +1 };
    for (int Mq : Mqs) {
        auto f = [](double e) { return e < 0.12 ? e : 0; };
        auto c = expansion_coefficients(M, Mq, f, es);
        std::cout << "quad. points=" << Mq << ", c=" << c << std::endl;
        Vec<double> c_exact { -0.31601, 0.4801, -0.185462, -0.000775683, 0.0255405, 0.000683006,
            -0.00536911, -0.000313801, 0.000758494, 0.0000425209 };
        std::cout << "mathematica c = " << c_exact << std::endl;
    }
}

void testMat() {
    int n = 20;
    SpMatCoo<double> H(n, n);
    
    RNG rng(0);
    std::uniform_int_distribution<uint32_t> uniform(0,n-1);
    for (int k = 1; k < 400; k++) {
        int i = uniform(rng);
        int j = uniform(rng);
        H.add(i, j, k);
    }
    
    arma::sp_mat Ha = H.to_arma().st();
    SpMatCsr<double> Hc(H);
    
    for (int p = 0; p < Ha.n_nonzero; p++) {
        assert(Ha.row_indices[p] == Hc.col_idx[p]);
    }
    cout << "Column indices match.\n";
    for (int j = 0; j <= Ha.n_cols; j++) {
        assert(Ha.col_ptrs[j] == Hc.row_ptr[j]);
    }
    cout << "Row pointers match.\n";
}

template <typename T>
void testKPM1() {
    int n = 4;
    SpMatCoo<T> H(n, n);
    H.add(0, 0, 5.0);
    H.add(1, 1, -5.0);
    H.add(2, 2, 0);
    H.add(3, 3, 0);
    auto g = [](double x) { return x*x; };
    auto f = [](double x) { return 2*x; }; // dg/dx
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(H, extra, tolerance);
    auto engine = mk_engine<T>();
    engine->set_R_identity(n);
    engine->set_H(H, es);
    
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    
    auto mu = engine->moments(M);
    double E1 = moment_product(g_c, mu);
    cout << "energy (v1) " << E1 << " expected 50.0004 for M=1000\n";
    
    auto gamma = moment_transform(mu, Mq);
    double E2 = density_product(gamma, g, es);
    cout << "energy (v2) " << E2 << endl;
    
    auto D = engine->Hs;
    engine->autodiff_matrix(g_c, D);
    cout << "derivative <";
    for (int i = 0; i < 4; i++)
        cout << D(i, i) << " ";
    cout << "> expected <10, -10, 0, 0>\n";
}

void testKPM2() {
    int n = 100;
    int s = n/4;
    double noise = 0.2;
    RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    SpMatCoo<cx_double> H(n, n);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        H.add(i, i, 0);
        H.add(i, j, x);
        H.add(j, i, conj(x));
    }
    auto H_dense = H.to_arma_dense();
    
    double kB_T = 0.0;
    double mu = 0.2;
    
    using std::placeholders::_1;
    auto g = std::bind(fermi_energy, _1, kB_T, mu);
    auto f = std::bind(fermi_density, _1, kB_T, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(H, extra, tolerance);
    int M = 2000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = mk_engine_cx();
    engine->set_H(H, es);
    auto D = engine->Hs;
    
    double E1 = exact_energy(H_dense, kB_T, mu);
    double eps = 1e-6;
    int i=0, j=1;
    arma::sp_cx_mat dH(n, n);
    dH(i, j) = eps;
    dH(j, i) = eps;
    double dE_dH_1 = (exact_energy(H_dense+dH, kB_T, mu)-exact_energy(H_dense-dH, kB_T, mu)) / (2*eps);
    
    engine->set_R_identity(n);
    double E2 = moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    double dE_dH_2 = (D(i, j) + D(j, i)).real();
    
    engine->set_R_uncorrelated(n, s, rng);
    double E3 = moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_3 = (D(i, j) + D(j, i)).real();
    
    Vec<int> groups(n);
    for (int i = 0; i < n; i++)
        groups[i] = i%s;
    engine->set_R_correlated(groups, rng);
    double E4 = moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_4 = (D(i, j) + D(j, i)).real();
    
    engine->autodiff_matrix(g_c, D);
    auto dE_dH_5 = (D(i, j) + D(j, i)).real();

    cout << "Exact energy            " << E1 << endl;
    cout << "Det. KPM energy         " << E2 << endl;
    cout << "Stoch. energy (uncorr.) " << E3 << endl;
    cout << "Stoch. energy (corr.)   " << E4 << endl << endl;
    
    cout << "Exact deriv.            " << dE_dH_1 << endl;
    cout << "Det. KPM deriv.         " << dE_dH_2 << endl;
    cout << "Stoch. deriv. (uncorr.) " << dE_dH_3 << endl;
    cout << "Stoch. deriv. (corr.)   " << dE_dH_4 << endl;
    cout << "Autodif. deriv. (corr.) " << dE_dH_5 << endl;
}

int main(int argc,char **argv) {
    testMat();
    testExpansionCoeffs();
    testKPM1<double>();
    testKPM2();
}

