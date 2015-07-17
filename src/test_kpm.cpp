#include <iomanip>
#include <cassert>
#include <complex>
#include "iostream_util.h"
#include "fastkpm.h"

using fkpm::Vec;
using fkpm::cx_float;
using fkpm::cx_double;

inline int pos_cubic(int x, int y, int z, int lx) { return x + lx * (y + lx * z); }
inline int pos_square(int x, int y, int lx) { int temp = x + lx * y; assert(temp>=0 && temp<lx*lx); return temp; }


template <typename T>
double exact_energy(arma::Mat<T> const& H, double kT, double mu) {
    return fkpm::electronic_grand_energy(arma::real(arma::eig_gen(H)), kT, mu);
}

void testExpansionCoeffs() {
    int M = 10;
    Vec<int> Mqs { 1*M, 10*M, 100*M, 1000*M };
    fkpm::EnergyScale es { -1, +1 };
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
    
    fkpm::RNG rng(0);
    std::uniform_int_distribution<uint32_t> uniform(0,n-1);
    fkpm::SpMatElems<double> elems(n, n, 1);
    for (int k = 1; k < 400; k++) {
        int i = uniform(rng);
        int j = uniform(rng);
        double v = k;
        elems.add(i, j, &v);
    }
    fkpm::SpMatBsr<double> H(elems);
    arma::sp_mat Ha = H.to_arma().st();
    
    for (int p = 0; p < Ha.n_nonzero; p++) {
        assert(Ha.row_indices[p] == H.col_idx[p]);
    }
    cout << "Column indices match.\n";
    for (int j = 0; j <= Ha.n_cols; j++) {
        assert(Ha.col_ptrs[j] == H.row_ptr[j]);
    }
    cout << "Row pointers match.\n";
}

template <typename T>
void testKPM1() {
    cout << "testKPM1: Energy/density matrix for simple Hamiltonian, templated input\n";
    int n = 4;
    fkpm::SpMatElems<T> elems(4, 4, 1);
    Vec<T> v {5, -5, 0, 0};
    elems.add(0, 0, &v[0]);
    elems.add(1, 1, &v[1]);
    elems.add(2, 2, &v[2]);
    elems.add(3, 3, &v[3]);
    fkpm::SpMatBsr<T> H(elems);
    auto g = [](double x) { return x*x; };
    auto f = [](double x) { return 2*x; }; // dg/dx
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    auto engine = fkpm::mk_engine<T>();
    engine->set_R_identity(n);
    engine->set_H(H, es);
    
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    
    auto mu = engine->moments(M);
    
    double E1 = fkpm::moment_product(g_c, mu);
    std::cout << std::setprecision(12);
    cout << "energy (v1) " << E1 << " expected 50.000432961 for M=1000\n";
    
    auto gamma = fkpm::moment_transform(mu, Mq);
    double E2 = fkpm::density_product(gamma, g, es);
    cout << "energy (v2) " << E2 << endl;
    
    auto D = H;
    engine->autodiff_matrix(g_c, D);

    cout << "derivative <";
    for (int i = 0; i < 4; i++)
        cout << *D(i, i) << " ";
    cout << ">\n";
    cout << "expected   <9.99980319956 -9.99980319958 8.42069190159e-14 8.42069190159e-14 >\n\n";
}

void testKPM2() {
    cout << "testKPM2: Energy/density matrix with various stochastic approximations\n";
    int n = 100;
    int s = n/4;
    double noise = 0.2;
    fkpm::RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    fkpm::SpMatElems<cx_double> elems(n, n, 1);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        cx_double v1 = 0, v2 = x, v3 = conj(x);
        elems.add(i, i, &v1);
        elems.add(i, j, &v2);
        elems.add(j, i, &v3);
    }
    fkpm::SpMatBsr<cx_double> H(elems);
    auto H_dense = H.to_arma_dense();
    
    double kT = 0.0;
    double mu = 0.2;
    
    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, kT, mu);
    auto f = std::bind(fkpm::fermi_density, _1, kT, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    int M = 2000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    auto D = H;
    
    double E1 = exact_energy(H_dense, kT, mu);
    double eps = 1e-6;
    int i=0, j=1;
    
    arma::sp_cx_mat dH(n, n);
    dH(i, j) = eps;
    dH(j, i) = eps;
    double dE_dH_1 = (exact_energy(H_dense+dH, kT, mu)-exact_energy(H_dense-dH, kT, mu)) / (2*eps);
    
    engine->set_R_identity(n);
    double E2 = fkpm::moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_2 = *D(i, j) + *D(j, i);
    
    engine->set_R_uncorrelated(n, s, rng);
    double E3 = fkpm::moment_product(g_c, engine->moments(M));
    engine->autodiff_matrix(g_c, D);
    auto dE_dH_3 = *D(i, j) + *D(j, i);
    
    Vec<int> groups(n);
    for (int i = 0; i < n; i++)
        groups[i] = i%s;
    engine->set_R_correlated(groups, rng);
    double E4 = fkpm::moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_4 = (*D(i, j) + *D(j, i));
    
    engine->moments(M);
    engine->autodiff_matrix(g_c, D);
    auto dE_dH_5 = (*D(i, j) + *D(j, i));
    
    cout << std::setprecision(15);
    cout << "Exact energy            " << E1 << endl;
    cout << "Det. KPM energy         " << E2 << endl;
    cout << "Stoch. energy (uncorr.) " << E3 << endl;
    cout << "Stoch. energy (corr.)   " << E4 << endl << endl;
    
    cout << "Exact deriv.            " << dE_dH_1 << endl;
    cout << "Det. KPM deriv.         " << dE_dH_2 << endl;
    cout << "Stoch. deriv. (uncorr.) " << dE_dH_3 << endl;
    cout << "Stoch. deriv. (corr.)   " << dE_dH_4 << endl;
    cout << "Autodif. deriv. (corr.) " << dE_dH_5 << endl << endl;
}

void testKPM3() {
    cout << "testKPM3: Comparing autodiff matrix with finite differencing\n";
    int n = 100;
    double noise = 0.2;
    fkpm::RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    fkpm::SpMatElems<cx_double> elems(n, n, 1);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        cx_double v1 = 0, v2 = x, v3 = conj(x);
        elems.add(i, i, &v1);
        elems.add(i, j, &v2);
        elems.add(j, i, &v3);
    }
    fkpm::SpMatBsr<cx_double> H(elems);
    
    double mu = 0.2;
    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, 0, mu);
    auto f = std::bind(fkpm::fermi_density, _1, 0, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    int M = 500;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_R_uncorrelated(n, 4, rng);
//    engine->set_R_identity(n);
    
    double eps = 1e-5;
    int i=0, j=1;
    
    auto finite_diff = [&](cx_double eps) {
        auto Hp = H;
        *Hp(i, j) += eps;
        *Hp(j, i) += conj(eps);
        engine->set_H(Hp, es);
        double Ep = fkpm::moment_product(g_c, engine->moments(M));
        auto Hm = H;
        *Hm(i, j) -= eps;
        *Hm(j, i) -= conj(eps);
        engine->set_H(Hm, es);
        double Em = fkpm::moment_product(g_c, engine->moments(M));
        return 0.5 * (Ep - Em) / (2.0*eps);
    };
    
    cx_double dE_dH = finite_diff(eps) - finite_diff(eps*cx_double(0, 1));
    
    engine->set_H(H, es);
    auto D1 = H;
    engine->moments(M);
    engine->autodiff_matrix(g_c, D1);
    
    std::cout << std::setprecision(10);
    cout << dE_dH << endl;
    cout << *D1(i, j) << endl << endl;
}

void testKPM4() {
    cout << "testKPM4: Energy/density matrix calculated using chunked R\n";
    int n = 20;
    int s = n/2;
    double noise = 0.2;
    fkpm::RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    fkpm::SpMatElems<cx_double> elems(n, n, 1);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        cx_double v1 = 0, v2 = x, v3 = conj(x);
        elems.add(i, i, &v1);
        elems.add(i, j, &v2);
        elems.add(j, i, &v3);
    }
    fkpm::SpMatBsr<cx_double> H(elems);
    auto H_dense = H.to_arma_dense();
    
    double kT = 0.0;
    double mu = 0.2;
    
    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, kT, mu);
    auto f = std::bind(fkpm::fermi_density, _1, 0, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    
    fkpm::RNG rng0 = rng;
    engine->set_R_uncorrelated(n, s, rng);
    double E0 = fkpm::moment_product(g_c, engine->moments(M));
    auto D0 = H;
    engine->autodiff_matrix(g_c, D0);
    
    rng = rng0;
    engine->set_R_uncorrelated(n, s, rng, 0, s/3);
    double E1 = fkpm::moment_product(g_c, engine->moments(M));
    auto D1a = H;
    engine->autodiff_matrix(g_c, D1a);
    engine->set_R_uncorrelated(n, s, rng, s/3, s);
    E1 += fkpm::moment_product(g_c, engine->moments(M));
    auto D1b = H;
    engine->autodiff_matrix(g_c, D1b);
    
    cout << "Energy identity (full) " << E0 << "\n";
    cout << "Energy identity (parts)" << E1 << "\n";
    cout << "Density matrix  (full) " << *D0(0,0) << "\n";
    cout << "Density matrix  (parts)" << (*D1a(0,0) + *D1b(0,0)) << "\n";
}


void testKPM5() {
    std::cout << std::endl << "testKPM5: Hall conductivity on square lattice." << std::endl;
    int lx = 100;
    int n  = lx * lx;
    int s  = 4;
    int M  = 40;
    int Mq = 2*M;
    
    double hopping     = 1.0;
    double B_over_phi0 = 0.05;
    double kT          = 0.001 * hopping;
    auto kernel        = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      -sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_xneg = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_ypos = -hopping;
            cx_double hop_yneg = -hopping;
            
            cx_double crt_xpos =  hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg = -hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos =  hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg = -hop_yneg * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;
            
            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);
            
            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);
            
            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);
            
            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    
    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);
    
    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR);
    auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, -3.5, 0.0, es, kernel);
    std::cout << "sigma_{xy}(mu = -3.5) = " << std::real(fkpm::moment_product(cmn, mu_xy)) << std::endl;
    std::cout << "expecting: 1.526508389 (increased moments would give 1)" << std::endl;
    
    cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, -2.8, 0.0, es, kernel);
    std::cout << "sigma_{xy}(mu = -2.8) = " << std::real(fkpm::moment_product(cmn, mu_xy)) << std::endl;
    std::cout << "expecting: 2.44531456  (increased moments would give 2)" << std::endl;
    
    H.clear();
    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
    
}

void test_AndersonModel() {
    int lx = 32;
    int n  = lx * lx * lx;
    int s  = 1;
    int num_replica = 10;
    int M  = 100;
    int Mq = 2*M;
    
    double hopping = 1.0;
    double W       = 12.0 * hopping;
    double kT      = 1.0 * hopping;
    double mu      =  0.0 * hopping;
    auto kernel    = fkpm::jackson_kernel(M);
    
    Vec<double> omega;
    for (double omega_elem = 0.2; omega_elem < 15.0; omega_elem += 0.2) omega.push_back(omega_elem);
    
    // need universal energy scale for all the replicas
    fkpm::EnergyScale es {-1.8*(2.0*hopping+W/2.0), 1.8*(2.0*hopping+W/2.0)};
    auto engine = fkpm::mk_engine<cx_double>();
    std::cout << es << std::endl;
    
    fkpm::RNG rng(0);
    std::uniform_real_distribution<double> dist_uniform(-W/2.0, W/2.0);
    
    //engine->set_R_identity(n);
    engine->set_R_uncorrelated(n, s, rng);
    //    Vec<int> groups(n);
    //    for (int i = 0; i < n; i++)
    //        groups[i] = i%s;
    //    engine->set_R_correlated(groups, rng);
    

    // Build Anderson model (Weisse paper, Eq 111)
    fkpm::SpMatElems<cx_double> elems_base(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            for (int zi = 0; zi < lx; zi++) {
                int pos_i = pos_cubic(xi, yi, zi, lx);
                int pos_j;
                cx_double v0 = -hopping;
                cx_double v1(0.0, hopping/sqrt(n));
                cx_double v2(0.0, -hopping/sqrt(n));
                
                Vec<int> xyzj = {(lx+xi-1)%lx, yi, zi};
                pos_j = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);      // +x
                elems_base.add(pos_i, pos_j, &v0);
                j1_elems.add(pos_i, pos_j, &v1);
                
                xyzj[0] = (lx+xi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -x
                elems_base.add(pos_i, pos_j, &v0);
                j1_elems.add(pos_i, pos_j, &v2);
                
                xyzj[0] = xi;
                xyzj[1] = (lx+yi-1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // +y
                elems_base.add(pos_i, pos_j, &v0);
                j2_elems.add(pos_i, pos_j, &v1);
                
                xyzj[1] = (lx+yi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -y
                elems_base.add(pos_i, pos_j, &v0);
                j2_elems.add(pos_i, pos_j, &v2);
                
                xyzj[1] = yi;
                xyzj[2] = (lx+zi-1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // +z
                elems_base.add(pos_i, pos_j, &v0);
                
                xyzj[2] = (lx+zi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -z
                elems_base.add(pos_i, pos_j, &v0);
            }
        }
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    
    Vec<Vec<cx_double>> gamma_jxy(Mq);
    Vec<double> optical;
    for (int i = 0; i < Mq; i++) {
        gamma_jxy[i].resize(Mq, cx_double(0.0,0.0));
    }
    optical.resize(omega.size(),0.0);
    
    cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    for (int replica = 0; replica < num_replica; replica++) {
        std::cout << std::endl << "replica #" << replica << std::endl;
        auto elems = elems_base;
        for (int xi = 0; xi < lx; xi++) {
            for (int yi = 0; yi < lx; yi++) {
                for (int zi = 0; zi < lx; zi++) {
                    int pos_i = pos_cubic(xi, yi, zi, lx);
                    cx_double epsiloni = dist_uniform(rng);
                    elems.add(pos_i, pos_i, &epsiloni);
                }
            }
        }
        fkpm::SpMatBsr<cx_double> H(elems);
        elems.clear();
        //auto H_dense = H.to_arma_dense();
        engine->set_H(H, es);
        
        Vec<Vec<cx_double>> mu_longitudinal = engine->moments2_v1(M, j1_BSR, j1_BSR);
        //Vec<Vec<cx_double>> mu_transverse   = engine->moments_tensor(M, j1_BSR, j2_BSR);
        
        auto temp = fkpm::moment_transform(mu_longitudinal, Mq, kernel);
        for (int i = 0; i < Mq; i++) {
            for (int j = 0; j < Mq; j++) {
                gamma_jxy[i][j] += temp[i][j];
            }
        }
        for (int i = 0; i < omega.size(); i++) {
            auto cmn    = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu, omega[i], es, kernel);
            optical[i] += std::real(fkpm::moment_product(cmn, mu_longitudinal));
        }
        
    }
    for (int i = 0; i < Mq; i++) {             // average over number of replicas
        for (int j = 0; j < Mq; j++) gamma_jxy[i][j] /= num_replica;
    }
    for (int i = 0; i < omega.size(); i++) optical[i] /= num_replica;
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    Vec<double> x, y;
    Vec<Vec<cx_double>> jxy;
    fkpm::density_function(gamma_jxy, es, x, y, jxy);
    std::ofstream fout1("jxy.dat");
    fout1 << std::scientific << std::right;
    //fout << std::setw(20) << "x" << std::setw(20) << "y" << std::setw(20) << "jxy" << std::endl;
    for (int i = 5; i < Mq-5; i++) {
        for (int j = 5; j < Mq-5; j++) {
            fout1 << std::setw(20) << x[i] << std::setw(20) << y[j] << std::setw(20) << std::real(jxy[i][j]) << std::endl;
        }
    }
    fout1.close();
    
    std::ofstream fout2("sigma.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
          << "omega" << std::setw(20) << "sigma" << std::endl;
    for (int i = 0; i < omega.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << omega[i]
              << std::setw(20) << optical[i] << std::endl;
    }
    fout2.close();
    
    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
    
}

void test_Hall_SquareLattice() {
    int lx = 100;
    int n  = lx * lx;
    int s  = 10;
    int M  = 200;
    int Mq = 2*M;
    
    double hopping     = 1.0;
    double B_over_phi0 = 0.05;
    double kT          = 0.001 * hopping;
    auto kernel        = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      -sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_xneg = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_ypos = -hopping;
            cx_double hop_yneg = -hopping;
            
            cx_double crt_xpos = -hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg =  hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos = -hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg =  hop_yneg * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;
            
            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);
            
            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);
            
            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);
            
            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    
    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);
    
    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);
    
    std::ofstream fout1("dos_hall.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();
    
    std::cout << "calculating transverse conductivity..." << std::endl;
    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR, 20);
    std::cout << "moments obtained." << std::endl;

    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    arma::Col<double> sigma_xy;
    sigma_xy.zeros(mu_list.size());
    
    for (int i = 0; i < mu_list.size(); i++) {
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
    }
    
    std::ofstream fout2("sigma_hall.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
          << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
              << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    
    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
    
}

// square lattice, each triangle threaded by quarter flux.
// for spinless electron, at half filling, sigma_{xy}=1
void test_PRL101_156402_v0() {
    int lx = 100;
    int n  = lx * lx;
    int s  = 10;
    int M  = 200;
    int Mq = 2*M;
    
    double hopping1 = 1.0;
    double hopping2 = 0.8;
    double flux     = 0.25;
    double kT       = 0.001 * hopping1;
    auto kernel     = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = -hopping1;
            cx_double hop_xneg = -hopping1;
            cx_double hop_ypos = -hopping1 * cx_double(cos(4.0*fkpm::Pi*flux*xi), sin(4.0*fkpm::Pi*flux*xi));
            cx_double hop_yneg = -hopping1 * cx_double(cos(4.0*fkpm::Pi*flux*xi),-sin(4.0*fkpm::Pi*flux*xi));
            cx_double hop_xpyp = -hopping2 * cx_double(cos(2.0*fkpm::Pi*flux*(2.0*xi+1.0)), sin(2.0*fkpm::Pi*flux*(2.0*xi+1.0)));
            cx_double hop_xpyn = -hopping2 * cx_double(cos(2.0*fkpm::Pi*flux*(2.0*xi-1.0)),-sin(2.0*fkpm::Pi*flux*(2.0*xi-1.0)));
            
            cx_double crt_xpos = -hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg =  hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos = -hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg =  hop_yneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xpyp = -hop_xpyp * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xpyn =  hop_xpyn * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;
            
            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);
            
            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);
            
            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);
            
            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);
            
            xyj[0] = (lx+xi-1)%lx;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpyp);
            j1_elems.add(pos_i, pos_j, &crt_xpyp);
            j2_elems.add(pos_i, pos_j, &crt_xpyp);
            
            xyj[0] = (lx+xi+1)%lx;
            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpyn);
            j1_elems.add(pos_i, pos_j, &crt_xpyn);
            j2_elems.add(pos_i, pos_j, &crt_xpyn);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    //fkpm::EnergyScale es {-5.0, 5.0};
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    
    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);
    
    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);
    
    std::ofstream fout1("dos_Martin_v0.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();
    
    std::cout << "calculating transverse conductivity..." << std::endl;
    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR, 20);
    std::cout << "moments obtained." << std::endl;
    
    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    arma::Col<double> sigma_xy;
    sigma_xy.zeros(mu_list.size());
    
    for (int i = 0; i < mu_list.size(); i++) {
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
    }
    
    std::ofstream fout2("sigma_Martin_v0.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
        << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    
    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
    
}


// bench mark Ivar Martin and Cristian's PRL 101, 156402 (2008)
// triangular lattice, chiral spin ordering
// S_1 = ( 1, 1, 1)
// S_2 = ( 1,-1,-1)
// S_3 = (-1, 1,-1)
// S_4 = (-1,-1, 1)
void test_PRL101_156402_v1() {
    int lx = 100;
    int n  = lx * lx;
    int s  = 40;
    int M  = 200;
    int Mq = M;
    
    double hopping = 1.0;
    double Jkondo  = 5.0;
    double kT      = 0.001 * hopping;
    double volume  = std::sqrt(3.0)/2.0;
    auto kernel    = fkpm::jackson_kernel(M);
    
    // Build current operators and the hopping part of H on triangular lattice
    fkpm::SpMatElems<cx_double> H_elems(2*n, 2*n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(2*n, 2*n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(2*n, 2*n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        cx_double v0 = -hopping;
        cx_double v1(0.0,  hopping/sqrt(n)/sqrt(volume));
        cx_double v2(0.0, -hopping/sqrt(n)/sqrt(volume));
        cx_double v3(0.0,  0.5*hopping/sqrt(n)/sqrt(volume));
        cx_double v4(0.0, -0.5*hopping/sqrt(n)/sqrt(volume));
        cx_double v5(0.0,  0.5*sqrt(3.0)*hopping/sqrt(n)/sqrt(volume));
        cx_double v6(0.0, -0.5*sqrt(3.0)*hopping/sqrt(n)/sqrt(volume));
        for (int yi = 0; yi < lx; yi++) {
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;
            
            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);      // +x
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v1);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v1);
            
            xyj[0] = (lx+xi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);    // -x
            H_elems.add(pos_i, pos_j, &v0);             // spin up
            j1_elems.add(pos_i, pos_j, &v2);
            H_elems.add(pos_i + n, pos_j + n, &v0);     // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v2);
            
            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v3);
            j2_elems.add(pos_i, pos_j, &v5);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v3);
            j2_elems.add(pos_i + n, pos_j + n, &v5);
            
            xyj[1] = (lx+yi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 4pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v4);
            j2_elems.add(pos_i, pos_j, &v6);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v4);
            j2_elems.add(pos_i + n, pos_j + n, &v6);
            
            xyj[0] = (lx+xi-1)%lx;
            xyj[1] = (lx+yi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 2pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v3);
            j2_elems.add(pos_i, pos_j, &v6);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v3);
            j2_elems.add(pos_i + n, pos_j + n, &v6);
            
            xyj[0] = (lx+xi+1)%lx;
            xyj[1] = (lx+yi-1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 5pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v4);
            j2_elems.add(pos_i, pos_j, &v5);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v4);
            j2_elems.add(pos_i + n, pos_j + n, &v5);
        }
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    
    //complete the hamiltonian with the Kondo terms
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double v1, v2, v3, v4;
            if (xi % 2 == 0 && yi % 2 == 0) {         // sublattice 1
                v1 = cx_double(-Jkondo, Jkondo);
                v2 = cx_double(-Jkondo,-Jkondo);
                v3 = cx_double(-Jkondo, 0.0);
                v4 = cx_double( Jkondo, 0.0);
            } else if (xi % 2 == 1 && yi % 2 == 0) {  // sublattice 2
                v1 = cx_double(-Jkondo,-Jkondo);
                v2 = cx_double(-Jkondo, Jkondo);
                v3 = cx_double( Jkondo, 0.0);
                v4 = cx_double(-Jkondo, 0.0);
            } else if (xi % 2 == 0 && yi % 2 == 1) {  // sublattice 3
                v1 = cx_double( Jkondo, Jkondo);
                v2 = cx_double( Jkondo,-Jkondo);
                v3 = cx_double( Jkondo, 0.0);
                v4 = cx_double(-Jkondo, 0.0);
            } else {                                  // sublattice 4
                v1 = cx_double( Jkondo,-Jkondo);
                v2 = cx_double( Jkondo, Jkondo);
                v3 = cx_double(-Jkondo, 0.0);
                v4 = cx_double( Jkondo, 0.0);
            }
            int pos_up = pos_square(xi, yi, lx);
            int pos_dn = pos_up + n;
            H_elems.add(pos_up, pos_dn, &v1);
            H_elems.add(pos_dn, pos_up, &v2);
            H_elems.add(pos_up, pos_up, &v3);
            H_elems.add(pos_dn, pos_dn, &v4);
        }
    }
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    //auto engine = fkpm::mk_engine<cx_double>();
    auto engine = fkpm::mk_engine_cpu<cx_double>();
    engine->set_H(H, es);
    
    fkpm::RNG rng(0);
    //engine->set_R_identity(2*n);
    engine->set_R_uncorrelated(2*n, s, rng);
    engine->R2 = engine->R;

    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);
    
    std::ofstream fout1("dos_Martin_v1.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();
    
    std::cout << "calculating transverse conductivity..." << std::endl;
    //auto mu_xx = engine->moments2_v1(M, j1_BSR, j1_BSR);
    auto mu_xy = engine->moments2_v2(M, j1_BSR, j2_BSR);
    
//    auto gamma_jxx = fkpm::moment_transform(mu_xx, Mq, kernel);
//    Vec<Vec<cx_double>> jxx;
//    Vec<double> x, y;
//    fkpm::density_function(gamma_jxx, es, x, y, jxx);
//    std::ofstream fout1("jxy_prl101_156402.dat");
//    fout1 << std::scientific << std::right;
//    for (int i = 5; i < Mq-5; i++) {
//        for (int j = 5; j < Mq-5; j++) {
//            fout1 << std::setw(20) << x[i] << std::setw(20) << y[j]
//            << std::setw(20) << std::real(jxx[i][j]) << std::endl;
//        }
//    }
//    fout1.close();
    
    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    //mu_list.push_back(0.0);
    arma::Col<double> sigma_xx;
    arma::Col<double> sigma_xy;
    sigma_xx.zeros(mu_list.size());
    sigma_xy.zeros(mu_list.size());
    // comment: slow in the loop below, probably something stupid here
    for (int i = 0; i < mu_list.size(); i++) {
        //std::cout << "mu=" << mu_list[i] << std::endl;
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
//        sigma_xx(i) = std::real(fkpm::moment_product(cmn, mu_xx));
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        //std::cout <<  "mu = " << mu_list[i] << ", sigma_xx = " << sigma_xx(i) << std::endl;
    }
    
    
    std::ofstream fout2("sigma_Martin_v1.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
          << "mu" << std::setw(20) << "sigma_xx" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
              << std::setw(20) << sigma_xx(i) << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    
    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
}


int main(int argc, char **argv) {
    // testExpansionCoeffs();
    // testMat();
    testKPM1<cx_float>();
    testKPM1<cx_double>();
    testKPM2();
    testKPM3();
    testKPM4();
    testKPM5();
    //test_AndersonModel();
    //test_PRL101_156402_v0();
    //test_PRL101_156402_v1();
    //test_Hall_SquareLattice();
}

