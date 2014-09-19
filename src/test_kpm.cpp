//
//  test.cpp
//  kondo
//
//  Created by Kipton Barros on 6/19/14.
//
//

#include <iomanip>
#include "iostream_util.h"
#include "fastkpm.h"


namespace tibidy {
    using namespace fkpm;
    
    template <typename T>
    arma::Mat<T> sparse_to_dense(arma::SpMat<T> that) {
        int m = that.n_rows;
        return arma::eye<arma::Mat<T>>(m, m) * that;
    }
    
    template <typename T>
    double exactEnergy(arma::SpMat<T> H, double kB_T, double mu) {
        auto eigs = arma::eig_gen(sparse_to_dense(H));
        double ret = 0.0;
        for (auto x : eigs) {
            ret += fermi_energy(x.real(), kB_T, mu);
        }
        return ret;
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
    
    void testKPM1() {
        int n = 4;
        SpMatCoo<arma::cx_double> H(n, n);
        H.add(0, 0, 5.0);
        H.add(1, 1, -5.0);
        auto g = [](double x) { return x*x; };
        auto f = [](double x) { return 2*x; }; // dg/dx
        
        double extra = 0.1;
        double tolerance = 1e-2;
        auto es = energy_scale(H, extra, tolerance);
        auto engine = mk_engine_cx(n, n);
        engine->set_H(H, es);
        engine->set_R_identity();
        
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
        
        engine->stoch_orbital(g_c);
        double E3 = arma::cdot(engine->R, engine->xi).real();
        cout << "energy (v3) " << E3 << endl;
        
        engine->stoch_orbital(f_c);
        double E4 = arma::cdot(engine->R, (H.to_arma()/2)*engine->xi).real(); // note: g(x) = (x/2) f(x)
        cout << "energy (v4) " << E4 << " expected 40.9998\n";
        
        cout << "derivative <";
        for (int i = 0; i < 4; i++)
            cout << engine->stoch_element(i, i);
        cout << "> expected <10, -10, 0, 0>\n";
    }
    
    void testKPM2() {
        int n = 100;
        double noise = 0.2;
        RNG rng;
        rng.seed(0);
        std::normal_distribution<double> normal;
        
        // Build noisy tri-diagonal matrix
        SpMatCoo<arma::cx_double> H(n, n);
        for (int i = 0; i < n; i++) {
            auto x = 1.0 + noise * arma::cx_double(normal(rng), normal(rng));
            int j = (i-1+n)%n;
            H.add(i, j, x);
            H.add(j, i, conj(x));
        }
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
        auto engine1 = mk_engine_cx(n, n);
        auto engine2 = mk_engine_cx(n, n/4);
        engine1->set_H(H, es);
        engine2->set_H(H, es);
        
        double E1 = exactEnergy(H.to_arma(), kB_T, mu);
        double eps = 1e-6;
        int i=0, j=1;
        arma::sp_cx_mat dH(n, n);
        dH(i, j) = eps;
        dH(j, i) = eps;
        double dE_dH_1 = (exactEnergy<arma::cx_double>(H.to_arma()+dH, kB_T, mu)-exactEnergy<arma::cx_double>(H.to_arma()-dH, kB_T, mu)) / (2*eps);
        
        engine1->set_R_identity();
        double E2 = moment_product(g_c, engine1->moments(M));
        engine1->stoch_orbital(f_c);
        double dE_dH_2 = (engine1->stoch_element(i, j) + engine1->stoch_element(j, i)).real();
        
        engine2->set_R_uncorrelated(rng);
        double E3 = moment_product(g_c, engine2->moments(M));
        engine2->stoch_orbital(f_c);
        auto dE_dH_3 = (engine2->stoch_element(i, j) + engine2->stoch_element(j, i)).real();
        
        Vec<int> grouping(n);
        for (int i = 0; i < n; i++)
            grouping[i] = i%engine2->s;
        engine2->set_R_correlated(grouping, rng);
        double E4 = moment_product(g_c, engine2->moments(M));
        engine2->stoch_orbital(f_c);
        auto dE_dH_4 = (engine2->stoch_element(i, j) + engine2->stoch_element(j, i)).real();
        
        cout << "Exact energy            " << E1 << endl;
        cout << "Det. KPM energy         " << E2 << endl;
        cout << "Stoch. energy (uncorr.) " << E3 << endl;
        cout << "Stoch. energy (corr.)   " << E4 << endl << endl;
        
        cout << "Exact deriv.            " << dE_dH_1 << endl;
        cout << "Det. KPM deriv.         " << dE_dH_2 << endl;
        cout << "Stoch. deriv. (uncorr.) " << dE_dH_3 << endl;
        cout << "Stoch. deriv. (corr.)   " << dE_dH_4 << endl;
    }
}

int main(int argc,char **argv) {
    tibidy::testExpansionCoeffs();
    tibidy::testKPM1();
    tibidy::testKPM2();
}

