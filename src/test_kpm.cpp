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
    
    // TODO: move to fastkpm
    double exactEnergy(arma::sp_cx_mat H, double kB_T, double mu) {
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
        arma::sp_cx_mat H(n, n);
        H(0, 0) = 5.0;
        H(1, 1) = -5.0;
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
        
        double E1 = engine->trace(g_c);
        cout << "energy (v1) " << E1 << " expected 50.0004 for M=1000\n";
        
        auto gamma = moment_transform(engine->moments(M), Mq);
        double E2 = density_product(gamma, g, es);
        cout << "energy (v2) " << E2 << endl;
        
        double E3 = engine->trace(g_c, arma::speye<arma::sp_cx_mat>(n, n));
        cout << "energy (v3) " << E3 << endl;
        
        double E4 = engine->trace(f_c, H/2); // note: g(x) = (x/2) f(x)
        cout << "energy (v4) " << E4 << " expected 40.9998\n";
        
        cout << "derivative " << engine->deriv(f_c) << endl;
        cout << "expected <10, -10, 0, 0>\n";
    }
    
    void testKPM2() {
        int n = 100;
        double noise = 0.2;
        RNG rng;
        rng.seed(0);
        std::normal_distribution<double> normal;
        
        // Build noisy tri-diagonal matrix
        MatrixBuilder<arma::cx_double> mb;
        for (int i = 0; i < n; i++) {
            auto x = 1.0 + noise * arma::cx_double(normal(rng), normal(rng));
            int j = (i-1+n)%n;
            mb.add(i, j, x);
            mb.add(j, i, conj(x));
        }
        auto H = mb.build(n, n);
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
        
        double E1 = exactEnergy(H, kB_T, mu);
        double eps = 1e-6;
        int i=0, j=1;
        arma::sp_cx_mat dH(n, n);
        dH(i, j) = eps;
        dH(j, i) = eps;
        auto dE_dH_1 = (exactEnergy(H+dH, kB_T, mu)-exactEnergy(H-dH, kB_T, mu)) / (2*eps);
        
        engine1->set_R_identity();
        double E2 = engine1->trace(g_c);
        auto dE_dH_2 = engine1->deriv(f_c);
        
        engine2->set_R_uncorrelated(rng);
        double E3 = engine2->trace(g_c);
        auto dE_dH_3 = engine2->deriv(f_c);
        
        Vec<int> grouping(n);
        for (int i = 0; i < n; i++)
            grouping[i] = i%engine2->s;
        engine2->set_R_correlated(grouping, rng);
        double E4 = engine2->trace(g_c);
        auto dE_dH_4 = engine2->deriv(f_c);
        
        cout << "Exact energy            " << E1 << endl;
        cout << "Det. KPM energy         " << E2 << endl;
        cout << "Stoch. energy (uncorr.) " << E3 << endl;
        cout << "Stoch. energy (corr.)   " << E4 << endl << endl;
        
        cout << "Exact deriv.            " << dE_dH_1 << endl;
        cout << "Det. KPM deriv.         " << (arma::cx_double)dE_dH_2(i, j) + (arma::cx_double)dE_dH_2(j, i) << endl;
        cout << "Stoch. deriv. (uncorr.) " << (arma::cx_double)dE_dH_3(i, j) + (arma::cx_double)dE_dH_3(j, i) << endl;
        cout << "Stoch. deriv. (corr.)   " << (arma::cx_double)dE_dH_4(i, j) + (arma::cx_double)dE_dH_4(j, i) << endl;
    }
}

int main(int argc,char **argv) {
    tibidy::testExpansionCoeffs();
    tibidy::testKPM1();
    tibidy::testKPM2();
}

