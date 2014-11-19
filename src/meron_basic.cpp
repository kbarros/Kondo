//
//  meron_basic.cpp
//  kondo
//
//  Created by Ryo Ozawa on 2014/11/20.
//
//

#include "kondo.h"
#include "iostream_util.h"


int main(int argc,char **argv) {
    RNG rng(0);
    int w = 16;
    double t1 = -1, t2 = 0, t3 = 0.5;
    double kT = 0;
    int n_colors = 64;
    int M = 1000;
    int Mq = 4*M;
    EnergyScale es{-8, 8};
    
    double min_J = 0.1;
    double max_J = 0.1 + 1e-4;
    double d_J = 0.1; // 0.1;
    
    double min_mu = -2.45;//-2.6;
    double max_mu = -2.45  +1e-4;//-2.3 + 1e-4;
    double d_mu = 0.05; // 0.5;
    
    double min_a = 0.0;
    double max_a = 0.1 + 1e-4;//1.0 + 1e-4;
    double d_a   = 0.05;
    double meron_a;
    int meron_Q;
    
    double filling;
    auto m = Model(SquareLattice::mk(w, w, t1, t2, t3), min_J, kT);
    
    auto engine = mk_engine_cx();
    engine->set_R_correlated(m.lattice->groups(n_colors), rng);
    
    cout << std::setprecision(9);
    cout << "# J mu Phi(, ED_Phi) \n";
    
    for (m.J = min_J; m.J < max_J; m.J += d_J) {
        for (meron_a = min_a; meron_a < max_a; meron_a += d_a) {
            for (meron_Q=0; meron_Q<w; meron_Q++) {
                dynamic_cast<SquareLattice *>(m.lattice.get())->set_spins_meron(meron_a, meron_Q, m.spin);
                m.set_hamiltonian(m.spin);
                engine->set_H(m.H, es);
                
                auto moments = engine->moments(M);
                auto gamma = moment_transform(moments, Mq);
                
                
                for (double mu = min_mu; mu < max_mu; mu += d_mu) {
                    double Phi = electronic_grand_energy(gamma, es, kT, mu) / m.n_sites;
                    filling = mu_to_filling(gamma, es, m.kB_T, mu);
                    
                    printf("%10lf, %10lf, %10lf, %10lf, ", m.J, mu, Phi);
                    
                    
                    bool print_exact = true;
                    //bool print_exact = false;
                    if (print_exact) {
                        arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
                        double Phi_exact = electronic_grand_energy(eigs, kT, mu) / m.n_sites;
                        //cout << "   [" << Phi_exact << "]\n";
                        cout << Phi_exact << endl;
                        
                    }
                    else {
                        cout << endl;
                    }
                }
                
            }
        }
        
    }
}
