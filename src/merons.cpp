#include "kondo.h"
#include "iostream_util.h"


int main(int argc,char **argv) {
    RNG rng(0);
    int w = 16;
    double t1 = -1, t2 = 0, t3 = 0;
    double kT = 0;
    int n_colors = 64;
    int M = 1000;
    int Mq = 4*M;
    EnergyScale es{-10, 10};
    
    double min_J = 0;
    double max_J = 5 + 1e-4;
    double d_J = 1; // 0.1;
    
    double min_mu = -5;
    double max_mu = +5 + 1e-4;
    double d_mu = 2; // 0.5;
    
    auto m = Model(SquareLattice::mk(w, w, t1, t2, t3), min_J, kT);
    
    auto engine = mk_engine_cx();
    engine->set_R_correlated(m.lattice->groups(n_colors), rng);
    
    cout << std::setprecision(9);
    cout << "# J mu Phi\n";
    
    for (m.J = min_J; m.J < max_J; m.J += d_J) {
        dynamic_cast<SquareLattice *>(m.lattice.get())->set_spins_meron(0, 2, m.spin);
        m.set_hamiltonian(m.spin);
        engine->set_H(m.H, es);
        
        auto moments = engine->moments(M);
        auto gamma = moment_transform(moments, Mq);
        
        for (double mu = min_mu; mu < max_mu; mu += d_mu) {
            auto g = std::bind(fermi_energy, std::placeholders::_1, kT, mu);
            double Phi = density_product(gamma, g, es) / m.lattice->n_sites();
            cout << m.J << " " << mu << " " << Phi;
            
            bool print_exact = false;
            if (print_exact) {
                arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
                double Phi_exact = electronic_grand_energy(eigs, kT, mu) / m.n_sites;
                cout << "   [" << Phi_exact << "]\n";
            }
            else {
                cout << endl;
            }
        }
    }
}
