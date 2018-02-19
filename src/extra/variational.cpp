#include <cassert>
#include "iostream_util.h"
#include "kondo.h"


void mostovoy1() {
    auto engine = fkpm::mk_engine_mpi<cx_flt>();
    
    fkpm::RNG rng(4);
    int lx = 16;
    auto m = MostovoyModel(lx, lx, lx);
    m.t_pds = 0.89;
    m.t_pp = 0.44;
    double filling = 1.0 / m.n_orbs;
    
    // KPM Parameters
    int lc = 4;
    assert(lx % lc == 0);
    int n_colors = lc*lc*lc;
    Vec<int> groups = m.groups(n_colors);
    engine->set_R_correlated(groups, rng);
    
    cout << std::setw(10) << "delta" << std::setw(10) << "J" << std::setw(10) << "q" << std::setw(10) << "e_kpm\n";
    for (double delta : Vec<double>{-3.1 /*, -3, -4 */}) {
        for (double J : Vec<double>{1 /* 5 , 20, 100*/}) {
            m.delta = delta;
            m.J = J;
            fkpm::EnergyScale es = {-std::abs(delta)-5, 8};
            
            for (int M: Vec<int>{500, 1000, 2000}) {
                cout << "\nM=" << M << endl;
                int Mq = 4*M;
                
                for (int q_idx = 0; q_idx <= lx/2; q_idx++) {
                    m.set_spins_helical(q_idx, q_idx, q_idx, m.spin);
                    fkpm::timer[0].reset();
                    m.set_hamiltonian(m.spin);
                    engine->set_H(m.H, es);
                    
                    fkpm::timer[0].reset();
                    auto moments = engine->moments(M);
                    cout << "time = " << fkpm::timer[0].measure() << "\n";
                    auto gamma = fkpm::moment_transform(moments, Mq);
                    double mu = fkpm::filling_to_mu(gamma, es, m.kT(), filling, 0);
                    double e_kpm = fkpm::electronic_energy(gamma, es, m.kT(), filling, mu) / m.n_sites;
                    
                    double q = 2*Pi*q_idx/lx;
                    cout << std::setw(10) << delta << std::setw(10) << J << std::setw(10) << q
                    << std::setw(10) << e_kpm << endl;
                }
            }
        }
    }
}


void sdw() {
    auto engine = fkpm::mk_engine_mpi<cx_flt>();
    
    fkpm::RNG rng(4);
    int lx = 96;
    auto m = SimpleModel::mk_triangular(lx, lx);
    double U = 5.6;
    m->J  = U/3.0;
    m->t1 = -1;
    m->s0 = U/6.0;
    double kT = 0.0;
    double filling = 0.75;
    
    fkpm::EnergyScale es {-10, 10};
    int M = 2000;
    int Mq = 4*M;
    
    int n_colors = 12*12;
    Vec<int> groups = m->groups(n_colors);
    engine->set_R_correlated(groups, rng);
    
    auto calc_energy = [&]() -> double {
        m->set_hamiltonian(m->spin);
        engine->set_H(m->H, es);
        
        auto moments = engine->moments(M);
        auto gamma = fkpm::moment_transform(moments, Mq);
        double mu = fkpm::filling_to_mu(gamma, es, kT, filling, 0);
        return (fkpm::electronic_energy(gamma, es, kT, filling, mu)
                + m->energy_classical(m->spin)) / m->n_sites;
    };
    
    cout << "# Delta Allout Collinear Ferro\n";
    
    double Delta0 = 0.09;
    double d_Delta = 0.005;
    for (int i = 0; i < 20; i++) {
        double Delta = Delta0 + i*d_Delta;
        
        auto toml = toml_from_str("Delta = " + std::to_string(Delta));
        
        m->set_spins("allout", toml, m->spin);
        double e1 = calc_energy();
        
        m->set_spins("3q_collinear", toml, m->spin);
        double e2 = calc_energy();
        
        m->spin.assign(m->n_sites, {0, 0, 3*Delta});
        double e3 = calc_energy();
        
        cout << Delta << " " << e1 << " " << e2 << " " << e3 << "\n";
    }
}


int main(int argc,char **argv) {
    // mostovoy1();
    sdw();
}


