#include "iostream_util.h"
#include "kondo.h"

using namespace fkpm;
using namespace std::placeholders;

void testKondo1() {
    int w = 6, h = 6;
    double t1 = -1, t2 = 0, t3 = -0.5;
    double J = 0.5;
    double kB_T = 0;
    double mu = 0.103;

    auto m = Model(Lattice::mk_square(w, h, t1, t2, t3), J, {0,0,0});
    m.lattice->set_spins("ferro", m.spin);
    m.spin[0] = vec3(1, 1, 1).normalized();
    
    m.set_hamiltonian(m.spin);
    int n = m.H.n_rows;
    
    arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
    double E1 = electronic_grand_energy(eigs, kB_T, mu) / m.n_sites;
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m.H, extra, tolerance);
    int M = 2000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fermi_energy, _1, kB_T, mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fermi_density, _1, kB_T, mu), es);
    auto engine = mk_engine_cx();
    engine->set_H(m.H, es);
    engine->set_R_identity(n);
    
    double E2 = moment_product(g_c, engine->moments(M)) / m.n_sites;
    
    auto Ha = m.H.to_arma();
    cout << "H: " << Ha(0, 0) << " " << Ha(1, 0) << "\n  [(-0.288675,0)  (-0.288675,-0.288675)]\n";
    cout << "   " << Ha(0, 1) << " " << Ha(1, 1) << "\n  [(-0.288675,0.288675) (0.288675,0)]\n\n";

    engine->stoch_orbital(f_c);
    auto D = std::bind(&Engine<cx_double>::stoch_element, engine, _1, _2);
    cout << "D: " << D(0, 0) << " " << D(1, 0) << "\n  [(0.481926,0) (0.0507966,0.0507966)\n";
    cout << "   " << D(0, 1) << " " << D(1, 1) << "\n  [(0.0507966,-0.0507966) (0.450972,0)]\n\n";
    
    Vec<vec3>& force = m.dyn_stor[0];
    m.set_forces(D, force);
    
    cout << std::setprecision(9);
    cout << "grand energy " <<  E1 << " " << E2 << "\n            [-1.98657216 -1.98657194]\n";
    cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=0.015476587>]\n\n";
}

void testKondo2() {
    int w = 6, h = 6;
    double t1 = -1;
    double J = 0.1;
    double kB_T = 0;
    double mu = -1.0;
    
    auto m = Model(Lattice::mk_kagome(w, h, t1), J, {0,0,0});
    m.lattice->set_spins("ncp2", m.spin);
    m.set_hamiltonian(m.spin);
    arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
    double e = electronic_grand_energy(eigs, kB_T, mu);
    
    cout << "ncp2 " << e/m.n_sites << "     [-1.04384301]\n";
}

void testKondo3() {
    RNG rng(0);
    int w = 32;
    int h = w;
    double t1 = -1; // t2 = 0, t3 = 0;
    double J = 0.5;
    vec3 B_zeeman(0, 0, 0);
    double kB_T = 0;
    double mu = 0;
    int n_colors = 3;
    int n_orbs = 2;
    int s = n_colors*n_orbs;
    
//    auto m = Model(Lattice::mk_square(w, h, t1, t2, t3), J, B_zeeman);
    auto m = Model(Lattice::mk_kagome(w, h, t1), J, B_zeeman);
    
//    m.lattice->set_spins_random(rng, m.spin);
    m.lattice->set_spins("ferro", m.spin);
    m.set_hamiltonian(m.spin);
    
    Vec<int> c;
    m.lattice->set_colors(n_colors, c);
    Vec<int> groups;
    for (int i = 0; i < m.n_sites; i++) {
        for (int o = 0; o < n_orbs; o++) {
            groups.push_back(c[i]*n_orbs + o);
        }
    }
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m.H, extra, tolerance);
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fermi_energy, _1, kB_T, mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fermi_density, _1, kB_T, mu), es);
    auto engine = mk_engine_cx();
    engine->set_H(m.H, es);
    
    Vec<vec3>& f1 = m.dyn_stor[0];
    Vec<vec3>& f2 = m.dyn_stor[1];
    
    auto D = std::bind(&Engine<cx_double>::stoch_element, engine, _1, _2);
    
//    engine->set_R_identity(m.H.n_rows);
//    engine->set_R_uncorrelated(m.H.n_rows, s, rng);
    engine->set_R_correlated(groups, s, rng);
    engine->stoch_orbital(f_c);
    m.set_forces(D, f1);
    
//    engine->set_R_identity(m.H.n_rows);
//    engine->set_R_uncorrelated(m.H.n_rows, s, rng);
    engine->set_R_correlated(groups, s, rng);
    engine->stoch_orbital(f_c);
    m.set_forces(D, f2);
    
    double acc = 0;
    for (int i = 0; i < m.n_sites; i++) {
        acc += (f1[i] - f2[i]).norm2()/2;
    }
    double f_var = acc / m.n_sites;
    
    cout << std::setprecision(9);
    cout << "f_var " << f_var << endl;
}

int main(int argc,char **argv) {
//    testKondo1();
//    testKondo2();
    testKondo3();
}

