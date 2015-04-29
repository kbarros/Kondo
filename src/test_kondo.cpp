#include "iostream_util.h"
#include "kondo.h"

using namespace fkpm;
using namespace std::placeholders;

void testKondo1() {
    int w = 6, h = 6;
    double mu = 0.103;
    auto m = SimpleModel::mk_square(w, h);
    m->J = 0.5;
    m->t1 = -1;
    m->t3 = -0.5;
    m->s1 = 0.1;
    m->set_spins("ferro", nullptr, m->spin);
    //m->lattice->set_spins("meron", nullptr, m->spin);
    m->spin[0] = vec3(1, 1, 1).normalized();
    
    m->set_hamiltonian(m->spin);
    int n = m->H.n_rows;
    
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
    double E1 = electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m->H, extra, tolerance);
    int M = 2000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fermi_energy, _1, m->kT(), mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fermi_density, _1, m->kT(), mu), es);
    auto engine = mk_engine<cx_flt>();
    engine->set_H(m->H, es);
    engine->set_R_identity(n);
    
    double E2 = moment_product(g_c, engine->moments(M)) / m->n_sites;
    
    cout << "H: " << *m->H(0, 0) << " " << *m->H(1, 0) << "\n  [(-0.288675,0)  (-0.288675,-0.288675)]\n";
    cout << "   " << *m->H(0, 1) << " " << *m->H(1, 1) << "\n  [(-0.288675,0.288675) (0.288675,0)]\n\n";
    
    engine->autodiff_matrix(g_c, m->D);
    cout << "D: " << *m->D(0, 0) << " " << *m->D(1, 0) << "\n  [(0.481926,0) (0.0507966,0.0507966)]\n";
    cout << "   " << *m->D(0, 1) << " " << *m->D(1, 1) << "\n  [(0.0507966,-0.0507966) (0.450972,0)]\n\n";
    
    Vec<vec3>& force = m->dyn_stor[0];
    m->set_forces(m->D, m->spin, force);
    
    cout << std::setprecision(9);
    cout << "grand energy " <<  E1 << " " << E2 << "\n            [-1.98657216 -1.98657194]\n";
    cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=0.015476587>]\n\n";
}

void testKondo2() {
    int w = 8, h = 8;
    double mu = -1.98397;
    auto m = SimpleModel::mk_kagome(w, h);
    m->J = 0.1;
    m->t1 = -1;
    EnergyScale es{-10, 10};
    int M = 1000;
    int Mq = 4*M;
    
    m->set_spins("ncp1", nullptr, m->spin);
    m->set_hamiltonian(m->spin);
    auto engine = mk_engine<cx_flt>();
    engine->set_H(m->H, es);
    
    cout << "calculating exact eigenvalues... " << std::flush;
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
    cout << "done.\n";
    
    cout << "calculating kpm moments... " << std::flush;
    RNG rng(0);
    int n_colors = 3*(w/2)*(h/2);
    Vec<int> groups = m->groups(n_colors);
    engine->set_R_correlated(groups, rng);
    auto moments = engine->moments(M);
    auto gamma = moment_transform(moments, Mq);
    cout << "done.\n";
    
    double e1 = electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;
    double e2 = electronic_grand_energy(gamma, es, m->kT(), mu) / m->n_sites;
    cout << "ncp1, grand energy, mu = " << mu << "\n";
    cout << "exact=" << e1 << "\n";
    cout << "kpm  =" << e2 << "\n";
    
    double filling = 0.25;
    double e3 = electronic_energy(eigs, m->kT(), filling) / m->n_sites;
    mu = filling_to_mu(gamma, es, m->kT(), filling, 0);
    double e4 = electronic_energy(gamma, es, m->kT(), filling, mu) / m->n_sites;
    cout << "ncp1, canonical energy, n=1/4\n";
    cout << "exact=" << e3 << "\n";
    cout << "kpm=" << e4 << "\n";
}

void testKondo3() {
    RNG rng(1);
    __attribute__((unused))
    int w = 2048, h = w;
    __attribute__((unused))
    //    auto m = SimpleModel::mk_square(w, h);
    //    auto m = SimpleModel::mk_kagome(w, h);
    auto m = SimpleModel::mk_linear(w);
    m->J = 0.5;
    m->t1 = -1;
    double mu = 0;
    
//    m->set_spins_random(rng, m->spin);
    m->set_spins("ferro", nullptr, m->spin);
    m->set_hamiltonian(m->spin);
    
    int n_colors = 4;
    Vec<int> groups = m->groups(n_colors);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m->H, extra, tolerance);
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fermi_energy, _1, m->kT(), mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fermi_density, _1, m->kT(), mu), es);
    auto engine = mk_engine<cx_flt>();
    engine->set_H(m->H, es);
    
    Vec<vec3>& f1 = m->dyn_stor[0];
    Vec<vec3>& f2 = m->dyn_stor[1];
    
//    auto calc1 = [&](Vec<vec3>& f) {
//        engine->set_R_identity(m->H.n_rows);
//        engine->stoch_orbital(f_c);
//        auto D = std::bind(&Engine<cx_flt>::stoch_element, engine, _1, _2);
//        m->set_forces(D, m->spin, f);
//    };
    __attribute__((unused))
    auto calc2 = [&](Vec<vec3>& f) {
        engine->set_R_uncorrelated(m->H.n_rows, n_colors*2, rng);
        engine->moments(M);
        engine->stoch_matrix(f_c, m->D);
        m->set_forces(m->D, m->spin, f);
    };
    __attribute__((unused))
    auto calc3 = [&](Vec<vec3>& f) {
        engine->set_R_correlated(groups, rng);
        engine->moments(M);
        engine->stoch_matrix(f_c, m->D);
        m->set_forces(m->D, m->spin, f);
    };
    __attribute__((unused))
    auto calc4 = [&](Vec<vec3>& f) {
        engine->set_R_correlated(groups, rng);
        engine->moments(M);
        engine->autodiff_matrix(g_c, m->D);
        m->set_forces(m->D, m->spin, f);
    };
    
    calc4(f1);
    calc4(f2);
    
    double acc = 0;
    for (int i = 0; i < m->n_sites; i++) {
        acc += (f1[i] - f2[i]).norm2()/2;
    }
    double f_var = acc / m->n_sites;
    
    cout << std::setprecision(9);
    cout << "f_var " << f_var << endl;
}

int main(int argc,char **argv) {
    testKondo1();
    testKondo2();
    testKondo3();
}

