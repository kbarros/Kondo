#include <cassert>
#include "iostream_util.h"
#include "kondo.h"


void testKondo1() {
    auto engine = fkpm::mk_engine<cx_flt>();
    
    int w = 6, h = 6;
    double mu = 0.103;
    auto m = SimpleModel::mk_square(w, h);
    m->J = 0.5;
    m->t1 = -1;
    m->t3 = -0.5;
    m->s1 = 0.1;
    m->set_spins("ferro", toml_from_str(""), m->spin);
    m->spin[0] = vec3(1, 1, 1).normalized();
    
    m->set_hamiltonian(m->spin);
    int n = m->H.n_rows;
    
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
    double E1 = fkpm::electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;
    
    auto es = engine->energy_scale(m->H, 0.1);
    int M = 2000;
    int Mq = 4*M;
    using std::placeholders::_1;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, _1, m->kT(), mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_density, _1, m->kT(), mu), es);
    
    engine->set_H(m->H, es);
    engine->set_R_identity(n);
    
    double E2 = fkpm::moment_product(g_c, engine->moments(M)) / m->n_sites;
    
    cout << "H: " << *m->H(0, 0) << " " << *m->H(1, 0) << "\n  [(-0.288675,0)  (-0.288675,-0.288675)]\n";
    cout << "   " << *m->H(0, 1) << " " << *m->H(1, 1) << "\n  [(-0.288675,0.288675) (0.288675,0)]\n\n";
    
    engine->autodiff_matrix(g_c, m->D);
    cout << "D: " << *m->D(0, 0) << " " << *m->D(1, 0) << "\n  [(0.481926,0) (0.0507966,0.0507966)]\n";
    cout << "   " << *m->D(0, 1) << " " << *m->D(1, 1) << "\n  [(0.0507966,-0.0507966) (0.450972,0)]\n\n";
    
    Vec<vec3>& force = m->dyn_stor[0];
    m->set_forces(m->D, m->spin, force);
    
    cout << std::setprecision(9);
    cout << "grand energy " <<  E1 << " " << E2 << "\n            [-1.98657216 -1.98657194]\n";
    cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=-0.384523162>]\n\n";
}

void testKondo2() {
    auto engine = fkpm::mk_engine<cx_flt>();
    
    int w = 8, h = 8;
    double mu = -1.98397;
    auto m = SimpleModel::mk_kagome(w, h);
    m->J = 0.1;
    m->t1 = -1;
    fkpm::EnergyScale es{-10, 10};
    int M = 1000;
    int Mq = 4*M;
    
    m->set_spins("ncp1", toml_from_str(""), m->spin);
    m->set_hamiltonian(m->spin);
    engine->set_H(m->H, es);
    
    cout << "calculating exact eigenvalues... " << std::flush;
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
    cout << "done.\n";
    
    cout << "calculating kpm moments... " << std::flush;
    fkpm::RNG rng(0);
    int n_colors = 3*(w/2)*(h/2);
    Vec<int> groups = m->groups(n_colors);
    engine->set_R_correlated(groups, rng);
    auto moments = engine->moments(M);
    auto gamma = fkpm::moment_transform(moments, Mq);
    cout << "done.\n";
    
    double e1 = fkpm::electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;
    double e2 = fkpm::electronic_grand_energy(gamma, es, m->kT(), mu) / m->n_sites;
    cout << "ncp1, grand energy, mu = " << mu << "\n";
    cout << "exact=" << e1 << "\n";
    cout << "kpm  =" << e2 << "\n";
    
    double filling = 0.25;
    double e3 = fkpm::electronic_energy(eigs, m->kT(), filling) / m->n_sites;
    mu = fkpm::filling_to_mu(gamma, es, m->kT(), filling, 0);
    double e4 = fkpm::electronic_energy(gamma, es, m->kT(), filling, mu) / m->n_sites;
    cout << "ncp1, canonical energy, n=1/4\n";
    cout << "exact=" << e3 << "\n";
    cout << "kpm=" << e4 << "\n";
}

void testKondo3() {
    auto engine = fkpm::mk_engine<cx_flt>();
    
    fkpm::RNG rng(1);
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
    m->set_spins("ferro", toml_from_str(""), m->spin);
    m->set_hamiltonian(m->spin);
    
    int n_colors = 4;
    Vec<int> groups = m->groups(n_colors);
    
    auto es = engine->energy_scale(m->H, 0.1);
    int M = 1000;
    int Mq = 4*M;
    using std::placeholders::_1;
    auto g_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, _1, m->kT(), mu), es);
    auto f_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_density, _1, m->kT(), mu), es);
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
    cout << "f_var " << f_var << " [order 0.0016]\n";
}

void conductivity() {
    auto engine = fkpm::mk_engine<cx_flt>();
    int w = 4, h = 4;
    int M = 200;
    int Mq = 2*M;
    int n_colors = 4*4;
    auto kernel = fkpm::jackson_kernel(M);
    
    auto m = SimpleModel::mk_triangular(w, h);
    m->J = 1;
    m->t1 = -1;
    m->set_spins("allout", toml_from_str(""), m->spin);
    m->set_hamiltonian(m->spin);
    
    auto es = engine->energy_scale(m->H, 0.1);
    engine->set_H(m->H, es);
    
    fkpm::RNG rng(1);
    engine->set_R_correlated(m->groups(n_colors), rng);
    
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    
    auto moments = engine->moments(M);
    auto moments_xy = engine->moments2_v1(M, jx, jy);
    
    auto gamma = fkpm::moment_transform(moments, Mq);
    auto mu = fkpm::filling_to_mu(gamma, es, m->kT(), 0.25, 0.0);
    auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
    std::cout << "sigma_xy " << std::real(fkpm::moment_product(cmn, moments_xy)) << " [-1.22811]\n";
}

void mostovoy_energy() {
    auto engine = fkpm::mk_engine<cx_flt>();
    
    fkpm::RNG rng(1);
    int lx = 4;
    auto m = MostovoyModel(lx, lx, lx);
    m.t_pds = 1.7;
    m.t_pp = 0.65;
    m.delta = -2.0;
    m.set_spins_helical(0, 0, 2, m.spin);
    double filling = 1.0 / m.n_orbs;
    
    cout << "Exact energy: -5.79133  (J=inf)\n";
    m.J = 10000;
    m.set_hamiltonian(m.spin);
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
    double E_diag = fkpm::electronic_energy(eigs, m.kT(), filling) / m.n_sites;
    cout << "E_diag      : " << E_diag << " [-5.79161] (J=10000)\n";
    
    m.J = 2;
    m.set_hamiltonian(m.spin);
    eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
    E_diag = fkpm::electronic_energy(eigs, m.kT(), filling) / m.n_sites;
    cout << "E_diag      : " << E_diag << " [-6.15479] (J=2)\n";
    
    auto es = engine->energy_scale(m.H, 0.1);
    int M = 200;
    int Mq = 4*M;
    
    engine->set_H(m.H, es);
    
    int n_colors = 4*4*4;
    Vec<int> groups = m.groups(n_colors);
    engine->set_R_correlated(groups, rng);
    
    cout << "calculating kpm moments... " << std::flush;
    auto moments = engine->moments(M);
    cout << "done.\n";
    auto gamma = fkpm::moment_transform(moments, Mq);
    
    double mu = filling_to_mu(gamma, es, m.kT(), filling, 0);
    double E_kpm = electronic_energy(gamma, es, m.kT(), filling, mu) / m.n_sites;
    cout << "E_kpm       : " << E_kpm << " [-6.1596] (J=2, M=200)\n";
}

int main(int argc,char **argv) {
    testKondo1();
    testKondo2();
    testKondo3();
    conductivity();
//    mostovoy_energy();
}

