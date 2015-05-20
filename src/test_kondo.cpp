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
    m->set_spins("ferro", mk_toml(""), m->spin);
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
    cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=-0.384523162>]\n\n";
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
    
    m->set_spins("ncp1", mk_toml(""), m->spin);
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
    m->set_spins("ferro", mk_toml(""), m->spin);
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

void testKondo4() {
    RNG rng(1);
    int lx = 4;
    auto m = MostovoyModel(lx, lx, lx);
    m.t_pds = 1.7;
    m.t_pp = 0.65;
    m.delta = -2.0;
    m.set_spins_helical(2, m.spin);
    double filling = 1.0 / m.n_orbs;
    
    cout << "Exact energy: -5.79133  (J=inf)\n";
    m.J = 10000;
    m.set_hamiltonian(m.spin);
    arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
    double E_diag = electronic_energy(eigs, m.kT(), filling) / m.n_sites;
    cout << "E_diag      : " << E_diag << " [-5.79161] (J=10000)\n";
    
    m.J = 2;
    m.set_hamiltonian(m.spin);
    eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
    E_diag = electronic_energy(eigs, m.kT(), filling) / m.n_sites;
    cout << "E_diag      : " << E_diag << " [-6.15479] (J=2)\n";
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m.H, extra, tolerance);
    int M = 200;
    int Mq = 4*M;
    
    auto engine = mk_engine<cx_flt>();
    engine->set_H(m.H, es);
    
    int n_colors = 4*4*4;
    Vec<int> groups = m.groups(n_colors);
    engine->set_R_correlated(groups, rng);
    
    cout << "calculating kpm moments... " << std::flush;
    auto moments = engine->moments(M);
    cout << "done.\n";
    auto gamma = moment_transform(moments, Mq);
    
    double mu = filling_to_mu(gamma, es, m.kT(), filling, 0);
    double E_kpm = electronic_energy(gamma, es, m.kT(), filling, mu) / m.n_sites;
    cout << "E_kpm       : " << E_kpm << " [-6.1596] (J=2, M=200)\n";
}

void testKondo5() {
    RNG rng(1);
    int lx = 6;
    auto m = MostovoyModel(lx, lx, lx);
    m.t_pds = 0.5;
    m.t_pp = 0.2; // 0.5;
    double filling = 1.0 / m.n_orbs;
    
    cout << std::setw(10) << "delta" << std::setw(10) << "J" << std::setw(10) << "q" << std::setw(10) << "e\n";
    for (double delta : Vec<double>{-1 /* -2, -5*/}) {
        for (double J : Vec<double>{2 /* 5 , 20, 100*/}) {
            m.delta = delta;
            m.J = J;
            for (int q_idx = 0; q_idx <= lx/2; q_idx++) {
                m.set_spins_helical(q_idx, m.spin);
                m.set_hamiltonian(m.spin);
                arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
                double e = electronic_energy(eigs, m.kT(), filling) / m.n_sites;
                double q = 2*Pi*q_idx/lx;
                cout << std::setw(10) << delta << std::setw(10) << J << std::setw(10) << q << std::setw(10) << e << "\n";
            }
        }
    }
}

int main(int argc,char **argv) {
    testKondo1();
    testKondo2();
    testKondo3();
//    testKondo4();
//    testKondo5();
}

