#include <cassert>
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
    m.set_spins_helical(0, 0, 2, m.spin);
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
    auto engine = mk_engine_mpi<cx_flt>();
    
    RNG rng(4);
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
            EnergyScale es = {-std::abs(delta)-5, 8};
            
            for (int M: Vec<int>{500, 1000, 2000}) {
                cout << "\nM=" << M << endl;
                int Mq = 4*M;
                
                for (int q_idx = 0; q_idx <= lx/2; q_idx++) {
                    m.set_spins_helical(q_idx, q_idx, q_idx, m.spin);
                    timer[0].reset();
                    m.set_hamiltonian(m.spin);
                    engine->set_H(m.H, es);
                    
                    timer[0].reset();
                    auto moments = engine->moments(M);
                    cout << "time = " << timer[0].measure() << "\n";
                    auto gamma = moment_transform(moments, Mq);
                    double mu = filling_to_mu(gamma, es, m.kT(), filling, 0);
                    double e_kpm = electronic_energy(gamma, es, m.kT(), filling, mu) / m.n_sites;
                    
                    double q = 2*Pi*q_idx/lx;
                    cout << std::setw(10) << delta << std::setw(10) << J << std::setw(10) << q
                         << std::setw(10) << e_kpm << endl;
                }
            }
        }
    }
}



void testKondo1_cubic() {//cubic
    
    auto engine = fkpm::mk_engine<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);

    
    int w = 6, h = 6, h_z = 6;
    int int_mu, total_mu=100;
    double mu =0.0, del_mu=12.4/total_mu, mu_start=-6.2;
    
    std::stringstream fname;

    auto m = SimpleModel::mk_cubic(w, h, h_z);
    m->J = 0.2;
    m->t1 = -1;
    m->t3 = 0.5;
    m->s1 = 0.0;
    m->set_spins("ferro", mk_toml(""), m->spin);
    //m->lattice->set_spins("meron", nullptr, m->spin);
    m->spin[0] = vec3(1, 1, 1).normalized();

    fname << "mu_n_cubic_test_L_" << w << "_t1_" <<  m->t1 << "_t2_" <<  m->t2 << "_t3_" <<  m->t3 << "_J_" <<  m->J << "_01.txt";
    
    m->set_hamiltonian(m->spin);
    int n = m->H.n_rows;

    std::ofstream dump_file(fname.str(), std::ios::trunc);

    
    for (int_mu=0; int_mu<total_mu; int_mu++) {
        mu = del_mu*int_mu + mu_start;
        
        Vec<double> gamma;
        Vec<double> moments;
        
        arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
        //double E1 = electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;
        
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
        
        
        
        //double E2 = moment_product(g_c, engine->moments(M)) / m->n_sites;
        
        //cout << "H: " << *m->H(0, 0) << " " << *m->H(1, 0) << "\n  [(-0.288675,0)  (-0.288675,-0.288675)]\n";
        //cout << "   " << *m->H(0, 1) << " " << *m->H(1, 1) << "\n  [(-0.288675,0.288675) (0.288675,0)]\n\n";
        
        engine->autodiff_matrix(g_c, m->D);
        //cout << "D: " << *m->D(0, 0) << " " << *m->D(1, 0) << "\n  [(0.481926,0) (0.0507966,0.0507966)]\n";
        //cout << "   " << *m->D(0, 1) << " " << *m->D(1, 1) << "\n  [(0.0507966,-0.0507966) (0.450972,0)]\n\n";
        
        Vec<vec3>& force = m->dyn_stor[0];
        m->set_forces(m->D, m->spin, force);
        
        //cout << std::setprecision(9);
        //cout << "grand energy " <<  E1 << " " << E2 << "\n            [-1.98657216 -1.98657194]\n";
        //cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=-0.384523162>]\n\n";

        moments = engine->moments(M);
        gamma = fkpm::moment_transform(moments, Mq);

        //fprintf(fp1, "%10f, %10lf\n", mu, g.get_unwrap<double>("ensemble.filling"));
        //printf("%10lf, %10f\n", mu, g.get_unwrap<double>("ensemble.filling"));
        double filling =  mu_to_filling(gamma, es, m->kT(), mu);
        dump_file << mu << ", " << filling << endl;
        cout  << mu << ", " << filling << endl;
    }
}


inline int pos_square(int x, int y, int lx) { int temp = x + lx * y; assert(temp>=0 && temp<lx*lx); return temp; }
arma::cx_mat transformU(int lx) {
    int n = lx * lx;
    arma::cx_mat ret(n,n);
    ret.zeros();
    for (int kx = 0; kx < lx; kx++) {
        for (int ky = 0; ky < lx; ky++) {
            int pos_k = pos_square(kx, ky, lx);
            for (int ix = 0; ix < lx; ix++) {
                for (int iy = 0; iy < lx; iy++) {
                    int pos_i = pos_square(ix, iy, lx);
                    ret(pos_k,pos_i) = std::exp(cx_double(0.0, 2.0 * (kx*ix + ky*iy) * Pi / n)) / std::sqrt(n);
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "i,j=" << i << "," << j << ", ret=" << ret(i,j) << std::endl;
        }
    }
    arma::cx_mat test(n,n);
    test = arma::trans(ret);
    test = ret * test;
//    for (int i = 0; i < n; i++) {
//        if (std::abs(test(i,i)-1.0)>1e-9) {
//            std::cout << "test(" << i << "," << i << ")=" << test(i,i) << std::endl;
//        }
//        for (int j = 0; j < n; j++) {
//            if (i != j && std::abs(test(i,j))>1e-9) {
//                std::cout << "test(" << i << "," << j << ")=" << test(i,j) << std::endl;
//            }
//        }
//    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "i,j=" << i << "," << j << ", test=" << test(i,j) << std::endl;
        }
    }
    return ret;
}

// triangular lattice
void testKondo6() {
    int w = 100, h = 100;
    auto m = SimpleModel::mk_triangular(w, h);
    m->J = 5.0 * sqrt(3.0);
    m->t1 = -1;
    int M = 200;
    int Mq = M;
    int n_colors = 12;
    auto kernel = fkpm::jackson_kernel(M);

    auto engine = mk_engine<cx_flt>();
    
    m->set_spins("allout", mk_toml(""), m->spin);
    m->set_hamiltonian(m->spin);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m->H, extra, tolerance);
    
    engine->set_H(m->H, es);

    RNG rng(0);
    //engine->set_R_correlated(m->groups(n_colors), rng);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    engine->R2 = engine->R;
    
//    auto u_fourier = transformU(4);
    
    double area = w*h*sqrt(3.0)/2.0;
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    jx.scale(1/sqrt(area));
    jy.scale(1/sqrt(area));
    
    cout << "calculating moments2... " << std::flush;
    timer[0].reset();
    auto mu_xy = engine->moments2_v1(M, jx, jy);
    cout << " done. " << timer[0].measure() << "s.\n";

    cout << "T=" << m->kT();
    
    cout << "calculating xy conductivities... " << std::flush;
    std::ofstream fout2("test.dat", std::ios::out /* | std::ios::app */);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20) << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    arma::Col<double> sigma_xy(Mq);
    sigma_xy.zeros();
    for (int i = 0; i < Mq; i++) {
        double mu = es.lo + i * (es.hi-es.lo) / Mq;
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/m->kT() << std::setw(20) << mu << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    cout << " done. " << timer[0].measure() << "s.\n";
    
//    auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -10.0, 0.0, es, kernel);
//    std::ofstream fout6("cmn_mu_m10_RE.dat", std::ios::out /* | std::ios::app */);
//    std::ofstream fout7("cmn_mu_m10_IM.dat", std::ios::out /* | std::ios::app */);
//    for (int m1 = 0; m1 < M; m1++) {
//        for (int m2 = 0; m2 < M; m2++) {
//            fout6 << std::setw(20) << std::real(cmn[m1][m2]);
//            fout7 << std::setw(20) << std::imag(cmn[m1][m2]);
//        }
//        fout6 << endl;
//        fout7 << endl;
//    }
//    fout6.close();
//    fout7.close();
//    
//    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -9.0, 0.0, es, kernel);
//    std::ofstream fout8("cmn_mu_m9_RE.dat", std::ios::out /* | std::ios::app */);
//    std::ofstream fout9("cmn_mu_m9_IM.dat", std::ios::out /* | std::ios::app */);
//    for (int m1 = 0; m1 < M; m1++) {
//        for (int m2 = 0; m2 < M; m2++) {
//            fout8 << std::setw(20) << std::real(cmn[m1][m2]);
//            fout9 << std::setw(20) << std::imag(cmn[m1][m2]);
//        }
//        fout8 << endl;
//        fout9 << endl;
//    }
//    fout8.close();
//    fout9.close();
//    
//    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -7.5, 0.0, es, kernel);
//    std::ofstream fout10("cmn_mu_m7d5_RE.dat", std::ios::out /* | std::ios::app */);
//    std::ofstream fout11("cmn_mu_m7d5_IM.dat", std::ios::out /* | std::ios::app */);
//    for (int m1 = 0; m1 < M; m1++) {
//        for (int m2 = 0; m2 < M; m2++) {
//            fout10 << std::setw(20) << std::real(cmn[m1][m2]);
//            fout11 << std::setw(20) << std::imag(cmn[m1][m2]);
//        }
//        fout10 << endl;
//        fout11 << endl;
//    }
//    fout10.close();
//    fout11.close();
//    
//    auto dmn = electrical_conductivity_coefficients(M, Mq, m->kT(), std::sqrt(2.0), 0.0, es, kernel);
//    std::ofstream fout16("cmn_mu_sqrt2_RE.dat", std::ios::out /* | std::ios::app */);
//    std::ofstream fout17("cmn_mu_sqrt2_IM.dat", std::ios::out /* | std::ios::app */);
//    for (int m1 = 0; m1 < M; m1++) {
//        for (int m2 = 0; m2 < M; m2++) {
//            fout16 << std::setw(20) << std::real(dmn[m1][m2]);
//            fout17 << std::setw(20) << std::imag(dmn[m1][m2]);
//        }
//        fout16 << endl;
//        fout17 << endl;
//    }
//    fout10.close();
//    fout11.close();
//    
//    auto f = std::bind(fkpm::fermi_density, _1, m->kT(), -9.0);
//    auto f_c = expansion_coefficients(M, Mq, f, es);
    
//    std::ofstream fout13("c_mu_m9_RE.dat", std::ios::out /* | std::ios::app */);
//    for (int m = 0; m < M; m++) {
//        fout13 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
//    }
//    fout13.close();
//    
//    f = std::bind(fkpm::fermi_density, _1, m->kT(), -7.5);
//    f_c = expansion_coefficients(M, Mq, f, es);
//    std::ofstream fout14("c_mu_m7d5_RE.dat", std::ios::out /* | std::ios::app */);
//    for (int m = 0; m < M; m++) {
//        fout14 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
//    }
//    fout14.close();
//    
//    f = std::bind(fkpm::fermi_density, _1, m->kT(), -10.0);
//    f_c = expansion_coefficients(M, Mq, f, es);
//    std::ofstream fout15("c_mu_m10_RE.dat", std::ios::out /* | std::ios::app */);
//    for (int m = 0; m < M; m++) {
//        fout15 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
//    }
//    fout15.close();
    
}

void testKondo7() {
    int w = 32, h = 32;
    auto m = SimpleModel::mk_kagome(w, h);
    m->J = 15.0 * sqrt(3.0);
    m->t1 = -3.5;
    int M = 300;
    int Mq = M;
    int Lc = 4;
    int n_colors = 3 * Lc * Lc;
    auto kernel = fkpm::jackson_kernel(M);
    
    auto engine = mk_engine<cx_flt>();
    
    //m->set_spins("allout", mk_toml(""), m->spin);
    
    double dz = 0.55;
    for(int i=0; i<m->n_sites; i++) {
        int v = i % 3;
        vec3 s(0, 0, 0);
        if(v == 0) s = vec3(-0.5 * sqrt(3.), -0.5, dz);
        else if(v == 1) s = vec3(0, 1, dz);
        else s = vec3(+0.5 * sqrt(3.), -0.5, dz);
        
        m->spin[i] = s.normalized();
    }
    
    m->set_hamiltonian(m->spin);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m->H, extra, tolerance);
    
    engine->set_H(m->H, es);
    
    RNG rng(0);
    //engine->set_R_correlated(m->groups(n_colors), rng);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    engine->R2 = engine->R;
    
    auto moments = engine->moments(M);
    std::cout << "time to calculate moments : " << fkpm::timer[0].measure() << "\n";
    auto gamma = fkpm::moment_transform(moments, Mq);

    std::ofstream fs("dos.dat");
    Vec<double> x, rho, irho;
    fkpm::density_function(gamma, es, x, rho);
    fkpm::integrated_density_function(gamma, es, x, irho);
    for (int i = 0; i < x.size(); i++) {
        fs << x[i] << " " << rho[i] / m->H.n_rows << " " << irho[i] / m->H.n_rows << "\n";
    }
    fs.close();
    
    //    auto u_fourier = transformU(4);
    
    double area = 4. * w*h*sqrt(3.0)/2.0;
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    jx.scale(1/sqrt(area));
    jy.scale(1/sqrt(area));
    
    cout << "calculating moments2... " << std::flush;
    timer[0].reset();
    auto mu_xy = engine->moments2_v1(M, jx, jy, 3);
    auto mu_xx = engine->moments2_v1(M, jx, jx, 3);
    cout << " done. " << timer[0].measure() << "s.\n";
    
    cout << "calculating xy conductivities... " << std::flush;
    std::ofstream fout2("test.dat", std::ios::out /* | std::ios::app */);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20) << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    double n_mus = 2*Mq;
    arma::Col<double> sigma_xy(n_mus);
    arma::Col<double> sigma_xx(n_mus);
    sigma_xy.zeros();
    sigma_xx.zeros();
    for (int i = 0; i < n_mus; i++) {
        double mu = es.lo + i * (es.hi-es.lo) / n_mus;
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        sigma_xx(i) = std::real(fkpm::moment_product(cmn, mu_xx));
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/m->kT() << std::setw(20) << mu << std::setw(20) << sigma_xy(i) << '\t' << sigma_xx(i) << std::endl;
    }
    fout2.close();
    cout << " done. " << timer[0].measure() << "s.\n";
}

int main(int argc,char **argv) {
//    testKondo1();
//    testKondo2();
//    testKondo3();
//    testKondo4();
//    testKondo5();
//    testKondo1_cubic();
//    testKondo6();
//    testKondo7();
}

