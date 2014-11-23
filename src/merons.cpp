#include "kondo.h"
#include "iostream_util.h"
#include "time.h"

using namespace fkpm;

Vec<double> slice(Vec<double> src, int begin, int end) {
    Vec<double> ret(end-begin);
    for (int i = begin; i < end; i++) {
        ret[i-begin] = src[i];
    }
    return ret;
}

int main(int argc,char **argv) {
    clock_t time1, time2;
    time1 = clock();
    
    RNG rng(0);
    int w = 8;
    double t1 = -1, t2 = 0, t3 = 0.5;
    double kT =  0;
    int n_colors = 8*8;
    Vec<int> Ms {500, 1000, 2000, 4000, 8000};
    //Vec<int> Ms {500, 1000, 2000, 4000};
    EnergyScale es{-9, 9};

    
    double min_J = 0.1;
    double max_J = 0.1 + 1e-4;
    double d_J = 0.1; // 0.1;
    
    double min_mu = -2.50;//-2.6;
    double max_mu = -2.30  +1e-4;//-2.3 + 1e-4;
    double d_mu = 0.05; // 0.5;

    double min_a = 0.0;
    double max_a = 1.0 + 1e-4;//1.0 + 1e-4;
    double d_a   = 0.05;
    double meron_a;
    int meron_Q;
    
    FILE *fp1;
    char filename1[100];
    //sprintf(filename1, "merons_variational_test_%dx%dsites_%03dclr_%05d_2.txt", w, w, n_colors, M);
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_%05d_2.txt", w, w, n_colors, Ms.back());
    sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_2.txt", w, w, n_colors, Ms.back());
    fp1 = fopen(filename1, "w");
    printf("filename:%s\n", filename1);
    double filling;
    auto m = Model(SquareLattice::mk(w, w, t1, t2, t3), min_J, kT);
    
    auto engine = mk_engine<cx_flt>();
    auto groups = m.lattice->groups(n_colors);
    cout << std::setprecision(9);
    cout << "# J, meron_a, meron_Q, mu, Phi, filling, M(, ED_Phi)\n";

    
    for (m.J = min_J; m.J < max_J; m.J += d_J) {
        for (meron_a = min_a; meron_a < max_a; meron_a += d_a) {
            //for (meron_Q=0; meron_Q<w; meron_Q++) {
            meron_Q = w/8;
            engine->set_R_correlated(groups, rng);

            dynamic_cast<SquareLattice *>(m.lattice.get())->set_spins_meron(meron_a, meron_Q, m.spin);
            m.set_hamiltonian(m.spin);
            engine->set_H(m.H, es);
            
            auto allMoments = engine->moments(Ms.back());
            
            for (int M : Ms) {
                auto moments = slice(allMoments, 0, M);
                auto gamma = moment_transform(moments, 4*M);
            
            
                for (double mu = min_mu; mu < max_mu; mu += d_mu) {
                    double Phi = electronic_grand_energy(gamma, es, kT, mu) / m.n_sites;
                    filling = mu_to_filling(gamma, es, m.kB_T, mu);
                    
                    printf("%10lf, %10lf, %5d, %10lf, %10lf, %10lf, %d, ", m.J, meron_a, meron_Q, mu, Phi, filling, M);
                    fprintf(fp1, "%10lf, %10lf, %5d, %10lf, %10lf, %10lf, %d, ", m.J, meron_a, meron_Q, mu, Phi, filling, M);
                    fflush(fp1);
                    
                    bool print_exact = true;
                    //bool print_exact = false;
                    if (print_exact) {
                        arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m.H.to_arma_dense()));
                        double Phi_exact = electronic_grand_energy(eigs, kT, mu) / m.n_sites;
                        //cout << "   [" << Phi_exact << "]\n";
                        cout << Phi_exact << endl;

                        fprintf(fp1, "%10lf\n", Phi_exact);
                    }
                    else {
                        cout << endl;
                        fprintf(fp1, "\n");
                    }
                }
            }
            fprintf(fp1, "\n");
            time2 = clock();
            printf("#time=%10lf sec, %10lf min, %10lf H\n",(double)(time2 -time1)/CLOCKS_PER_SEC,(double)(time2 -time1)/CLOCKS_PER_SEC/60,(double)(time2 -time1)/CLOCKS_PER_SEC/3600);


        }
        fprintf(fp1, "\n");
        
    }
    time2 = clock();
    printf("#time=%10lf sec, %10lf min, %10lf H\n",(double)(time2 -time1)/CLOCKS_PER_SEC,(double)(time2 -time1)/CLOCKS_PER_SEC/60,(double)(time2 -time1)/CLOCKS_PER_SEC/3600);
    fprintf(fp1, "#time=%10lf sec, %10lf min, %10lf H\n",(double)(time2 -time1)/CLOCKS_PER_SEC,(double)(time2 -time1)/CLOCKS_PER_SEC/60,(double)(time2 -time1)/CLOCKS_PER_SEC/3600);

}
