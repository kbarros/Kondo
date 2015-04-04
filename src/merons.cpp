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

    if (argc != 3) {
         printf("Correct usage: %s <J_index> <device_index>\n", argv[0]);
         return 0;
    }
  
     int J_index = atoi(argv[1]);
     int device_index = atoi(argv[2]);
  
     Vec<double> Js {0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0};
   //Vec<int>   Q1s {   9,   8,   6,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};//96x96
   //Vec<int>   Q2s {  18,  18,  17,  15,  14,  13,  12,  11,  10,   9,   7,   3,   2,   2};//96x96
     Vec<int>   Q1s {   8,   8,   7,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};//120x120
     Vec<int>   Q2s {  26,  26,  25,  21,  18,  18,  18,  18,  18,  18,  18,  18,  18,  18};//120x120

     if (J_index < 0 || J_index >= Js.size()) {
     	printf("J index %d is out of bounds [0 %d]\n", J_index, (int)Js.size());
      //	printf("J index %d is out of bounds [0 10]\n", index);
        return 0;
     }
  
     double J = Js[J_index];
     int Q1 = Q1s[J_index];
     int Q2 = Q2s[J_index];
     printf("Device index=%d, J value=%g\n", device_index, J);
    
    clock_t time1, time2;
    time1 = clock();
    
    RNG rng(0);
    int w = 120;
    double t1 = -1, t2 = 0, t3 = 0.5, s1 = 0;
    double kT =  0;
    int n_colors = 15*15;
    //Vec<int> Ms {500, 1000, 2000, 4000, 8000};
    //Vec<int> Ms {500, 1000, 2000, 4000};
    //Vec<int> Ms {500, 1000, 2000};
    Vec<int> Ms {1000,2000};
    //Vec<int> Ms {500, 1000};
    EnergyScale es{-5.0-J, 7.0+J};

    
    double min_J = J;//0.1;
    double max_J = J+ 1e-4;;//0.1 + 1e-4;
    double d_J = 0.1; // 0.1;
    
    double min_mu = -2.70;//-2.6;
    double max_mu = -2.20  +1e-4;//-2.3 + 1e-4;
    double d_mu = 0.01;//0.01; // 0.5;

    double min_a = 0.0;//step3
    //double min_a = 1.0;//step1
    double max_a = 1.0 + 1e-4;//1.0 + 1e-4;
    double d_a   = 0.1;
    double meron_a;

    __attribute__((unused))
    int min_Q = Q1, max_Q = Q2;
    int meron_Q;
    
    FILE *fp1;
    char filename1[100];
    //sprintf(filename1, "merons_variational_test_%dx%dsites_%03dclr_%05d_2.txt", w, w, n_colors, M);
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_%05d_2.txt", w, w, n_colors, Ms.back());
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_2.txt", w, w, n_colors, Ms.back());//fix Q
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J%2.4lf.txt", w, w, n_colors, Ms.back(), J);//change Q
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J%2.4lf_step1.txt", w, w, n_colors, Ms.back(), J);//change Q //fix a //step1
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J%2.4lf_step3.txt", w, w, n_colors, Ms.back(), J);//change Q //fix a //step3
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J0.10_loop.txt", w, w, n_colors, Ms.back());//change Q and random seeds
    //sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J%2.4lf_step1B.txt", w, w, n_colors, Ms.back(), J);//change Q //fix a //step1B
    sprintf(filename1, "merons_variational_error_check_%dx%dsites_%03dclr_Ms%05d_J%2.4lf_step3B.txt", w, w, n_colors, Ms.back(), J);//several Q //change a //step3B

    fp1 = fopen(filename1, "w");
    printf("filename:%s\n", filename1);
    double filling;
    auto m = Model(SquareLattice::mk(w, w, t1, t2, t3, s1), min_J, kT);
    
    //auto engine = mk_engine_cx();
    //auto engine = mk_engine<cx_double>();
    auto engine = mk_engine_cuSPARSE<cx_flt>(device_index);
    //auto engine = mk_engine<cx_double>(device_index);
    auto groups = m.lattice->groups(n_colors);
    cout << std::setprecision(9);
    cout << "# J, meron_a, meron_Q, mu, Phi, filling, M(, ED_Phi)\n";

    int loop;               

    engine->set_R_correlated(groups, rng);//step1B, step3B
    for (loop=0; loop<1; loop++){
    for (m.J = min_J; m.J < max_J; m.J += d_J) {
        for (meron_a = min_a; meron_a < max_a; meron_a += d_a) {
            //for (meron_Q=0; meron_Q<=(w/2); meron_Q++) {//step1
            for (meron_Q=Q1; meron_Q<=Q2; meron_Q++) {//step3
	   //meron_Q = w/8;
           //engine->set_R_correlated(groups, rng);//step1,step3

            dynamic_cast<SquareLattice *>(m.lattice.get())->set_spins_meron(meron_a, meron_Q, m.spin);
            m.set_hamiltonian(m.spin);
            engine->set_H(m.H, es);
            
            auto allMoments = engine->moments(Ms.back());
            
            for (int M : Ms) {
                auto moments = slice(allMoments, 0, M);
                auto gamma = moment_transform(moments, 4*M);
            
            
                for (double mu = min_mu; mu < max_mu; mu += d_mu) {
                    double Phi = electronic_grand_energy(gamma, es, kT, mu) / m.n_sites;
                    filling = mu_to_filling(gamma, es, m.kT(), mu);
                    
                    printf("%10lf, %10lf, %5d, %10lf, %10lf, %10lf, %d, ", m.J, meron_a, meron_Q, mu, Phi, filling, M);
                    fprintf(fp1, "%10lf, %10lf, %5d, %10lf, %10lf, %10lf, %d, ", m.J, meron_a, meron_Q, mu, Phi, filling, M);
                    fflush(fp1);
                    
                    //bool print_exact = true;
                    bool print_exact = false;
                    if (print_exact) {
                        //arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
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
}
