#include "iostream_util.h"
#include "kondo.h"

using namespace fkpm;

void testKondo1() {
    int w = 6, h = 6;
    int n_sites = w*h;
    double t1 = -1, t2 = 0, t3 = -0.5;
    double J = 0.5;
    double kB_T = 0;
    double mu = 0.103;
    
    auto m = Model(J, Lattice::square(w, h, t1, t2, t3));
    m.lattice->set_spins("ferro", m.spin);
    m.spin[0] = vec3(1, 1, 1).normalize();
    
    m.set_hamiltonian();
    
    arma::vec eigs = arma::real(arma::eig_gen(m.H.to_arma_dense()));
    std::sort(eigs.begin(), eigs.end());
    
    cout << std::setprecision(9);
    // cout << m.H.to_arma_dense() << endl;
    // cout << "eigs " << eigs << endl;
    cout << "grand energy " << electronic_grand_energy(eigs, kB_T, mu) / n_sites << endl;

}

int main(int argc,char **argv) {
    testKondo1();
}

