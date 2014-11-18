#include "kondo.h"
#include "iostream_util.h"

#include <cassert>


Model::Model(std::unique_ptr<Lattice> lattice, double J, double kB_T, vec3 B_zeeman,
             vec3 current, double current_growth, double current_freq):
    n_sites(lattice->n_sites()), lattice(std::move(lattice)), J(J), kB_T(kB_T), B_zeeman(B_zeeman),
    current(current), current_growth(current_growth), current_freq(current_freq)
{
    spin.assign(n_sites, vec3{0, 0, 0});
    
    dyn_stor[0].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[1].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[2].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[3].assign(n_sites, vec3{0, 0, 0});
}


// {s1, s2} components of pauli matrix vector,
// sigma1     sigma2     sigma3
//  0  1       0 -I       1  0
//  1  0       I  0       0 -1
static Vec3<fkpm::cx_double> pauli[2][2] {
    {{0, 0, 1}, {1, -I, 0}},
    {{1, I, 0}, {0, 0, -1}}
};

void Model::set_hamiltonian(Vec<vec3> const& spin) {
    H_elems.clear();
    
    lattice->add_hoppings(*this, H_elems);
    
    for (int i = 0; i < n_sites; i++) {
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                H_elems.add(2*i+s1, 2*i+s2, -J * pauli[s1][s2].dot(spin[i]));
            }
        }
    }
    H.build(2*n_sites, 2*n_sites, H_elems);
    D = H;
    D.zeros();
}

double Model::classical_potential() {
    double acc = 0;
    for (int i = 0; i < n_sites; i++) {
        acc += -B_zeeman.dot(spin[i]);
    }
    return acc;
}

void Model::set_forces(SpMatCsr<cx_double> const& D, Vec<vec3>& force) {
    for (int k = 0; k < n_sites; k++) {
        Vec3<fkpm::cx_double> dE_dS(0, 0, 0);
        
        // Apply chain rule: dE/dS = dH_ij/dS D_ji
        // where D_ij = dE/dH_ji is the density matrix
        for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
                auto dH_ij_dS = -J * pauli[s1][s2];
                auto D_ji = D(2*k+s2, 2*k+s1);
                dE_dS += dH_ij_dS * D_ji;
            }
        }
        
        // add extra term that couples to current J
        
        
        assert(imag(dE_dS).norm() < 1e-5);
        force[k] = -real(dE_dS);
        
        force[k] += B_zeeman;
    }
}
