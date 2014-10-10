#include "kondo.h"
#include "iostream_util.h"

#include <cassert>


class SquareLattice: public Lattice {
public:
    int w, h;
    double t1, t2, t3;
    
    SquareLattice(int w, int h, double t1, double t2, double t3):
        w(w), h(h), t1(t1), t2(t2), t3(t3)
    {}
    
    int n_sites() { return w*h; }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        return { x, y, 0 };
    }
    
    void set_spins(std::string const& name, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(w*h, vec3{1, 0, 0});
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    int coord2idx(int x, int y) {
        int xp = (x+w)%w;
        int yp = (y+h)%h;
        return xp + yp*w;
    }
    
    void add_hoppings(fkpm::SpMatCoo<fkpm::cx_double>& H) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                
                static int nn1_sz = 4;
                static int nn1_dx[] { 1, 0, -1, 0 };
                static int nn1_dy[] { 0, 1, 0, -1 };
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int j = coord2idx(x + nn1_dx[nn], y + nn1_dy[nn]);
                    H.add(2*i+0, 2*j+0, t1);
                    H.add(2*i+1, 2*j+1, t1);
                }
                
                static int nn3_sz = 4;
                static int nn3_dx[] { 2, 0, -2, 0 };
                static int nn3_dy[] { 0, 2, 0, -2 };
                for (int nn = 0; nn < nn3_sz; nn++) {
                    int j = coord2idx(x + nn3_dx[nn], y + nn3_dy[nn]);
                    H.add(2*i+0, 2*j+0, t3);
                    H.add(2*i+1, 2*j+1, t3);
                }
            }
        }
    }
};

std::unique_ptr<Lattice> Lattice::square(int w, int h, double t1, double t2, double t3) {
    return std::unique_ptr<Lattice>(new SquareLattice(w, h, t1, t2, t3));
}


Model::Model(double J, std::unique_ptr<Lattice> lattice) {
    this->J = J;
    this->lattice = std::move(lattice);
    int n_sites = this->lattice->n_sites();
    
    H = fkpm::SpMatCoo<fkpm::cx_double>(2*n_sites, 2*n_sites);
    
    spin.assign(n_sites, vec3{0, 0, 0});
    force.assign(n_sites, vec3{0, 0, 0});
    vel.assign(n_sites, vec3{0, 0, 0});
    
    scratch1.assign(n_sites, vec3{0, 0, 0});
    scratch2.assign(n_sites, vec3{0, 0, 0});
}


// {s1, s2} components of pauli matrix vectors,
// sigma1     sigma2     sigma3
//  0  1       0 -I       1  0
//  1  0       I  0       0 -1
static fkpm::cx_double I(0, 1);
static Vec3<fkpm::cx_double> pauli[2][2] {
    {{0, 0, 1}, {1, -I, 0}},
    {{1, I, 0}, {0, 0, -1}}
};

fkpm::SpMatCoo<fkpm::cx_double>& Model::set_hamiltonian(Vec<vec3> const& spin) {
    H.clear();
    
    lattice->add_hoppings(H);
    
    for (int i = 0; i < lattice->n_sites(); i++) {
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                H.add(2*i+s1, 2*i+s2, -J * pauli[s1][s2].dot(spin[i]));
            }
        }
    }
    
    return H;
}

void Model::set_forces(std::function<fkpm::cx_double(int, int)> const& D, Vec<vec3>& force) {
    for (int k = 0; k < lattice->n_sites(); k++) {
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
        
        assert(imag(dE_dS).norm() < 1e-8);
        force[k] = -real(dE_dS);
    }
}
