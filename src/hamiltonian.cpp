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

fkpm::SpMatCoo<fkpm::cx_double>& Model::set_hamiltonian() {
    H.clear();
    
    // hopping term
    lattice->add_hoppings(H);
    
    // hund coupling term
    for (int i = 0; i < lattice->n_sites(); i++) {
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                fkpm::cx_double coupling = -J * pauli[s1][s2].dot(spin[i]);
                H.add(2*i+s1, 2*i+s2, coupling);
            }
        }
    }
    
    return H;
}

Vec<vec3>& Model::set_forces(std::function<fkpm::cx_double(int, int)> const& dE_dH) {
    // Use chain rule to transform derivative wrt matrix elements dE/dH, into derivative wrt spin indices
    //   dE/dS = dE/dH_ij dH_ij/dS
    int n_sites = lattice->n_sites();
    
    for (int i = 0; i < n_sites; i++) {
        Vec3<fkpm::cx_double> dE_dS(0, 0, 0);
        
        for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
                Vec3<fkpm::cx_double> dH_dS = -J * pauli[s1][s2];
                dE_dS += dE_dH(2*i+s1, 2*i+s2) * dH_dS;
            }
        }
        
        assert(imag(dE_dS).norm() < 1e-8);
        force[i] = -real(dE_dS);
    }
    
    return force;
}
