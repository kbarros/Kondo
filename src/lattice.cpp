#include "kondo.h"
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
        return {x, y, 0};
    }
    
    void set_spins(std::string const& name, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(w*h, vec3{0, 0, 1});
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

std::unique_ptr<Lattice> Lattice::mk_square(int w, int h, double t1, double t2, double t3) {
    return std::make_unique<SquareLattice>(w, h, t1, t2, t3);
}
