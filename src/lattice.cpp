#include "kondo.h"
#include <cassert>


void Lattice::set_spins_random(RNG &rng, Vec<vec3> &spin) {
    for (int i = 0; i < spin.size(); i++) {
        spin[i] = gaussian_vec3<double>(rng).normalized();
    }
}

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
            spin.assign(n_sites(), vec3{0, 0, 1});
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    int coord2idx(int x, int y) {
        int xp = (x%w+w)%w;
        int yp = (y%h+h)%h;
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
                
                // nn3 dx and dy are same as nn1, but multiplied by 2
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int j = coord2idx(x + 2*nn1_dx[nn], y + 2*nn1_dy[nn]);
                    H.add(2*i+0, 2*j+0, t3);
                    H.add(2*i+1, 2*j+1, t3);
                }
            }
        }
    }
};

std::unique_ptr<Lattice> Lattice::mk_square(int w, int h, double t1, double t2, double t3) {
    assert(t2 == 0);
    return std::make_unique<SquareLattice>(w, h, t1, t2, t3);
}



class TriangularLattice: public Lattice {
public:
    int w, h;
    double t1, t2, t3;
    
    TriangularLattice(int w, int h, double t1, double t2, double t3):
    w(w), h(h), t1(t1), t2(t2), t3(t3)
    {}
    
    int n_sites() { return w*h; }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        double a = 1.0;                // horizontal distance between columns
        double b = 0.5*sqrt(3.0)*a;    // vertical distance between rows
        return {a*x - 0.5*a*y, b*y, 0};

    }
    
    int coord2idx(int x, int y) {
        int xp = (x%w+w)%w;
        int yp = (y%h+h)%h;
        return xp + yp*w;
    }
    
    void set_spins(std::string const& name, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites(), vec3{0, 0, 1});
        } else if (name == "allout") {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int i = coord2idx(x, y);
                    switch (2*(y%2) + x%2) {
                        case 0: spin[i] = vec3(+1, +1, +1).normalized(); break;
                        case 1: spin[i] = vec3(-1, +1, -1).normalized(); break;
                        case 2: spin[i] = vec3(+1, -1, -1).normalized(); break;
                        case 3: spin[i] = vec3(-1, -1, +1).normalized(); break;
                    }
                    abort();
                }
            }
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    void add_hoppings(fkpm::SpMatCoo<fkpm::cx_double>& H) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                
                //      2   1
                //      | /
                //  3 - o - 0
                //    / |
                //  4   5
                static int nn1_sz = 6;
                static int nn1_dx[] {1, 1, 0, -1, -1, 0};
                static int nn1_dy[] {0, 1, 1, 0, -1, -1};
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int j = coord2idx(x + nn1_dx[nn], y + nn1_dy[nn]);
                    H.add(2*i+0, 2*j+0, t1);
                    H.add(2*i+1, 2*j+1, t1);
                }
                
                // nn3 dx and dy are same as nn1, but multiplied by 2
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int j = coord2idx(x + 2*nn1_dx[nn], y + 2*nn1_dy[nn]);
                    H.add(2*i+0, 2*j+0, t3);
                    H.add(2*i+1, 2*j+1, t3);
                }
            }
        }
    }
};

std::unique_ptr<Lattice> Lattice::mk_triangular(int w, int h, double t1, double t2, double t3) {
    assert(t2 == 0);
    return std::make_unique<TriangularLattice>(w, h, t1, t2, t3);
}


class KagomeLattice: public Lattice {
public:
    int w, h;
    double t1;
    
    KagomeLattice(int w, int h, double t1):
    w(w), h(h), t1(t1)
    {}
    
    int n_sites() { return 3*w*h; }
    
    int coord2idx(int v, int x, int y) {
        int xp = (x%w+w)%w;
        int yp = (y%h+h)%h;
        return v + xp*(3) + yp*(3*w);
    }
    
    //
    //         1         1         1
    //        /D\       /E\       /F\
    //   --- 2 - 0 --- 2 - 0 --- 2 - 0
    //   \ /       \ /       \ /
    //    1         1         1
    //   /A\       /B\       /C\
    //  2 - 0 --- 2 - 0 --- 2 - 0 ---
    //        \ /       \ /       \ /
    //
    vec3 position(int i) {
        int v = i%3;
        int x = (i/3)%w;
        int y = i/(3*w);
        double a = 2.0;                       // horizontal distance between letters (A <-> B)
        double b = 0.5*sqrt(3)*a;             // vertical distance between letters   (A <-> D)
        double r = 1 / (2*sqrt(3))*a;         // distance between letter and number  (A <-> 0)
        double theta = -Pi/6 + (2*Pi/3)*v;    // angle from letter to number
        return {a*x + 0.5*a*y + r*cos(theta), b*y + r*sin(theta), 0};
    }
    
    void set_spins_3q(Vec<Vec<vec3>> b, Vec<vec3>& spin) {
        for (int y = 0; y < h; y++ ) {
            for (int x = 0; x < w; x++) {
                for (int v = 0; v < 3; v++) {
                    vec3 s(0, 0, 0);
                    if (x%2 == 0 && y%2 == 0) {
                        s =  b[0][v] + b[1][v] + b[2][v];
                    } else if (x%2 == 1 && y%2 == 0) {
                        s =  b[0][v] - b[1][v] - b[2][v];
                    } else if (x%2 == 0 && y%2 == 1) {
                        s = -b[0][v] + b[1][v] - b[2][v];
                    } else if (x%2 == 1 && y%2 == 1) {
                        s = -b[0][v] - b[1][v] + b[2][v];
                    }
                    spin[coord2idx(v, x, y)] = s.normalized();
                }
            }
        }
    }
    
    void set_spins(std::string const& name, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites(), vec3{0, 0, 1});
        } else if (name == "ncp1") {
            // Orthogonal sublattice vectors
            //        v0          v1          v2
            // b_i
            set_spins_3q({
                { {0, 0,  0}, {-1, 0, 0}, {0,  0, 0} },
                { {0, 0, -1}, { 0, 0, 0}, {0,  0, 0} },
                { {0, 0,  0}, { 0, 0, 0}, {0, -1, 0} }
            }, spin);
        } else if (name == "ncp2") {
            // Zero chirality on triangles
            // 5/12 filling
            set_spins_3q({
                { {-1, 0,  0}, {0,  0, 0}, {1, 0, 0} },
                { { 0, 0,  0}, {0, -1, 0}, {0, 1, 0} },
                { { 0, 0, -1}, {0,  0, 1}, {0, 0, 0} }
            }, spin);
        } else if (name == "ncp3") {
            // Vortex crystal, 1/12, 8/12 fillings
            set_spins_3q({
                { {1, 0, 0}, {0, 0, 0}, {1, 0, 0} },
                { {0, 0, 0}, {0, 1, 0}, {0, 1, 0} },
                { {0, 0, 1}, {0, 0, 1}, {0, 0, 0} }
            }, spin);
        } else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    // returns (v, x, y) indices for the `nn`th neighbor site
    int neighbor(int v, int x, int y, int nn) {
        if (v == 0) {
            //     1
            //    /D\       /E
            //   2 - 0 --- 2 -
            //         \ /
            //          1
            //         /B\.
            switch (nn) {
                case 0: return coord2idx(2, x+1, y+0);
                case 1: return coord2idx(1, x+0, y+0);
                case 2: return coord2idx(2, x+0, y+0);
                case 3: return coord2idx(1, x+1, y-1);
            }
        } else if (v == 1) {
            //     D\       /E
            //     - 0 --- 2 -
            //         \ /
            //          1
            //         /B\
            //     - 2 --- 0 -
            switch (nn) {
                case 0: return coord2idx(2, x+0, y+1);
                case 1: return coord2idx(0, x-1, y+1);
                case 2: return coord2idx(2, x+0, y+0);
                case 3: return coord2idx(0, x+0, y+0);
            }
        } else if (v == 2) {
            //              1
            //    D\       /E\
            //    - 0 --- 2 - 0 -
            //        \ /
            //         1
            //        /B\.
            switch (nn) {
                case 0: return coord2idx(0, x+0, y+0);
                case 1: return coord2idx(1, x+0, y+0);
                case 2: return coord2idx(0, x-1, y+0);
                case 3: return coord2idx(1, x+0, y-1);
            }
        }
        std::abort();
    }
    
    void add_hoppings(fkpm::SpMatCoo<fkpm::cx_double>& H) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int v = 0; v < 3; v++) {
                    int i = coord2idx(v, x, y);
                    for (int nn = 0; nn < 4; nn++) {
                        int j = neighbor(v, x, y, nn);
                        H.add(2*i+0, 2*j+0, t1);
                        H.add(2*i+1, 2*j+1, t1);
                    }
                }
            }
        }
    }
};

std::unique_ptr<Lattice> Lattice::mk_kagome(int w, int h, double t1) {
    return std::make_unique<KagomeLattice>(w, h, t1);
}
