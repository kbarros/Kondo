#include "kondo.h"
#include <cassert>


static Vec<int> colors_to_groups(Vec<int> const& colors, int n_orbs) {
    Vec<int> groups(colors.size()*n_orbs);
    for (int i = 0; i < colors.size(); i++) {
        for (int o = 0; o < n_orbs; o++) {
            groups[i*n_orbs+o] = n_orbs*colors[i] + o;
        }
    }
    return groups;
}


void Lattice::set_spins_random(RNG &rng, Vec<vec3> &spin) {
    for (int i = 0; i < spin.size(); i++) {
        spin[i] = gaussian_vec3<double>(rng).normalized();
    }
}

class LinearLattice: public Lattice {
public:
    int w;
    double t1, t2;
    
    LinearLattice(int w, double t1, double t2):
    w(w), t1(t1), t2(t2)
    { }
    
    int n_sites() { return w; }
    
    vec3 position(int i) {
        return {double(i), 0, 0};
    }
    
    void set_spins(std::string const& name, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites(), vec3{0, 0, 1});
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    int coord2idx(int x) {
        return(x%w+w)%w;
    }
    
    void add_hoppings(Model const& m, fkpm::SpMatElems<fkpm::cx_double>& H_elems) {
        for (int i = 0; i < w; i++) {
            static int nn1_sz = 2;
            static int nn1_dx[] { 1, -1 };
            for (int nn = 0; nn < nn1_sz; nn++) {
                // nn1
                int j = coord2idx(i + nn1_dx[nn]);
                H_elems.add(2*i+0, 2*j+0, t1);
                H_elems.add(2*i+1, 2*j+1, t1);
                
                // nn3, dx scaled by 2
                j = coord2idx(i + 2*nn1_dx[nn]);
                H_elems.add(2*i+0, 2*j+0, t2);
                H_elems.add(2*i+1, 2*j+1, t2);
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites());
        if (n_sites() % n_colors != 0) {
            std::cerr << "n_colors=" << n_colors << "is not a divisor of lattice size w=" << n_sites() << std::endl;
            abort();
        }
        Vec<int> colors(n_sites());
        for (int i = 0; i < n_sites(); i++) {
            colors[i] = i % n_colors;
        }
        return colors_to_groups(colors, 2);
    }
};

std::unique_ptr<Lattice> Lattice::mk_linear(int w, double t1, double t2) {
    return std::make_unique<LinearLattice>(w, t1, t2);
}


class SquareLattice: public Lattice {
public:
    int w, h;
    double t1, t2, t3;
    
    SquareLattice(int w, int h, double t1, double t2, double t3):
    w(w), h(h), t1(t1), t2(t2), t3(t3)
    { assert(t2==0 && "t2 not yet implemented for square lattice."); }
    
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
        else if (name == "meron") {
            spin.assign(n_sites(), vec3{0, 0, 1});
            
            double factor, q1[2], q2[2], q3[2], q4[2], q5[2];//k-points
            factor = Pi/4.;
            q1[0] = factor;
            q1[1] = factor;
            
            q2[0] = factor;
            q2[1] =-factor;
            
            q3[0] =    factor;
            q3[1] =-3.*factor;
            
            q4[0] = 3.*factor;
            q4[1] =   -factor;
            
            q5[0] = 3.*factor;
            q5[1] =-3.*factor;
            
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    int i = x + y*w;
                    spin[i].x =
                     0.042337*cos(q1[0]*x + q1[1]*y) + -0.279259*sin(q1[0]*x + q1[1]*y) +
                    -0.065991*cos(q2[0]*x + q2[1]*y) +  0.222163*sin(q2[0]*x + q2[1]*y) +
                    -0.006856*cos(q3[0]*x + q3[1]*y) +  0.012791*sin(q3[0]*x + q3[1]*y) +
                     0.009325*cos(q4[0]*x + q4[1]*y) + -0.010813*sin(q4[0]*x + q4[1]*y) +
                    -0.007072*cos(q5[0]*x + q5[1]*y) +  0.005165*sin(q5[0]*x + q5[1]*y);
                    
                    spin[i].y =
                     0.036650*cos(q1[0]*x + q1[1]*y) +  0.356878*sin(q1[0]*x + q1[1]*y) +
                    -0.052121*cos(q2[0]*x + q2[1]*y) +  0.173066*sin(q2[0]*x + q2[1]*y) +
                     0.010411*cos(q3[0]*x + q3[1]*y) + -0.013487*sin(q3[0]*x + q3[1]*y) +
                    -0.008524*cos(q4[0]*x + q4[1]*y) +  0.015746*sin(q4[0]*x + q4[1]*y) +
                    -0.004883*cos(q5[0]*x + q5[1]*y) +  0.004514*sin(q5[0]*x + q5[1]*y);

                    spin[i].z =
                    0.448280*cos(q1[0]*x + q1[1]*y) + -0.001420*sin(q1[0]*x + q1[1]*y) +
                    0.009444*cos(q2[0]*x + q2[1]*y) + -0.035105*sin(q2[0]*x + q2[1]*y) +
                    0.017824*cos(q3[0]*x + q3[1]*y) +  0.013186*sin(q3[0]*x + q3[1]*y) +
                    0.019415*cos(q4[0]*x + q4[1]*y) +  0.010644*sin(q4[0]*x + q4[1]*y) +
                    0.001066*cos(q5[0]*x + q5[1]*y) + -0.000823*sin(q5[0]*x + q5[1]*y);

                    spin[i] = spin[i].normalized();
                }
            }
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
    
    void add_hoppings(Model const& m, fkpm::SpMatElems<fkpm::cx_double>& H_elems) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                
                static int nn1_sz = 4;
                static int nn1_dx[] { 1, 0, -1, 0 };
                static int nn1_dy[] { 0, 1, 0, -1 };
                
                for (int nn = 0; nn < nn1_sz; nn++) {
                    auto add_hopping = [&](int dx, int dy, double t) {
                        double theta = 2*Pi*(dx*m.spin_orb_Bx/w + dy*m.spin_orb_By/h);
                        theta *= (1 + m.spin_orb_growth*m.time) * cos(m.spin_orb_freq*m.time);
                        cx_double phase = exp(I*theta);
                        int j = coord2idx(x+dx,y+dy);
                        H_elems.add(2*i+0, 2*j+0, phase*t);
                        H_elems.add(2*i+1, 2*j+1, phase*t);
                    };
                    int dx = nn1_dx[nn];
                    int dy = nn1_dy[nn];
                    add_hopping(dx, dy, t1);
                    add_hopping(2*dx, 2*dy, t3);
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites());
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            abort();
        }
        Vec<int> colors(n_sites());
        for (int i = 0; i < n_sites(); i++) {
            int x = i%w;
            int y = i/h;
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
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
    { assert(t2==0 && "t2 not yet implemented for triangular lattice."); }
    
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
    
    void add_hoppings(Model const& m, fkpm::SpMatElems<fkpm::cx_double>& H_elems) {
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
                    // nn1
                    int j = coord2idx(x + nn1_dx[nn], y + nn1_dy[nn]);
                    H_elems.add(2*i+0, 2*j+0, t1);
                    H_elems.add(2*i+1, 2*j+1, t1);

                    // nn3, dx and dy scaled by 2
                    j = coord2idx(x + 2*nn1_dx[nn], y + 2*nn1_dy[nn]);
                    H_elems.add(2*i+0, 2*j+0, t3);
                    H_elems.add(2*i+1, 2*j+1, t3);
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites());
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            abort();
        }
        Vec<int> colors(n_sites());
        for (int i = 0; i < n_sites(); i++) {
            int x = i%w;
            int y = i/h;
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
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
    
    void add_hoppings(Model const& m, fkpm::SpMatElems<fkpm::cx_double>& H_elems) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int v = 0; v < 3; v++) {
                    int i = coord2idx(v, x, y);
                    for (int nn = 0; nn < 4; nn++) {
                        int j = neighbor(v, x, y, nn);
                        H_elems.add(2*i+0, 2*j+0, t1);
                        H_elems.add(2*i+1, 2*j+1, t1);
                    }
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites());
        if (n_colors%3 != 0) {
            std::cerr << "n_colors=" << n_colors << " is not a multiple of 3\n";
            abort();
        }
        int c_len = int(sqrt(n_colors/3));
        if (c_len*c_len != n_colors/3) {
            std::cerr << "n_colors/3=" << n_colors/3 << " is not a perfect square\n";
            abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors/3)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            abort();
        }
        Vec<int> colors(n_sites());
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int cx = x%c_len;
                int cy = y%c_len;
                for (int v = 0; v < 3; v++) {
                    int i = coord2idx(v, x, y);
                    colors[i] = 3*(cy*c_len+cx) + v;
                }
            }
        }
        return colors_to_groups(colors, 2);
    }
};

std::unique_ptr<Lattice> Lattice::mk_kagome(int w, int h, double t1) {
    return std::make_unique<KagomeLattice>(w, h, t1);
}
