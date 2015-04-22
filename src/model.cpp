#include "kondo.h"
#include "iostream_util.h"
#include <cassert>


Model::Model(int n_sites, int n_rows_def, Vec<int> spinor_indices_def):
    n_sites(n_sites),
    n_rows(n_rows_def > 0 ? n_rows_def : 2*n_sites),
    H_elems(n_rows, n_rows, 1)
{
    if (spinor_indices_def.size() == 0) {
        // By default, assume every Hamiltonian index has a local moment
        assert(n_rows == 2*n_sites);
        spinor_indices = Vec<int>(n_sites);
        for (int i = 0; i < n_sites; i++) {
            spinor_indices[i] = 2*i;
        }
    }
    else {
        spinor_indices = spinor_indices_def;
    }
    assert(spinor_indices.size() == n_sites);
    
    spin.assign(n_sites, vec3{0, 0, 0});
    dyn_stor[0].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[1].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[2].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[3].assign(n_sites, vec3{0, 0, 0});
    dyn_stor[4].assign(n_sites, vec3{0, 0, 0});
}

void Model::set_spins_random(fkpm::RNG &rng, Vec<vec3> &spin) {
    for (int i = 0; i < spin.size(); i++) {
        spin[i] = gaussian_vec3<double>(rng).normalized();
    }
}

double Model::kT() {
    return kT_init * exp(- time*kT_decay_rate);
}

// {s1, s2} components of pauli matrix vector,
// sigma1     sigma2     sigma3
//  0  1       0 -I       1  0
//  1  0       I  0       0 -1
static Vec3<cx_flt> pauli[2][2] {
    {{0, 0, 1}, {1, -I, 0}},
    {{1, I, 0}, {0, 0, -1}}
};

void Model::set_hamiltonian(Vec<vec3> const& spin) {
    H_elems.clear();
    for (int k = 0; k < n_sites; k++) {
        int idx = spinor_indices[k];
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                cx_flt v = -flt(J) * pauli[s1][s2].dot(spin[k]);
                H_elems.add(idx+s1, idx+s2, &v);
            }
        }
    }
    add_hoppings(H_elems);
    H.build(H_elems);
    D = H; // TODO: don't calculate unnecessary D elements
    D.zeros();
}

double Model::classical_potential(Vec<vec3> const& spin) {
    double acc = 0;
    for (int i = 0; i < n_sites; i++) {
        acc += -B_zeeman.dot(spin[i]);
        acc += -easy_z*spin[i].z*spin[i].z;
    }
    return acc;
}

void Model::set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    for (int k = 0; k < n_sites; k++) {
        int idx = spinor_indices[k];
        
        // Electronic part
        Vec3<cx_flt> dE_dS(0, 0, 0);
        for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
                // Apply chain rule: dE/dS = dH_ij/dS D_ji
                // where D_ij = dE/dH_ji is the density matrix
                Vec3<cx_flt> dH_ij_dS = -J * pauli[s1][s2];
                cx_flt D_ji = *D(idx+s2, idx+s1);
                dE_dS += dH_ij_dS * D_ji;
            }
        }
        assert(imag(dE_dS).norm() < 1e-5);
        force[k] = -real(dE_dS);
        
        // Classical part
        force[k] += B_zeeman;
        force[k].z += easy_z*spin[k].z;
    }
}


static Vec<int> colors_to_groups(Vec<int> const& colors, int n_orbs) {
    Vec<int> groups(colors.size()*n_orbs);
    for (int i = 0; i < colors.size(); i++) {
        for (int o = 0; o < n_orbs; o++) {
            groups[i*n_orbs+o] = n_orbs*colors[i] + o;
        }
    }
    return groups;
}

class LinearModel: public Model {
public:
    int w;
    double t1, t2;
    
    LinearModel(int w, double t1, double t2): Model(w), w(w), t1(t1), t2(t2)
    {}
    
    vec3 position(int i) {
        return {double(i), 0, 0};
    }
    
    void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    int coord2idx(int x) {
        return(x%w+w)%w;
    }
    
    void add_hoppings(fkpm::SpMatElems<cx_flt>& H_elems) {
        for (int i = 0; i < w; i++) {
            static int nn1_sz = 2;
            static int nn1_dx[] { 1, -1 };
            for (int nn = 0; nn < nn1_sz; nn++) {
                // nn1
                int j = coord2idx(i + nn1_dx[nn]);
                cx_flt v1 = t1;
                H_elems.add(2*i+0, 2*j+0, &v1);
                H_elems.add(2*i+1, 2*j+1, &v1);
                
                // nn3, dx scaled by 2
                j = coord2idx(i + 2*nn1_dx[nn]);
                cx_flt v2 = t2;
                H_elems.add(2*i+0, 2*j+0, &v2);
                H_elems.add(2*i+1, 2*j+1, &v2);
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        if (n_sites % n_colors != 0) {
            std::cerr << "n_colors=" << n_colors << "is not a divisor of lattice size w=" << n_sites << std::endl;
            abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            colors[i] = i % n_colors;
        }
        return colors_to_groups(colors, 2);
    }
};

std::unique_ptr<Model> Model::mk_linear(int w, double t1, double t2) {
    return std::make_unique<LinearModel>(w, t1, t2);
}


class SquareModel: public Model {
public:
    int w, h;
    double t1, t2, t3;
    double s1;
    
    static const     int nn1_sz = 4;
    static constexpr int nn1_dx[] { 1, 0, -1, 0 };
    static constexpr int nn1_dy[] { 0, 1, 0, -1 };
    static const     int nn2_sz = 4;
    static constexpr int nn2_dx[] { 1, -1, -1,  1 };
    static constexpr int nn2_dy[] { 1,  1, -1, -1 };
    
    SquareModel(int w, int h, double t1, double t2, double t3, double s1):
    Model(w*h), w(w), h(h), t1(t1), t2(t2), t3(t3), s1(s1)
    {}
    
    double classical_potential(Vec<vec3> const& spin) override {
        double acc = Model::classical_potential(spin);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int dx = nn1_dx[nn];
                    int dy = nn1_dy[nn];
                    int j = coord2idx(x+dx, y+dy);
                    acc += s1 * spin[i].dot(spin[j]);
                }
            }
        }
        return acc/2; // adjust for double counting
    }
    
    void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) override {
        Model::set_forces(D, spin, force);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int dx = nn1_dx[nn];
                    int dy = nn1_dy[nn];
                    int j = coord2idx(x+dx, y+dy);
                    force[i] += - s1 * spin[j];
                }
            }
        }
    }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        return {x, y, 0};
    }
    
    void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
        }
        else if (name == "meron") {
            set_spins_meron(params->get_unwrap<double>("a"), params->get_unwrap<int64_t>("q"), spin);
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    void set_spins_meron(double a, int q, Vec<vec3>& spin) {
        assert(w % h == 0); // need periodicity in both dimensions
        double b = sqrt(1.0-(a*a));
        double factor, q1[2], q2[2];//k-points
        double q1_phase, q2_phase;
        
        if ((1.0-(a*a))<1e-4){// in case (1-a*a) = -0.000 at a = 1
            b = 0.0;
            a = 1.;
        }
        
        factor = 2.*Pi*q/h;
        q1[0] = factor;
        q1[1] = factor;
        
        q2[0] = factor;
        q2[1] =-factor;
        
        printf("meron params: %10lf, %10lf, %10lf, %10lf\n", factor, a, b, 1.0-(a*a));
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = x + y*w;
                
                q1_phase = q1[0]*x + q1[1]*y;
                q2_phase = q2[0]*x + q2[1]*y;
                
                spin[i].x =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*cos(q1_phase);
                spin[i].y =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*sin(q1_phase);
                spin[i].z =b*sin(q2_phase);
                
                //spin[i] = spin[i].normalized();
            }
        }
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                //int i = x + y*w;
                //printf("%3d, %3d, %10lf, %10lf, %10lf, %10lf\n", x, y, spin[i].x, spin[i].y, spin[i].z,
                //spin[i].x*spin[i].x + spin[i].y*spin[i].y + spin[i].z*spin[i].z);
            }
        }
    }
    
    int coord2idx(int x, int y) {
        int xp = (x%w+w)%w;
        int yp = (y%h+h)%h;
        return xp + yp*w;
    }
    
    void idx2coord(int i, int &x, int &y) {
        x = i%w;
        y = i/w;
    }
    
    void add_hoppings(fkpm::SpMatElems<cx_flt>& H_elems) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int i = coord2idx(x, y);
                auto add_hopping = [&](int dx, int dy, double t) {
                    if (t != 0.0) {
                        flt theta = 2*Pi*(dx*current.x/w + dy*current.y/h);
                        theta *= (1 + current_growth*time) * cos(current_freq*time);
                        cx_flt v = exp(I*theta)*flt(t);
                        int j = coord2idx(x+dx,y+dy);
                        H_elems.add(2*i+0, 2*j+0, &v);
                        H_elems.add(2*i+1, 2*j+1, &v);
                    }
                };
                for (int nn = 0; nn < nn1_sz; nn++) {
                    int dx = nn1_dx[nn];
                    int dy = nn1_dy[nn];
                    add_hopping(dx, dy, t1);
                    add_hopping(2*dx, 2*dy, t3);
                }
                for (int nn = 0; nn < nn2_sz; nn++) {
                    int dx = nn2_dx[nn];
                    int dy = nn2_dy[nn];
                    add_hopping(dx, dy, t2);
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i%w;
            int y = i/h;
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
    }
};

// Somehow, these declarations are necessary to satisfy the linker.
const int SquareModel::nn1_dx[];
const int SquareModel::nn1_dy[];
const int SquareModel::nn2_dx[];
const int SquareModel::nn2_dy[];

std::unique_ptr<Model> Model::mk_square(int w, int h, double t1, double t2, double t3, double s1) {
    return std::make_unique<SquareModel>(w, h, t1, t2, t3, s1);
}


class TriangularModel: public Model {
public:
    int w, h;
    double t1, t2, t3;
    
    TriangularModel(int w, int h, double t1, double t2, double t3):
        Model(w*h), w(w), h(h), t1(t1), t2(t2), t3(t3)
    { assert(t2==0 && "t2 not yet implemented for triangular lattice."); }
    
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
    
    void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
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
    
    void add_hoppings(fkpm::SpMatElems<cx_flt>& H_elems) {
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
                    cx_flt v1 = t1;
                    H_elems.add(2*i+0, 2*j+0, &v1);
                    H_elems.add(2*i+1, 2*j+1, &v1);
                    
                    // nn3, dx and dy scaled by 2
                    j = coord2idx(x + 2*nn1_dx[nn], y + 2*nn1_dy[nn]);
                    cx_flt v3 = t3;
                    H_elems.add(2*i+0, 2*j+0, &v3);
                    H_elems.add(2*i+1, 2*j+1, &v3);
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i%w;
            int y = i/h;
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
    }
};

std::unique_ptr<Model> Model::mk_triangular(int w, int h, double t1, double t2, double t3) {
    assert(t2 == 0);
    return std::make_unique<TriangularModel>(w, h, t1, t2, t3);
}


class KagomeModel: public Model {
public:
    int w, h;
    double t1;
    
    KagomeModel(int w, int h, double t1):
    Model(3*w*h), w(w), h(h), t1(t1)
    {}
    
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
    
    void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
        } else if (name == "ncp1") {
            // Chiral phase, 3/12, 7/12 fillings
            // Orthogonal sublattice vectors
            //        v0          v1          v2
            // b_i
            set_spins_3q({
                { {0, 0,  0}, {-1, 0, 0}, {0,  0, 0} },
                { {0, 0, -1}, { 0, 0, 0}, {0,  0, 0} },
                { {0, 0,  0}, { 0, 0, 0}, {0, -1, 0} }
            }, spin);
        } else if (name == "ncp2") {
            // Zero chirality on triangles, 5/12 filling
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
    
    void add_hoppings(fkpm::SpMatElems<cx_flt>& H_elems) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int v = 0; v < 3; v++) {
                    int i = coord2idx(v, x, y);
                    for (int nn = 0; nn < 4; nn++) {
                        int j = neighbor(v, x, y, nn);
                        cx_flt v1 = t1;
                        H_elems.add(2*i+0, 2*j+0, &v1);
                        H_elems.add(2*i+1, 2*j+1, &v1);
                    }
                }
            }
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
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
        Vec<int> colors(n_sites);
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

std::unique_ptr<Model> Model::mk_kagome(int w, int h, double t1) {
    return std::make_unique<KagomeModel>(w, h, t1);
}
