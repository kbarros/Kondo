#include "kondo.h"
#include "iostream_util.h"
#include <cassert>


Model::Model(int n_sites, int n_rows):
    n_sites(n_sites),
    n_rows(n_rows),
    H_elems(n_rows, n_rows, 1),
    D_elems(n_rows, n_rows, 1)
{
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
    return kT_init * exp(- time*kT_decay);
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
    D_elems.clear();
    accum_hamiltonian_hund(spin);
    accum_hamiltonian_hopping();
    H.build(H_elems);
    D.build(D_elems);
}

void Model::set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    for (auto& f : force) {
        f = {0,0,0};
    }
    accum_forces_classical(spin, force);
    accum_forces_hund(D, spin, force);
}

void Model::accum_forces_classical(Vec<vec3> const& spin, Vec<vec3>& force) {
    for (int i = 0; i < n_sites; i++) {
        force[i] += B_zeeman;
        force[i].z += easy_z*spin[i].z;
    }
}

double Model::energy_classical(Vec<vec3> const& spin) {
    double acc = 0;
    for (int i = 0; i < n_sites; i++) {
        acc += -B_zeeman.dot(spin[i]);
        acc += -easy_z*spin[i].z*spin[i].z;
    }
    return acc;
}


SimpleModel::SimpleModel(int n_sites): Model(n_sites, 2*n_sites) {
}

void SimpleModel::accum_hamiltonian_hopping() {
    Vec<int> js;
    Vec<double> ts = {t1, t2, t3};
    for (int rank = 0; rank < ts.size(); rank++) {
        if (ts[rank] != 0.0) {
            cx_flt t = ts[rank];
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                
                // To handle current, would need to add phase:
                // // flt theta = 2*Pi*(dx*current.x/w + dy*current.y/h);
                // // theta *= (1 + current_growth*time) * cos(current_freq*time);
                // // cx_flt v = exp(I*theta)*flt(t);
                
                for (int j : js) {
                    H_elems.add(2*i+0, 2*j+0, &t);
                    H_elems.add(2*i+1, 2*j+1, &t);
                }
            }
        }
    }
}

void SimpleModel::accum_hamiltonian_hund(Vec<vec3> const& spin) {
    cx_flt zero = 0;
    for (int i = 0; i < n_sites; i++) {
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                cx_flt v = -flt(J) * pauli[s1][s2].dot(spin[i]);
                H_elems.add(2*i+s1, 2*i+s2, &v);
                D_elems.add(2*i+s1, 2*i+s2, &zero);
            }
        }
    }
}

void SimpleModel::accum_forces_classical(Vec<vec3> const& spin, Vec<vec3>& force) {
    Model::accum_forces_classical(spin, force);
    Vec<int> js;
    Vec<double> ss = {s1, s2, s3};
    for (int rank = 0; rank < ss.size(); rank++) {
        if (ss[rank] != 0.0) {
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    force[i] += - ss[rank] * spin[j];
                }
            }
        }
    }
}

double SimpleModel::energy_classical(Vec<vec3> const& spin) {
    double acc = Model::energy_classical(spin);
    Vec<int> js;
    Vec<double> ss = {s1, s2, s3};
    for (int rank = 0; rank < ss.size(); rank++) {
        if (ss[rank] != 0.0) {
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    acc += ss[rank] * spin[i].dot(spin[j]) / 2; // adjust for double counting
                }
            }
        }
    }
    return acc;
}

void SimpleModel::accum_forces_hund(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    for (int i = 0; i < n_sites; i++) {
        Vec3<cx_flt> dE_dS(0, 0, 0);
        for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
                // Apply chain rule: dE/dS = dH_ij/dS D_ji
                // where D_ij = dE/dH_ji is the density matrix
                Vec3<cx_flt> dH_ij_dS = -J * pauli[s1][s2];
                cx_flt D_ji = *D(2*i+s2, 2*i+s1);
                dE_dS += dH_ij_dS * D_ji;
            }
        }
        assert(imag(dE_dS).norm() < 1e-5);
        force[i] += -real(dE_dS);
    }
}

static inline int positive_mod(int i, int n) {
    return (i%n + n) % n;
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


class LinearModel: public SimpleModel {
public:
    int w;
    
    LinearModel(int w): SimpleModel(w), w(w) {
    }
    
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
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x;};
        static Vec<Vec<Delta>> deltas {
            { {-1}, {+1} },
            { {-2}, {+2} },
            { {-3}, {+3} },
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            idx[n] = positive_mod(i + d[n].x, w);
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        if (n_sites % n_colors != 0) {
            std::cerr << "n_colors=" << n_colors << "is not a divisor of lattice size w=" << n_sites << std::endl;
            std::abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            colors[i] = i % n_colors;
        }
        return colors_to_groups(colors, 2);
    }
};

std::unique_ptr<SimpleModel> SimpleModel::mk_linear(int w) {
    return std::make_unique<LinearModel>(w);
}


class SquareModel: public SimpleModel {
public:
    int w, h;
    
    SquareModel(int w, int h): SimpleModel(w*h), w(w), h(h) {
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
        // printf("meron params: %10lf, %10lf, %10lf, %10lf\n", factor, a, b, 1.0-(a*a));
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            q1_phase = q1[0]*x + q1[1]*y;
            q2_phase = q2[0]*x + q2[1]*y;
            spin[i].x =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*cos(q1_phase);
            spin[i].y =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*sin(q1_phase);
            spin[i].z =b*sin(q2_phase);
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x; int y;};
        static Vec<Vec<Delta>> deltas {
            { {1, 0}, {0, 1}, {-1, 0}, {0, -1} },
            { {1, 1}, {-1, 1}, {-1, -1}, {1, -1} },
            { {2, 0}, {0, 2}, {-2, 0}, {0, -2} },
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        
        int x = i % w;
        int y = i / w;
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            int xp = positive_mod(x+d[n].x, w);
            int yp = positive_mod(y+d[n].y, h);
            idx[n] = xp + yp*w;
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            std::abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            int cx = x % c_len;
            int cy = y % c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_square(int w, int h) {
    return std::make_unique<SquareModel>(w, h);
}


class TriangularModel: public SimpleModel {
public:
    int w, h;
    
    TriangularModel(int w, int h): SimpleModel(w*h), w(w), h(h) {
    }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        double a = 1.0;                // horizontal distance between columns
        double b = 0.5*sqrt(3.0)*a;    // vertical distance between rows
        return {a*x - 0.5*a*y, b*y, 0};
    }
    
    void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
        } else if (name == "allout") {
            for (int i = 0; i < n_sites; i++) {
                int x = i % w;
                int y = i / w;
                switch (2*(y%2) + x%2) {
                    case 0: spin[i] = vec3(+1, +1, +1).normalized(); break;
                    case 1: spin[i] = vec3(-1, +1, -1).normalized(); break;
                    case 2: spin[i] = vec3(+1, -1, -1).normalized(); break;
                    case 3: spin[i] = vec3(-1, -1, +1).normalized(); break;
                }
                std::abort();
            }
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x; int y;};
        static Vec<Vec<Delta>> deltas {
            // . . C 1 B
            // . 2 c b 0
            // D d * a A
            // 3 e f 5 .
            // E 4 F . .
            { {1, 0}, {1, 1}, {0,  1}, {-1,  0}, {-1, -1}, {0, -1} }, // a b c d e f
            { {2, 1}, {1, 2}, {-1, 1}, {-2, -1}, {-1, -2}, {1, -1} }, // 0 1 2 3 4 5
            { {2, 0}, {2, 2}, {0,  2}, {-2,  0}, {-2, -2}, {0, -2} }, // A B C D E F
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        
        int x = i % w;
        int y = i / w;
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            int xp = positive_mod(x+d[n].x, w);
            int yp = positive_mod(y+d[n].y, h);
            idx[n] = xp + yp*w;
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            std::abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            int cx = x % c_len;
            int cy = y % c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors, 2);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_triangular(int w, int h) {
    return std::make_unique<TriangularModel>(w, h);
}


class KagomeModel: public SimpleModel {
public:
    int w, h;
    
    KagomeModel(int w, int h): SimpleModel(3*w*h), w(w), h(h) {
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
        for (int i = 0; i < n_sites; i++) {
            int v = i%3;
            int x = (i/3)%w;
            int y = i/(3*w);

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
            spin[i] = s.normalized();
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
    
    void set_neighbors(int rank, int k, Vec<int>& idx) {
        if (rank != 0) {
            std::cerr << "Only nearest neighbors supported on kagome lattice\n";
            std::abort();
        }
        idx.resize(4);
        int v = k%3;
        int x = (k/3)%w;
        int y = k/(3*w);
        
        auto coord2idx = [&](int v, int x, int y) -> int {
            int xp = (x%w+w)%w;
            int yp = (y%h+h)%h;
            return v + xp*(3) + yp*(3*w);
        };
        
        if (v == 0) {
            //     1
            //    /D\       /E
            //   2 - 0 --- 2 -
            //         \ /
            //          1
            //         /B\.
            idx[0] = coord2idx(2, x+1, y+0);
            idx[1] = coord2idx(1, x+0, y+0);
            idx[2] = coord2idx(2, x+0, y+0);
            idx[3] = coord2idx(1, x+1, y-1);
        } else if (v == 1) {
            //     D\       /E
            //     - 0 --- 2 -
            //         \ /
            //          1
            //         /B\
            //     - 2 --- 0 -
            idx[0] = coord2idx(2, x+0, y+1);
            idx[1] = coord2idx(0, x-1, y+1);
            idx[2] = coord2idx(2, x+0, y+0);
            idx[3] = coord2idx(0, x+0, y+0);
        } else if (v == 2) {
            //              1
            //    D\       /E\
            //    - 0 --- 2 - 0 -
            //        \ /
            //         1
            //        /B\.
            idx[0] = coord2idx(0, x+0, y+0);
            idx[1] = coord2idx(1, x+0, y+0);
            idx[2] = coord2idx(0, x-1, y+0);
            idx[3] = coord2idx(1, x+0, y-1);
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        if (n_colors%3 != 0) {
            std::cerr << "n_colors=" << n_colors << " is not a multiple of 3\n";
            std::abort();
        }
        int c_len = int(sqrt(n_colors/3));
        if (c_len*c_len != n_colors/3) {
            std::cerr << "n_colors/3=" << n_colors/3 << " is not a perfect square\n";
            std::abort();
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors/3)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::abort();
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int v = i%3;
            int x = (i/3)%w;
            int y = i/(3*w);
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = 3*(cy*c_len+cx) + v;
        }
        return colors_to_groups(colors, 2);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_kagome(int w, int h) {
    return std::make_unique<KagomeModel>(w, h);
}
