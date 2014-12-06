#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"
#include "cpptoml.h"


typedef float flt;
// typedef double flt;
typedef std::complex<flt> cx_flt;

using fkpm::Vec;
using fkpm::Pi;
constexpr cx_flt I(0, 1);


// C++14 feature missing in C++11
namespace std {
    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args ) {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }
}

template<typename T>
Vec3<T> gaussian_vec3(fkpm::RNG& rng) {
    static std::normal_distribution<T> dist;
    return { dist(rng), dist(rng), dist(rng) };
}


class Model;

class Lattice {
public:
    static void set_spins_random(fkpm::RNG& rng, Vec<vec3>& spin);
    virtual int n_sites() = 0;
    virtual vec3 position(int i) = 0;
    virtual void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) = 0;
    virtual void add_hoppings(Model const& model, fkpm::SpMatElems<cx_flt>& H_elems) = 0;
    virtual Vec<int> groups(int n_colors) = 0;
};
class LinearLattice: public Lattice {
public:
    static std::unique_ptr<LinearLattice> mk(int w, double t1, double t2);
};
class SquareLattice: public Lattice {
public:
    static std::unique_ptr<SquareLattice> mk(int w, int h, double t1, double t2, double t3);
    virtual void set_spins_meron(double a, int q, Vec<vec3>& spin) = 0;
};
class TriangularLattice: public Lattice {
public:
    static std::unique_ptr<TriangularLattice> mk(int w, int h, double t1, double t2, double t3);
};
class KagomeLattice: public Lattice {
public:
    static std::unique_ptr<KagomeLattice> mk(int w, int h, double t1);
};


class Model {
public:
    int n_sites;
    std::unique_ptr<Lattice> lattice;
    double J, kB_T;
    vec3 B_zeeman;
    vec3 current; double current_growth, current_freq;
    fkpm::SpMatElems<cx_flt> H_elems;
    fkpm::SpMatBsr<cx_flt> H, D;
    Vec<vec3> spin;
    double time = 0;
    
    // used by Dynamics to store intermediate data between steps
    Vec<vec3> dyn_stor[5];
    
    Model(std::unique_ptr<Lattice> lattice, double J, double kB_T, vec3 B_zeeman={0,0,0},
          vec3 current={0,0,0}, double current_growth=0, double current_freq=0);
    
    void set_hamiltonian(Vec<vec3> const& spin);
    double classical_potential();
    void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3>& force);
};


class Dynamics {
public:
    typedef std::function<void(Vec<vec3> const& spin, Vec<vec3>& force)> CalcForce;
    int n_steps = 0;
    double dt;
    
    // Overdamped relaxation using Euler integration
    static std::unique_ptr<Dynamics> mk_overdamped(double dt);
    // Inertial Langevin dynamics using Grønbech-Jensen Farago, velocity explicit
    static std::unique_ptr<Dynamics> mk_gjf(double alpha, double dt);
    // Stochastic Landau Lifshitz using Heun-p
    static std::unique_ptr<Dynamics> mk_sll(double alpha, double dt);
    
    virtual void init(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) = 0;
};


#endif /* defined(__kondo__) */
