#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"

using namespace fkpm;


// C++14 feature missing in C++11
namespace std {
    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args ) {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }
}

template<typename T>
Vec3<T> gaussian_vec3(RNG& rng) {
    static std::normal_distribution<T> dist;
    return { dist(rng), dist(rng), dist(rng) };
}

// project vector p onto plane that is normal to x
template<typename T>
Vec3<T> project_tangent(vec3 x, vec3 p) {
    return p - x * (p.dot(x) / x.norm2());
}

constexpr double Pi = 3.141592653589793238463;
constexpr cx_double I(0, 1);


class Model;

class Lattice {
public:
    static void set_spins_random(RNG& rng, Vec<vec3>& spin);
    static std::unique_ptr<Lattice> mk_linear(int w, double t1, double t2);
    static std::unique_ptr<Lattice> mk_square(int w, int h, double t1, double t2, double t3);
    static std::unique_ptr<Lattice> mk_triangular(int w, int h, double t1, double t2, double t3);
    static std::unique_ptr<Lattice> mk_kagome(int w, int h, double t1);
    
    virtual int n_sites() = 0;
    virtual vec3 position(int i) = 0;
    virtual void set_spins(std::string const& name, Vec<vec3>& spin) = 0;
    virtual void add_hoppings(Model const& model, SpMatElems<cx_double>& H_elems) = 0;
    virtual Vec<int> groups(int n_colors) = 0;
};


class Model {
public:
    int n_sites;
    std::unique_ptr<Lattice> lattice;
    double J, kB_T;
    vec3 B_zeeman;
    vec3 current; double current_growth, current_freq;
    SpMatElems<cx_double> H_elems;
    SpMatCsr<cx_double> H, D;
    Vec<vec3> spin;
    double time = 0;
    
    // used by Dynamics to store intermediate data between steps
    Vec<vec3> dyn_stor[4];
    
    Model(std::unique_ptr<Lattice> lattice, double J, double kB_T, vec3 B_zeeman={0,0,0},
          vec3 current={0,0,0}, double current_growth=0, double current_freq=0);
    
    void set_hamiltonian(Vec<vec3> const& spin);
    double classical_potential();
    void set_forces(SpMatCsr<cx_double> const& D, Vec<vec3>& force);
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
    
    virtual void init(CalcForce const& calc_force, RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, RNG& rng, Model& m) = 0;
};


#endif /* defined(__kondo__) */
