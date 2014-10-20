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


class Lattice {
public:
    static void set_spins_random(RNG& rng, Vec<vec3>& spin);
    static std::unique_ptr<Lattice> mk_square(int w, int h, double t1, double t2, double t3);
    static std::unique_ptr<Lattice> mk_triangular(int w, int h, double t1, double t2, double t3);
    static std::unique_ptr<Lattice> mk_kagome(int w, int h, double t1);
    
    virtual int n_sites() = 0;
    virtual vec3 position(int i) = 0;
    virtual void set_spins(std::string const& name, Vec<vec3>& spin) = 0;
    virtual void add_hoppings(SpMatCoo<cx_double>& H) = 0;
};


class Model {
public:
    int n_sites;
    std::unique_ptr<Lattice> lattice;
    double J;
    SpMatCoo<cx_double> H;
    
    Vec<vec3> spin;
    
    // used by Dynamics to store intermediate data between steps
    Vec<vec3> dyn_stor[4];
    
    Model(std::unique_ptr<Lattice> lattice, double J);
    
    SpMatCoo<cx_double>& set_hamiltonian(Vec<vec3> const& spin);
    void set_forces(std::function<cx_double(int, int)> const& D, Vec<vec3>& force);
};


class Dynamics {
public:
    typedef std::function<void(Vec<vec3> const& spin, Vec<vec3>& force)> CalcForce;
    double dt;
    
    // Overdamped relaxation using Euler integration
    static std::unique_ptr<Dynamics> mk_overdamped(double kB_T, double dt);
    // Inertial Langevin dynamics using Grønbech-Jensen Farago, velocity explicit
    static std::unique_ptr<Dynamics> mk_gjf(double alpha, double kB_T, double dt);
    // Stochastic Landau Lifshitz using Heun-p
    static std::unique_ptr<Dynamics> mk_sll(double alpha, double kB_T, double dt);
    
    virtual void init_step(CalcForce const& calc_force, RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, RNG& rng, Model& m) = 0;
};


#endif /* defined(__kondo__) */
