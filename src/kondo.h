#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"

using namespace fkpm;


class Lattice {
public:
    static std::unique_ptr<Lattice> mk_square(int w, int h, double t1, double t2, double t3);
    
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
protected:
    // vector components are each unit gaussian
    vec3 gaussian_vec3(RNG& rng);
    // project p to space perpendicular to x
    vec3 project_tangent(vec3 x, vec3 p);
    
public:
    typedef std::function<void(Vec<vec3> const& spin, Vec<vec3> const& force)> CalcForce;
    
    // Overdamped relaxation using Euler integration
    static std::unique_ptr<Dynamics> mk_overdamped(double kB_T, double dt);
    // Inertial Langevin dynamics using Grønbech-Jensen Farago, velocity explicit
    static std::unique_ptr<Dynamics> mk_gjf(double gamma, double kB_T, double dt);
    // Stochastic Landau Lifshitz using Heun-p
    static std::unique_ptr<Dynamics> mk_sll(double gamma, double kB_T, double dt);
    
    virtual void init_step(CalcForce const& calc_force, RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, RNG& rng, Model& m) = 0;
};


#endif /* defined(__kondo__) */
