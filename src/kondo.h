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

class Model {
public:
    int n_sites; // Number of classical spins
    int n_rows;  // Number of rows in Hamilitonian
    double kT_init = 0, kT_decay = 0;
    vec3 B_zeeman = {0, 0, 0};
    double easy_z = 0;
    fkpm::SpMatElems<cx_flt> H_elems, D_elems;
    fkpm::SpMatBsr<cx_flt> H, D;
    Vec<vec3> spin;
    double time = 0;
    Vec<vec3> dyn_stor[5]; // used by Dynamics to store intermediate data between steps
    
    Model(int n_sites, int n_rows);
    static void set_spins_random(fkpm::RNG& rng, Vec<vec3>& spin);
    double kT();
    
    void set_hamiltonian(Vec<vec3> const& spin);
    virtual void accum_hamiltonian_hopping() = 0;
    virtual void accum_hamiltonian_hund(Vec<vec3> const& spin) = 0;
    
    void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force);
    virtual void accum_forces_classical(Vec<vec3> const& spin, Vec<vec3>& force);
    virtual void accum_forces_hund(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) = 0;
    
    virtual double energy_classical(Vec<vec3> const& spin);
    
    virtual vec3 position(int i) = 0;
    virtual void set_spins(std::string const& name, std::shared_ptr<cpptoml::toml_group> params, Vec<vec3>& spin) = 0;
    virtual Vec<int> groups(int n_colors) = 0;
};

class SimpleModel: public Model {
public:
    double J = 0;
    double t1=0, t2=0, t3=0;
    double s1=0, s2=0, s3=0;
    // vec3 current = {0, 0, 0}; double current_growth = 0, current_freq = 0;
    
    SimpleModel(int n_sites);
    
    void accum_hamiltonian_hopping();
    void accum_hamiltonian_hund(Vec<vec3> const& spin);
    
    void accum_forces_classical(Vec<vec3> const& spin, Vec<vec3>& force);
    void accum_forces_hund(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force);
    
    double energy_classical(Vec<vec3> const& spin);
    
    virtual void set_neighbors(int rank, int k, Vec<int>& idx) = 0;
    
    // instantiations
    static std::unique_ptr<SimpleModel> mk_linear(int w);
    static std::unique_ptr<SimpleModel> mk_square(int w, int h);
    static std::unique_ptr<SimpleModel> mk_triangular(int w, int h);
    static std::unique_ptr<SimpleModel> mk_kagome(int w, int h);
};

class Dynamics {
public:
    typedef std::function<void(Vec<vec3> const& spin, Vec<vec3>& force)> CalcForce;
    int n_steps = 0;
    double dt;
    
    // Overdamped relaxation using Euler integration
    static std::unique_ptr<Dynamics> mk_overdamped(double dt);
    // Inertial Langevin dynamics using Grønbech-Jensen Farago, velocity explicit
    // NOTE: This integrator is only O(dt) accurate due to poor constraint implementation
    static std::unique_ptr<Dynamics> mk_gjf(double alpha, double dt);
    // Stochastic Landau Lifshitz using Heun-p
    static std::unique_ptr<Dynamics> mk_sll(double alpha, double dt);
    
    virtual void init(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) = 0;
    virtual double pseudo_kinetic_energy(Model const& m) { return 0; }
};


#endif /* defined(__kondo__) */
