#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"

template <typename T>
using Vec = std::vector<T>;


class Lattice {
public:
    static std::unique_ptr<Lattice> square(int w, int h, double t1, double t2, double t3);
    
    virtual int n_sites() = 0;
    virtual vec3 position(int i) = 0;
    virtual void set_spins(std::string const& name, Vec<vec3>& spin) = 0;
    virtual void add_hoppings(fkpm::SpMatCoo<fkpm::cx_double>& H) = 0;
};

class Model {
public:
    double J = 0;
    std::unique_ptr<Lattice> lattice;
    fkpm::SpMatCoo<fkpm::cx_double> H;
    
    Vec<vec3> spin;
    Vec<vec3> force;
    Vec<vec3> vel;
    
    Vec<vec3> scratch1;
    Vec<vec3> scratch2;
    
    Model(double J, std::unique_ptr<Lattice> lattice);
    
    fkpm::SpMatCoo<fkpm::cx_double>& set_hamiltonian(Vec<vec3> const& spin);
    void set_forces(std::function<fkpm::cx_double(int, int)> const& D, Vec<vec3>& force);
};

#endif /* defined(__kondo__) */
