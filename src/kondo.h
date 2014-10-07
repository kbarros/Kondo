#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"

template <typename T>
using Vec = std::vector<T>;


// hamiltonian.cpp

class Lattice {
public:
    static std::unique_ptr<Lattice> build_square(int w, int h, double t1, double t2, double t3);

    Vec<vec3<double>> spins;
    Vec<vec3<double>> forces;
    virtual int n_sites() = 0;
    virtual void set_spins(std::string const& name) = 0;
    virtual void fill_hoppings(fkpm::SpMatCoo<fkpm::cx_double>& H) = 0;
};

class Hamiltonian {
public:
    int n_sites;
    fkpm::SpMatCoo<fkpm::cx_double> H;
    std::unique_ptr<Lattice> lattice;
    
    fkpm::SpMatCoo<fkpm::cx_double>& build();
    Vec<vec3<double>>& calculate_forces(std::function<double(int, int)> const& dE_dH);
};

#endif /* defined(__kondo__) */
