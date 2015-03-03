#ifndef __wannier__
#define __wannier__

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


class Wannier;


class Wannier {
public:
    int n_kpts;
    int n_orbitals;
//    vec3 hopping_vector[n_kpts];
    fkpm::SpMatElems<cx_flt> H_elems;
    
    // used by Dynamics to store intermediate data between steps
    
 //   Wannier(int n_kpts,int n_orbitals, double J,vec3 hopping_vectors[n_kpts]);
    
    //void set_hamiltonian(Vec<vec3> const& spin);
};



#endif /* defined(__wannier__) */
