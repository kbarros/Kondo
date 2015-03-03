#ifndef __wannier__
#define __wannier__

#include "fastkpm.h"
#include "vec3.h"
#include "cpptoml.h"

typedef float flt;
// typedef double flt;
typedef std::complex<flt> cx_flt;

using fkpm::Vec;
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


class BlockMatrix {
public:
    int n_orbitals;
    int dx, dy, dz;
    Vec<cx_flt> elems;
    
    BlockMatrix(int n_orbitals);
};

class Wannier {
public:
    int n_orbitals;
    int L;
    flt J;
    
    Vec<vec3> spins;
    Vec<BlockMatrix> blocks;
    
    fkpm::SpMatElems<cx_flt> H_elems;
    fkpm::SpMatBsr<cx_flt> H, D;
    
    Wannier(int n_orbitals, int L, flt J, Vec<BlockMatrix> blocks);

    int coord_to_idx(int x, int y, int z);
    void idx_to_coord(int i, int &x, int &y, int &z);
    
    //void set_hamiltonian(Vec<vec3> const& spin);
};



#endif /* defined(__wannier__) */
