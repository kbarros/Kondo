#include <iomanip>
#include <cassert>
#include <boost/algorithm/string.hpp>
#include "iostream_util.h"
#include "fastkpm.h"
#include "wannier.h"


BlockMatrix::BlockMatrix(int n_orbitals):
    n_orbitals(n_orbitals), dx(0), dy(0), dz(0), elems(n_orbitals*n_orbitals)
{}


int Wannier::coord_to_idx(int x, int y, int z) {
    return z*L*L + y*L + x;
}

void Wannier::idx_to_coord(int i, int &x, int &y, int &z) {
    x = i % L;
    y = (i / L) % L;
    z = (i / L) / L;
}

Wannier::Wannier(int n_orbitals, int L, flt J, Vec<BlockMatrix> blocks):
    n_orbitals(n_orbitals), L(L), J(J), blocks(blocks), H_elems(L*L*L, L*L*L, n_orbitals)
{
    for (int z = 0; z < L; z++) {
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                int i = coord_to_idx(x, y, z);
                for (BlockMatrix &b : blocks) {
                    int xp = (x + b.dx + L) % L;
                    int yp = (y + b.dy + L) % L;
                    int zp = (z + b.dz + L) % L;
                    int j = coord_to_idx(xp, yp, zp);
                    H_elems.add(i, j, b.elems.data());
                }
            }
        }
    }
    H = fkpm::SpMatBsr<cx_flt>(H_elems);
}


Vec<BlockMatrix> test_readfile(std::string filename) {
    std::ifstream f_params(filename);
    std::string line;
  
    //Head line
    std::getline(f_params, line);
    
    
    std::vector<std::string> strs;
    
    //number of orbitals
    std::getline(f_params, line);
    boost::trim_if(line, boost::is_any_of("\t\n "));// remove the white spaces
    boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
    int n_orbitals=std::stof(strs[0]);
    
    //number of k-points
    std::getline(f_params, line);
    boost::trim_if(line, boost::is_any_of("\t\n "));
    boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
    int n_kpts=std::stof(strs[0]);
    
    int weight_kpts[n_kpts];
    int kpts_cnt = 0;
    
    while (kpts_cnt < n_kpts && std::getline(f_params, line)) {
        // std::vector<std::string> strs;
        boost::trim_if(line, boost::is_any_of("\t\n "));
        boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
        
        // store the kpoint-weight informations
        for (int i = 0; i < strs.size(); i++) {
            flt x = std::stof(strs[i]);
//            std::cout << " " << x;
            weight_kpts[kpts_cnt++]=x; //consider error
        }
    }
    std::cout << std::endl;
    
    Vec<BlockMatrix> blocks;
    
    for (int kpt_idx = 0; kpt_idx < n_kpts; kpt_idx++) {
        BlockMatrix b(n_orbitals);
        
        for (int orb_idx = 0; orb_idx < n_orbitals*n_orbitals; orb_idx++) {
            std::getline(f_params, line);
            boost::trim_if(line, boost::is_any_of("\t\n "));
            boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
            assert(strs.size() == 7);
            b.dz = std::stof(strs[0]);
            b.dy = std::stof(strs[1]);
            b.dx = std::stof(strs[2]);
            b.elems[orb_idx] = cx_flt(std::stof(strs[5]), std::stof(strs[6])) / (flt)weight_kpts[kpt_idx];
        }
        
        if (sqrt(b.dx*b.dx + b.dy*b.dy + b.dz*b.dz) < 1.1) {
            blocks.push_back(b);
        }
    }
    
//    for (int i = 0; i < blocks.size(); i++) {
//        cout << blocks[i].dx << " " << blocks[i].elems[0] << " " << blocks[i].elems[1] << endl;
//    }
    
//    std::cout << "Number of orbitals: " << n_orbitals << std::endl;
//    std::cout << "Number of k-points " << n_kpts << std::endl;
    assert(n_kpts == kpts_cnt);
    
    return blocks;
}

int main(int argc, char **argv) {
    std::string filename = "/Users/chhcl/Code/DATA/SrVO3.dat";
    // std::string filename = "/Users/kbarros/Desktop/SrVO3.dat";
    Vec<BlockMatrix> blocks = test_readfile(filename);
    assert(blocks.size() > 0);
    Wannier w(blocks[0].n_orbitals, 3, 0, blocks);
    
    auto Ha = w.H.to_arma();
    std::cout << (Ha - Ha.t()) << std::endl; // 0
    
    std::cout << arma::eig_gen(w.H.to_arma_dense()) << std::endl;
}

