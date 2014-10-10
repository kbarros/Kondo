
#ifndef __iostream__util__
#define __iostream__util__

#include <iostream>
#include <iomanip>
#include <vector>

template<class T>
std::ostream& operator<< (std::ostream& os, std::vector<T> const& v) {
    os << "vector { ";
    for (auto const& x : v) {
        os << x << " ";
    }
    os << "}";
    return os;
}

using std::cout;
using std::cerr;
using std::endl;

#endif /* defined(__iostream__util__) */
