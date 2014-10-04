#include "kondo.h"
#include "cpptoml.h"

#include "iostream_util.h"
#include <sstream>


int main()
{
    std::string x = R"(
T_elec = 0.0
T_bomd = 0.0
J = 0.1
B_zeeman = 0.0
B_orbital_idx = 0
time_per_dump = 10.0

[kpm]
  cheby_order = 500
  cheby_order_precise = 500
  n_vectors = 64
  n_vectors_precise = 64

[ensemble]
  type = "grand"
  mu = 0.0

[dynamics]
  type = "overdamped"
  gamma = 0.2
  dt = 0.2

[lattice]
  type = "triangle"
  t1 = -1.0
)";
    auto y = std::stringstream{x};
    
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    cpptoml::parser p{y};
    cpptoml::toml_group g = p.parse();
    double J = g.get_unwrap<double>("J");
    cout << "J " << J << endl;
    double t1 = g.get_unwrap<double>("lattice.t1");
    cout << "t1 " << t1 << endl;
    std::cout << std::endl;
    return 0;
}
