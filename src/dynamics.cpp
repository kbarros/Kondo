#include "kondo.h"

class Overdamped: public Dynamics {
public:
    double kB_T;
    Overdamped(double kB_T): kB_T(kB_T) {}
    
    void step(CalcForce const& calc_force, double dt, Model& m) {
        Vec<vec3>& force = m.scratch1;
        calc_force(m.spin, force);
        for (int i = 0; i < m.n_sites; i++) {
            
        }
    }
};

std::unique_ptr<Dynamics> Dynamics::mk_overdamped(double kB_T) {
    return std::unique_ptr<Dynamics>(new Overdamped(kB_T));
}
