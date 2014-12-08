#include "kondo.h"


// project vector p onto plane that is normal to x
vec3 project_tangent(vec3 x, vec3 p) {
    return p - x * (p.dot(x) / x.norm2());
}


class Overdamped: public Dynamics {
public:
    Overdamped(double dt) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& f = m.dyn_stor[0];
        calc_force(m.spin, f);
        for (int i = 0; i < m.n_sites; i++) {
            vec3 beta = sqrt(dt*2*m.kT()) * gaussian_vec3<double>(rng);
            m.spin[i] += project_tangent(m.spin[i], dt*f[i]+beta);
            m.spin[i] = m.spin[i].normalized();
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_overdamped(double dt) {
    return std::make_unique<Overdamped>(dt);
}


class GJF: public Dynamics {
public:
    double alpha;
    double a, b;
    double mass = 1;
    
    GJF(double alpha, double dt): alpha(alpha) {
        this->dt = dt;
        double denom = 1 + alpha*dt/(2*mass);
        a = (1 - alpha*dt/(2*mass))/denom;
        b = 1 / denom;
    }
    
    void init(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        v.assign(m.n_sites, vec3(0,0,0));
        calc_force(m.spin, f1);
    }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s = m.spin;
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        Vec<vec3>& f2 = m.dyn_stor[2];
        Vec<vec3>& beta = m.dyn_stor[3];
        
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*alpha*m.kT()) * gaussian_vec3<double>(rng);
            vec3 ds = b*dt*v[i] + (b*dt*dt/(2*mass))*f1[i] + (b*dt/(2*mass))*beta[i];
            s[i] += project_tangent(s[i], ds);
            s[i] = s[i].normalized();
        }
        
        calc_force(s, f2);
        
        for (int i = 0; i < m.n_sites; i++) {
            v[i] = a*v[i] + (dt/(2*mass))*(a*f1[i] + f2[i]) + (b/mass)*beta[i];
            v[i] = project_tangent(s[i], v[i]);
            
            // forces will be reused in the next timestep
            f1[i] = f2[i];
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
    
    double pseudo_kinetic_energy(Model const& m) {
        Vec<vec3> const& v = m.dyn_stor[0];
        double acc = 0;
        for (int i = 0; i < m.n_sites; i++) {
            acc += 0.5 * mass * v[i].norm2();
        }
        return acc;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_gjf(double alpha, double dt) {
    return std::make_unique<GJF>(alpha, dt);
}


class SLL: public Dynamics {
public:
    double alpha;
    
    SLL(double alpha, double dt): alpha(alpha) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        Vec<vec3>& spp  = m.dyn_stor[1];
        Vec<vec3>& f    = m.dyn_stor[2];
        Vec<vec3>& fp   = m.dyn_stor[3];
        Vec<vec3>& beta = m.dyn_stor[4];
        
        double D = (alpha / (1 + alpha*alpha)) * m.kT();
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*D) * gaussian_vec3<double>(rng);
        }
        
        // one euler step starting from s (with force f), accumulated into sp
        auto accum_euler = [&](Vec<vec3> const& s, Vec<vec3> const& f, double scale, Vec<vec3>& sp) {
            for (int i = 0; i < m.n_sites; i++) {
                vec3 a     = - f[i]    - alpha*s[i].cross(f[i]);
                vec3 sigma = - beta[i] - alpha*s[i].cross(beta[i]);
                sp[i] += scale * s[i].cross(a*dt + sigma);
            }
        };
        
        calc_force(s, f);
        sp = s;
        accum_euler(s, f, 1.0, sp);
        calc_force(sp, fp);
        spp = s;
        accum_euler(s, f,  0.5, spp);
        accum_euler(sp, fp, 0.5, spp);
        s = spp;
        for (int i = 0; i < m.n_sites; i++) {
            s[i] = s[i].normalized();
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_sll(double alpha, double dt) {
    return std::make_unique<SLL>(alpha, dt);
}


