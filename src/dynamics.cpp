#include "kondo.h"


class Overdamped: public Dynamics {
public:
    double kB_T;
    Overdamped(double kB_T, double dt): kB_T(kB_T) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, RNG& rng, Model& m) {
        Vec<vec3>& f = m.dyn_stor[0];
        calc_force(m.spin, f);
        for (int i = 0; i < m.n_sites; i++) {
            vec3 beta = sqrt(dt*2*kB_T) * gaussian_vec3<double>(rng);
            m.spin[i] += project_tangent<double>(m.spin[i], dt*f[i]+beta);
            m.spin[i] = m.spin[i].normalized();
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_overdamped(double kB_T, double dt) {
    return std::make_unique<Overdamped>(kB_T, dt);
}


class GJF: public Dynamics {
public:
    double alpha, kB_T;
    double a, b;
    double mass = 1;
    
    GJF(double alpha, double kB_T, double dt): alpha(alpha), kB_T(kB_T) {
        this->dt = dt;
        double denom = 1 + alpha*dt/(2*mass);
        a = (1 - alpha*dt/(2*mass))/denom;
        b = 1 / denom;
    }
    
    void init(CalcForce const& calc_force, RNG& rng, Model& m) {
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        v.assign(m.n_sites, vec3(0,0,0));
        calc_force(m.spin, f1);
    }
    
    void step(CalcForce const& calc_force, RNG& rng, Model& m) {
        Vec<vec3>& s = m.spin;
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        Vec<vec3>& f2 = m.dyn_stor[2];
        Vec<vec3>& beta = m.dyn_stor[3];
        
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*alpha*kB_T) * gaussian_vec3<double>(rng);
            vec3 ds = b*dt*v[i] + (b*dt*dt/(2*mass))*f1[i] + (b*dt/(2*mass))*beta[i];
            s[i] += project_tangent<double>(s[i], ds);
            s[i] = s[i].normalized();
        }
        
        calc_force(m.spin, f2);
        
        for (int i = 0; i < m.n_sites; i++) {
            v[i] = a*v[i] + (dt/(2*mass))*(a*f1[i] + f2[i]) + (b/mass)*beta[i];
            v[i] = project_tangent<double>(s[i], v[i]);
            
            // forces will be reused in the next timestep
            f1[i] = f2[i];
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_gjf(double alpha, double kB_T, double dt) {
    return std::make_unique<GJF>(alpha, kB_T, dt);
}


class SLL: public Dynamics {
public:
    double alpha, kB_T;
    
    SLL(double alpha, double kB_T, double dt): alpha(alpha), kB_T(kB_T) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        Vec<vec3>& f    = m.dyn_stor[1];
        Vec<vec3>& fp   = m.dyn_stor[2];
        Vec<vec3>& beta = m.dyn_stor[3];
        
        double D = (alpha / (1 + alpha*alpha)) * kB_T;
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*D) * gaussian_vec3<double>(rng);
        }
        
        // one euler step accumulated into s
        auto accum_euler = [&](Vec<vec3> const& f, double scale, Vec<vec3>& s) {
            for (int i = 0; i < m.n_sites; i++) {
                vec3 a     = - f[i]    - alpha*s[i].cross(f[i]);
                vec3 sigma = - beta[i] - alpha*s[i].cross(beta[i]);
                vec3 ds    = s[i].cross(a*dt + sigma);
                s[i] += scale * ds;
            }
        };
        
        calc_force(s, f);
        sp = s;
        accum_euler(f, 1.0, sp);
        calc_force(sp, fp);
        sp = s;
        accum_euler(f,  0.5, sp);
        accum_euler(fp, 0.5, sp);
        s = sp;
        for (int i = 0; i < m.n_sites; i++) {
            s[i] = s[i].normalized();
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_sll(double alpha, double kB_T, double dt) {
    return std::make_unique<SLL>(alpha, kB_T, dt);
}


