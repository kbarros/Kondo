#include "kondo.h"
#include "cpptoml.h"
#include "iostream_util.h"

#include <sstream>
#include <boost/filesystem.hpp>

using namespace std::placeholders;


std::unique_ptr<Lattice> mk_lattice(cpptoml::toml_group g) {
    auto type = g.get_unwrap<std::string>("lattice.type");
    if (type == "square") {
        return Lattice::mk_square(
            g.get_unwrap<int64_t>("lattice.w"), g.get_unwrap<int64_t>("lattice.h"), g.get_unwrap<double>("lattice.t1"),
            g.get_unwrap<double>("lattice.t2", 0.0), g.get_unwrap<double>("lattice.t3", 0.0));
    } else if (type == "triangle") {
        return nullptr;
    }
    cerr << "Unsupported lattice type `" << type << "`!\n";
    std::abort();
}

Model mk_model(cpptoml::toml_group g) {
    double B_zeeman = g.get_unwrap<double>("B_zeeman", 0);
    return {mk_lattice(g), g.get_unwrap<double>("J"), {B_zeeman, 0, 0}};
}

std::unique_ptr<Dynamics> mk_dynamics(cpptoml::toml_group g) {
    double kB_T = g.get_unwrap<double>("kB_T");
    double dt = g.get_unwrap<double>("dynamics.dt");
    auto type = g.get_unwrap<std::string>("dynamics.type");
    if (type == "overdamped") {
        return Dynamics::mk_overdamped(kB_T, dt);
    } else if (type == "gjf") {
        return Dynamics::mk_gjf(g.get_unwrap<double>("dynamics.alpha"), kB_T, dt);
    } else if (type == "sll") {
        return Dynamics::mk_sll(g.get_unwrap<double>("dynamics.alpha"), kB_T, dt);
    }
    cerr << "Unsupported dynamics type `" << type << "`!\n";
    std::abort();
}




int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <base_dir>\n";
        std::exit(EXIT_SUCCESS);
    }
    std::string base_dir(argv[1]);
    auto input_name = base_dir + "/config.toml";
    std::ifstream input_file(input_name);
    if (!input_file.is_open()) {
        cerr << "Unable to open file `" << input_name << "`!\n";
        std::abort();
    }
    
    cout << "Using input file `" << input_name << "`\n";

    cpptoml::parser p{input_file};
    cpptoml::toml_group g = p.parse();
    
    RNG rng(g.get_unwrap<int64_t>("seed"));
    double kB_T = g.get_unwrap<double>("kB_T");
    bool overwrite_dump = g.get_unwrap<bool>("overwrite_dump");
    int steps_per_dump = g.get_unwrap<int64_t>("steps_per_dump");
    int max_steps = g.get_unwrap<int64_t>("max_steps");
    
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto ensemble_type = g.get_unwrap<std::string>("ensemble.type");
    double mu=nan, filling=nan, delta_filling=nan;
    if (ensemble_type == "grand") {
        mu = g.get_unwrap<double>("ensemble.mu");
    } else {
        filling = g.get_unwrap<double>("ensemble.filling");
        delta_filling = g.get_unwrap<double>("ensemble.delta_filling");
    }
    
    Model m = mk_model(g);
    auto init_spins = g.get_unwrap<std::string>("init_spins");
    if (init_spins == "random") {
        Lattice::set_spins_random(rng, m.spin);
    } else {
        m.lattice->set_spins(init_spins, m.spin);
    }
    
    m.set_hamiltonian(m.spin);
    
    EnergyScale es{g.get_unwrap<double>("kpm.energy_scale_lo"), g.get_unwrap<double>("kpm.energy_scale_hi")};
    // double extra = 0.1;
    // double tolerance = 1e-2;
    // auto es = energy_scale(m.H, extra, tolerance);
    
    int M = g.get_unwrap<int64_t>("kpm.cheby_order");
    int s = g.get_unwrap<int64_t>("kpm.n_vectors");
    int Mq = 4*M;
    // int M_prec = g.get_unwrap<int64_t>("kpm.cheby_order_precise");
    // int s_prec = g.get_unwrap<int64_t>("kpm.n_vectors_precise");
    
    // variables that will be updated in `build_kpm(spin)`
    auto engine = mk_engine_cx();
    Vec<double> moments(M);
    Vec<double> gamma(Mq);
    
    // Write json file for visualization
    std::ofstream json_file(base_dir + "/cfg.json");
    json_file <<
    R"(// Automatically generated for backward-compatibility with visualization tools
{
  "T": 0,
  "mu": 0,
  "order": 0,
  "order_exact": 0,
  "dt_per_rand": 0,
  "nrand": 0,
  "dumpPeriod": 0,
  "initSpin": 0,
  "model": {
)";
    json_file << "    \"type\": \"" << g.get_unwrap<std::string>("lattice.type") << "\",\n";
    json_file << "    \"w\": " << g.get_unwrap<int64_t>("lattice.w") << ",\n";
    json_file << "    \"h\": " << g.get_unwrap<int64_t>("lattice.h") << ",";
    json_file << R"(
    "t": 0,
    "t1": 0,
    "t2": 0,
    "J_H": 0,
    "B_n": 0
  }
})";
    json_file.close();
    
    auto build_kpm = [&](Vec<vec3> const& spin) {
        m.set_hamiltonian(spin);
        engine->set_H(m.H, es);
        int n = m.H.n_rows;
        if (s >= n) {
            engine->set_R_identity(n);
        } else {
            engine->set_R_uncorrelated(n, s, rng);
        }
        moments = engine->moments(M);
        gamma = moment_transform(moments, Mq);
        if (ensemble_type == "canonical") {
            mu = filling_to_mu(gamma, es, /*kB_T*/0, filling, delta_filling);
        } else {
            filling = mu_to_filling(gamma, es, kB_T, mu);
        }
        auto c = expansion_coefficients(M, Mq, std::bind(fermi_density, _1, kB_T, mu), es);
        engine->stoch_orbital(c);
    };
    
    auto calc_force = [&](Vec<vec3> const& spin, Vec<vec3>& force) {
        build_kpm(spin);
        auto D = std::bind(&Engine<cx_double>::stoch_element, engine, _1, _2);
        m.set_forces(D, force);
    };
    
    // assumes build_kpm() has already been called
    auto dump = [&](int step, double dt) {
        build_kpm(m.spin);
        double e = m.classical_potential();
        if (ensemble_type == "canonical") {
            e += electronic_energy(gamma, es, kB_T, filling, mu) / m.n_sites;
        } else {
            e += electronic_grand_energy(gamma, es, kB_T, mu) / m.n_sites;
        }
        if (!boost::filesystem::is_directory(base_dir+"/dump")) {
            boost::filesystem::create_directory(base_dir+"/dump");
        }
        std::stringstream fname;
        fname << base_dir << "/dump/dump" << std::setfill('0') << std::setw(4) << step << ".json";
        if (!overwrite_dump && boost::filesystem::exists(fname.str())) {
            cerr << "Refuse to overwrite file '" << fname.str() << "'!\n";
            std::abort();
        }
        
        cout << "Dumping file '" << fname.str() << "', time=" << step*dt << ", energy=" << e << ", n=" << filling << ", mu=" << mu << endl;
        std::ofstream dump_file(fname.str(), std::ios::trunc);
        dump_file << "{\"time\":" << step*dt <<
            ",\"action\":" << e <<
            ",\"filling\":" << filling <<
            ",\"mu\":" << mu <<
            ",\"eig\":[]" <<
            ",\"moments\":[";
        for (int i = 0; i < moments.size(); i++) {
            dump_file << moments[i];
            if (i < moments.size()-1) dump_file << ",";
        }
        dump_file << "],\"spin\":[";
        for (int i = 0; i < m.n_sites; i++) {
            dump_file << m.spin[i].x << ",";
            dump_file << m.spin[i].y << ",";
            dump_file << m.spin[i].z;
            if (i < m.n_sites-1) dump_file << ",";
        }
        dump_file << "]}";
        dump_file.close();
    };
    
    auto dynamics = mk_dynamics(g);
    dynamics->init_step(calc_force, rng, m);
    for (int step = 0; step < max_steps; step++) {
        if (step % steps_per_dump == 0) {
            dump(step, dynamics->dt);
        }
        dynamics->step(calc_force, rng, m);
    }

    return 0;
}
