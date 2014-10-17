#include "kondo.h"
#include "cpptoml.h"
#include "iostream_util.h"

#include <sstream>
#include <boost/filesystem.hpp>

using namespace std::placeholders;


std::string default_input = R"(
J = 0.5
kB_T = 0.0
B_zeeman = 0.0
B_orbital_idx = 0
time_per_dump = 10.0
init_spins = "ferro"

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
dt = 0.2

[lattice]
type = "square"
w = 6
h = 6
t1 = -1.0
)";


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
    return {mk_lattice(g), g.get_unwrap<double>("J")};
}

std::unique_ptr<Dynamics> mk_dynamics(cpptoml::toml_group g) {
    double kB_T = g.get_unwrap<double>("kB_T");
    double dt = g.get_unwrap<double>("dynamics.dt");
    auto type = g.get_unwrap<std::string>("dynamics.type");
    if (type == "overdamped") {
        return Dynamics::mk_overdamped(kB_T, dt);
    } else if (type == "gjf") {
        return Dynamics::mk_gjf(g.get_unwrap<double>("alpha"), kB_T, dt);
    } else if (type == "sll") {
        return Dynamics::mk_sll(g.get_unwrap<double>("alpha"), kB_T, dt);
    }
    cerr << "Unsupported dynamics type `" << type << "`!\n";
    std::abort();
}




int main(int argc, char *argv[]) {
    std::unique_ptr<std::istream> input;
    if (argc == 1) {
        cout << "Using default input parameters.\n";
        input = std::make_unique<std::stringstream>(default_input);
    }
    else if (argc == 2) {
        auto file = std::make_unique<std::ifstream>(argv[1]);
        if (file->is_open()) {
            cout << "Using input file `" << argv[1] << "`\n";
            input = std::move(file);
        }
        else {
            cerr << "Unable to open file `" << argv[1] << "`!\n";
            std::abort();
        }
    }
    else {
        cout << "Usage: " << argv[0] << " <config.toml>\n";
        std::exit(EXIT_SUCCESS);
    }
    
    cpptoml::parser p{*input};
    cpptoml::toml_group g = p.parse();
    
    RNG rng(g.get_unwrap<int64_t>("seed"));
    double kB_T = g.get_unwrap<double>("kB_T");
    double time_per_dump = g.get_unwrap<double>("time_per_dump");
    double max_time = g.get_unwrap<double>("max_time");
    
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto ensemble_type = g.get_unwrap<std::string>("ensemble.type");
    auto mu = g.get_unwrap<double>("ensemble.mu", nan);
    auto filling = g.get_unwrap<double>("ensemble.filling", nan);
    auto delta_filling = g.get_unwrap<double>("ensemble.delta_filling", nan);
    
    Model m = mk_model(g);
    m.lattice->set_spins(g.get_unwrap<std::string>("init_spins"), m.spin);
    
    m.set_hamiltonian(m.spin);
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m.H, extra, tolerance);
    
    int M = g.get_unwrap<int64_t>("kpm.cheby_order");
    int s = g.get_unwrap<int64_t>("kpm.n_vectors");
    int Mq = 4*M;
    // int M_prec = g.get_unwrap<int64_t>("kpm.cheby_order_precise");
    // int s_prec = g.get_unwrap<int64_t>("kpm.n_vectors_precise");
    
    // variables that will be updated in `build_kpm(spin)`
    auto engine = mk_engine_cx();
    Vec<double> moments(M);
    Vec<double> gamma(Mq);
    
    auto build_kpm = [&](Vec<vec3> const& spin) {
        m.set_hamiltonian(spin);
        engine->set_H(m.H, es);
        int n = m.H.n_rows;
        if (s >= n)
            engine->set_R_identity(n);
        else
            engine->set_R_uncorrelated(n, s, rng);
        moments = engine->moments(M);
        gamma = moment_transform(moments, Mq);
        if (ensemble_type == "canonical") {
            mu = filling_to_mu(gamma, es, /*kB_T*/0, filling, delta_filling);
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
        double e;
        if (ensemble_type == "canonical") {
            e = electronic_energy(gamma, es, kB_T, filling, mu) / m.n_sites;
        } else {
            e = electronic_grand_energy(gamma, es, kB_T, mu) / m.n_sites;
        }
        if (!boost::filesystem::is_directory("dump/")) {
            boost::filesystem::create_directory("dump/");
        }
        std::stringstream fname;
        fname << "dump/dump" << std::setfill('0') << std::setw(4) << step << ".json";
        if (boost::filesystem::exists(fname.str())) {
            cerr << "Refuse to overwrite file '" << fname.str() << "'!\n";
            std::abort();
        }
        
        cout << "Dumping file '" << fname.str() << "', time=" << step*dt << ", energy=" << e << endl;
        std::ofstream dump_file(fname.str());
        dump_file << "Writing this to a file.\n";
        dump_file.close();
    };
    
    auto dynamics = mk_dynamics(g);
    dynamics->init_step(calc_force, rng, m);
    for (int step = 0; step*dynamics->dt < max_time; step++) {
        int steps_per_dump = int(time_per_dump/dynamics->dt + 0.5);
        if (step % steps_per_dump == 0) {
            dump(step, dynamics->dt);
        }
        dynamics->step(calc_force, rng, m);
    }
    
    build_kpm(m.spin);

    return 0;
}
