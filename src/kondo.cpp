#include "kondo.h"
#include "iostream_util.h"

#include <sstream>
#include <boost/filesystem.hpp>

using namespace std::placeholders;


std::unique_ptr<Lattice> mk_lattice(cpptoml::toml_group g) {
    auto type = g.get_unwrap<std::string>("lattice.type");
    if (type == "square") {
        return SquareLattice::mk(
            g.get_unwrap<int64_t>("lattice.w"), g.get_unwrap<int64_t>("lattice.h"), g.get_unwrap<double>("lattice.t1"),
            g.get_unwrap<double>("lattice.t2", 0.0), g.get_unwrap<double>("lattice.t3", 0.0));
    } else if (type == "triangular") {
        return TriangularLattice::mk(
            g.get_unwrap<int64_t>("lattice.w"), g.get_unwrap<int64_t>("lattice.h"), g.get_unwrap<double>("lattice.t1"),
            g.get_unwrap<double>("lattice.t2", 0.0), g.get_unwrap<double>("lattice.t3", 0.0));
    } else if (type == "kagome") {
        return KagomeLattice::mk(
            g.get_unwrap<int64_t>("lattice.w"), g.get_unwrap<int64_t>("lattice.h"), g.get_unwrap<double>("lattice.t1"));
    }
    cerr << "Unsupported lattice type `" << type << "`!\n";
    std::abort();
}

Model mk_model(cpptoml::toml_group g) {
    return {
        mk_lattice(g), g.get_unwrap<double>("J"), g.get_unwrap<double>("kT"), g.get_unwrap<double>("kT_decay", 0),
        {g.get_unwrap<double>("Bx_zeeman", 0), g.get_unwrap<double>("By_zeeman", 0), g.get_unwrap<double>("Bz_zeeman", 0)},
        {g.get_unwrap<double>("current_x", 0), g.get_unwrap<double>("current_y", 0), g.get_unwrap<double>("current_z", 0)},
        g.get_unwrap<double>("current_growth", 0), g.get_unwrap<double>("current_freq", 0), g.get_unwrap<double>("easy_z", 0)
    };
}

std::unique_ptr<Dynamics> mk_dynamics(cpptoml::toml_group g) {
    double dt = g.get_unwrap<double>("dynamics.dt");
    auto type = g.get_unwrap<std::string>("dynamics.type");
    if (type == "overdamped") {
        return Dynamics::mk_overdamped(dt);
    } else if (type == "gjf") {
        return Dynamics::mk_gjf(g.get_unwrap<double>("dynamics.alpha"), dt);
    } else if (type == "sll") {
        return Dynamics::mk_sll(g.get_unwrap<double>("dynamics.alpha"), dt);
    }
    cerr << "Unsupported dynamics type `" << type << "`!\n";
    std::abort();
}




int main(int argc, char *argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <base_dir> <device #>\n";
        std::exit(EXIT_SUCCESS);
    }
    std::string base_dir(argv[1]);
    int device_num = std::stoi(argv[2]);
    
    auto input_name = base_dir + "/config.toml";
    std::ifstream input_file(input_name);
    if (!input_file.is_open()) {
        cerr << "Unable to open file `" << input_name << "`!\n";
        std::abort();
    }
    
    cout << "Using input file `" << input_name << "`\n";

    cpptoml::parser p{input_file};
    cpptoml::toml_group g = p.parse();
    
    fkpm::RNG rng(g.get_unwrap<int64_t>("seed"));
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
    auto dynamics = mk_dynamics(g);
    
    auto init_spins_type = g.get_unwrap<std::string>("init_spins.type");
    if (init_spins_type == "random") {
        Lattice::set_spins_random(rng, m.spin);
    } else {
        m.lattice->set_spins(init_spins_type, g.get_group("init_spins"), m.spin);
    }
    
    m.set_hamiltonian(m.spin);
    
    fkpm::EnergyScale es{g.get_unwrap<double>("kpm.energy_scale_lo"), g.get_unwrap<double>("kpm.energy_scale_hi")};
    // double extra = 0.1;
    // double tolerance = 1e-2;
    // auto es = energy_scale(m.H, extra, tolerance);
    
    int M                = g.get_unwrap<int64_t>("kpm.cheby_order");
    int M_prec           = g.get_unwrap<int64_t>("kpm.cheby_order_precise");
    int Mq               = 4*M;
    int Mq_prec          = 4*M_prec;
    Vec<int> groups      = m.lattice->groups(g.get_unwrap<int64_t>("kpm.n_colors"));
    Vec<int> groups_prec = m.lattice->groups(g.get_unwrap<int64_t>("kpm.n_colors_precise"));
    
    // variables that will be updated in `build_kpm(spin)`
    auto engine = fkpm::mk_engine_cuSPARSE<cx_flt>(device_num);
    Vec<double> moments;
    Vec<double> gamma;
    double energy;
    
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
    
    // assumes random vectors R have been set appropriately
    auto build_kpm = [&](Vec<vec3> const& spin, int M, int Mq) {
        m.set_hamiltonian(spin);
        engine->set_H(m.H, es);
        moments = engine->moments(M);
        gamma = fkpm::moment_transform(moments, Mq);
        energy = m.classical_potential() + dynamics->pseudo_kinetic_energy(m);
        if (ensemble_type == "canonical") {
            mu = filling_to_mu(gamma, es, m.kT(), filling, delta_filling);
            energy += electronic_energy(gamma, es, m.kT(), filling, mu);
        } else {
            filling = mu_to_filling(gamma, es, m.kT(), mu);
            energy += electronic_grand_energy(gamma, es, m.kT(), mu);
        }
        auto c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, _1, m.kT(), mu), es);
        engine->autodiff_matrix(c, m.D);
    };
    
    auto dump = [&](int step) {
        engine->set_R_correlated(groups_prec, rng);
        build_kpm(m.spin, M_prec, Mq_prec);
        double e = energy / m.n_sites;
        
        if (!boost::filesystem::is_directory(base_dir+"/dump")) {
            boost::filesystem::create_directory(base_dir+"/dump");
        }
        std::stringstream fname;
        fname << base_dir << "/dump/dump" << std::setfill('0') << std::setw(4) << step/steps_per_dump << ".json";
        if (!overwrite_dump && boost::filesystem::exists(fname.str())) {
            cerr << "Refuse to overwrite file '" << fname.str() << "'!\n";
            std::abort();
        }
        cout << "Dumping file '" << fname.str() << "', time=" << m.time << ", energy=" << e << ", n=" << filling << ", mu=" << mu << endl;
        std::ofstream dump_file(fname.str(), std::ios::trunc);
        dump_file << "{\"time\":" << m.time <<
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

    auto calc_force = [&](Vec<vec3> const& spin, Vec<vec3>& force) {
        build_kpm(spin, M, Mq);
        m.set_forces(m.D, force);
    };
    
    engine->set_R_correlated(groups, rng);
    dynamics->init(calc_force, rng, m);
    
    dump(dynamics->n_steps);
    while (dynamics->n_steps < max_steps) {
        engine->set_R_correlated(groups, rng);
        dynamics->step(calc_force, rng, m);
        if (dynamics->n_steps % steps_per_dump == 0) {
            dump(dynamics->n_steps);
        }
    }
    
    return 0;
}
