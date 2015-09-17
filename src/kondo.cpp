#include "kondo.h"
#include "iostream_util.h"

#include <climits>
#include <cstdint>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


// workaround missing "is_trivially_copyable" in g++ < 5.0
#if __GNUG__ && __GNUC__ < 5
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif
static const std::string nibble2hex = "0123456789ABCDEF";
int hex2nibble(char c) {
    for (int i = 0; i < 16; i++) {
        if (c == nibble2hex[i]) return i;
    }
    assert(false);
}
template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
void serialize_to_hex(T const& data, std::ostream &os) {
    uint8_t *bytes = (uint8_t *)&data;
    for (size_t i = 0; i < sizeof(T)*CHAR_BIT/8; i++) {
        os << nibble2hex[(bytes[i] >> 4) & 0xF];
        os << nibble2hex[bytes[i] & 0xF];
    }
}
template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
void deserialize_from_hex(std::istream& is, T& data) {
    uint8_t *bytes = (uint8_t *)&data;
    for (int i = 0; i < sizeof(T)*CHAR_BIT/8; i++) {
        char c1, c2;
        is >> c1 >> c2;
        bytes[i] = (hex2nibble(c1) << 4) | hex2nibble(c2);
    }
}


std::unique_ptr<Model> mk_model(cpptoml::toml_group g) {
    std::unique_ptr<Model> ret;
    auto type = g.get_unwrap<std::string>("model.type");
    if (type == "simple") {
        auto lattice = g.get_unwrap<std::string>("model.lattice");
        std::unique_ptr<SimpleModel> m;
        if (lattice == "linear") {
            m = SimpleModel::mk_linear(g.get_unwrap<int64_t>("model.w"));
        } else if (lattice == "square") {
            m = SimpleModel::mk_square(g.get_unwrap<int64_t>("model.w"), g.get_unwrap<int64_t>("model.h"));
        } else if (lattice == "triangular") {
            m = SimpleModel::mk_triangular(g.get_unwrap<int64_t>("model.w"), g.get_unwrap<int64_t>("model.h"));
        } else if (lattice == "kagome") {
            m = SimpleModel::mk_kagome(g.get_unwrap<int64_t>("model.w"), g.get_unwrap<int64_t>("model.h"));
        } else if (lattice == "cubic") {
            m = SimpleModel::mk_cubic(g.get_unwrap<int64_t>("model.lx"), g.get_unwrap<int64_t>("model.ly"), g.get_unwrap<int64_t>("model.lz"));
        } else {
            std::cerr << "Simple model lattice '" << lattice << "' not supported.\n";
            std::exit(EXIT_FAILURE);
        }
        m->J  = g.get_unwrap<double>("model.J");
        m->t1 = g.get_unwrap<double>("model.t1", 0);
        m->t2 = g.get_unwrap<double>("model.t2", 0);
        m->t3 = g.get_unwrap<double>("model.t3", 0);
        ret = std::move(m);
    } else if (type == "mostovoy") {
        auto lattice = g.get_unwrap<std::string>("model.lattice");
        if (lattice != "cubic") {
            std::cerr << "Mostovoy model requires `lattice = \"cubic\"`\n";
            std::exit(EXIT_FAILURE);
        }
        auto m = std::make_unique<MostovoyModel>(g.get_unwrap<int64_t>("model.lx"),
                                                 g.get_unwrap<int64_t>("model.ly"),
                                                 g.get_unwrap<int64_t>("model.lz"));
        m->J     = g.get_unwrap<double>("model.J");
        m->t_pds = g.get_unwrap<double>("model.t_pds");
        m->t_pp  = g.get_unwrap<double>("model.t_pp");
        m->delta = g.get_unwrap<double>("model.delta");
        ret = std::move(m);
    } else {
        std::cerr << "Model type '" << type << "' not supported.\n";
        std::exit(EXIT_FAILURE);
    }
    ret->kT_init  = g.get_unwrap<double>("model.kT");
    ret->kT_decay = g.get_unwrap<double>("model.kT_decay", 0);
    ret->zeeman   = {g.get_unwrap<double>("model.zeeman_x",  0), g.get_unwrap<double>("model.zeeman_y",  0), g.get_unwrap<double>("model.zeeman_z",  0)};
    ret->current  = {g.get_unwrap<double>("model.current_x", 0), g.get_unwrap<double>("model.current_y", 0), g.get_unwrap<double>("model.current_z", 0)};
    ret->easy_z   = g.get_unwrap<double>("model.easy_z", 0);
    ret->s0       = g.get_unwrap<double>("model.s0", 0);
    ret->s1       = g.get_unwrap<double>("model.s1", 0);
    ret->s2       = g.get_unwrap<double>("model.s2", 0);
    ret->s3       = g.get_unwrap<double>("model.s3", 0);
    return ret;
}

std::unique_ptr<Dynamics> mk_dynamics(cpptoml::toml_group g) {
    double dt = g.get_unwrap<double>("dynamics.dt");
    auto type = g.get_unwrap<std::string>("dynamics.type");
    if (type == "overdamped") {
        return Dynamics::mk_overdamped(dt);
    } else if (type == "sll") {
        return Dynamics::mk_sll(g.get_unwrap<double>("dynamics.alpha"), dt);
    } else if (type == "sll_sib") {
        return Dynamics::mk_sll_sib(g.get_unwrap<double>("dynamics.alpha"), dt);
    } else if (type == "gjf") {
        return Dynamics::mk_gjf(g.get_unwrap<double>("dynamics.alpha"), dt);
    }
    
    cerr << "Unsupported dynamics type `" << type << "`!\n";
    std::exit(EXIT_FAILURE);
}

void read_state_from_dump(boost::filesystem::path const& dump_dir, fkpm::RNG &rng, Model &m, Dynamics &dynamics) {
    Vec<boost::filesystem::path> v;
    boost::filesystem::path p(dump_dir);
    std::copy(boost::filesystem::directory_iterator(p),
              boost::filesystem::directory_iterator(),
              std::back_inserter(v));
    std::sort(v.begin(), v.end());
    boost::filesystem::ifstream is(v.back());
    boost::property_tree::ptree pt;
    boost::property_tree::read_json (is, pt);
    
    std::cout << "RESUMING from " << v.back() << "!\n";
    
    // Read random state
    std::string rng_state = pt.get<std::string>("rng_state");
    std::stringstream ss(rng_state);
    deserialize_from_hex(ss, rng);
    
    // Read spin configuration
    Vec<double> data;
    for (auto& elem : pt.get_child("spin")) {
        assert(elem.first.empty()); // array elements have no names
        data.push_back(std::stod(elem.second.data()));
    }
    assert(data.size() == 3*m.n_sites);
    for (int i = 0; i < m.n_sites; i++) {
        m.spin[i] = {data[3*i+0], data[3*i+1], data[3*i+2]};
    }
    
    // Read n_steps
    dynamics.n_steps = pt.get<size_t>("n_steps");
    
    // TODO: load velocity in GJ-F dynamics... ?
}

int main(int argc, char *argv[]) {
    auto engine = fkpm::mk_engine_mpi<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <base_dir>\n";
        std::exit(EXIT_SUCCESS);
    }
    boost::filesystem::path base_dir(argv[1]);
    
    auto input_name = base_dir / "config.toml";
    boost::filesystem::ifstream input_file(input_name);
    if (!input_file.is_open()) {
        cerr << "Unable to open file " << input_name << "!\n";
        std::exit(EXIT_FAILURE);
    }
    
    cout << "Using input file " << input_name << ".\n";
    
    cpptoml::parser p{input_file};
    cpptoml::toml_group g = p.parse();
    
    fkpm::RNG rng(g.get_unwrap<int64_t>("random_seed"));
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
    
    auto m = mk_model(g);
    auto dynamics = mk_dynamics(g);
    
    fkpm::EnergyScale es{g.get_unwrap<double>("kpm.energy_scale_lo"), g.get_unwrap<double>("kpm.energy_scale_hi")};
    // double extra = 0.1;
    // double tolerance = 1e-2;
    // auto es = energy_scale(m.H, extra, tolerance);
    
    int M                = g.get_unwrap<int64_t>("kpm.cheby_order");
    int M_prec           = g.get_unwrap<int64_t>("kpm.cheby_order_precise");
    int Mq               = 4*M;
    int Mq_prec          = 4*M_prec;
    Vec<int> groups      = m->groups(g.get_unwrap<int64_t>("kpm.n_colors"));
    Vec<int> groups_prec = m->groups(g.get_unwrap<int64_t>("kpm.n_colors_precise"));
    
    // variables that will be updated in `build_kpm(spin)`
    Vec<double> moments;
    Vec<double> gamma;
    double energy;
    
    // assumes random vectors R have been set appropriately
    auto build_kpm = [&](Vec<vec3> const& spin, int M, int Mq) {
        m->set_hamiltonian(spin);
        engine->set_H(m->H, es);
        moments = engine->moments(M);
        gamma = fkpm::moment_transform(moments, Mq);
        energy = m->energy_classical(m->spin) + dynamics->pseudo_kinetic_energy(*m);
        if (ensemble_type == "canonical") {
            mu = filling_to_mu(gamma, es, m->kT(), filling, delta_filling);
            energy += electronic_energy(gamma, es, m->kT(), filling, mu);
        } else {
            filling = mu_to_filling(gamma, es, m->kT(), mu);
            energy += electronic_grand_energy(gamma, es, m->kT(), mu);
        }
        auto c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, std::placeholders::_1, m->kT(), mu), es);
        engine->autodiff_matrix(c, m->D);
    };
    
    auto calc_force = [&](Vec<vec3> const& spin, Vec<vec3>& force) {
        build_kpm(spin, M, Mq);
        m->set_forces(m->D, spin, force);
    };
    
    auto dump = [&](int step) {
        engine->set_R_correlated(groups_prec, rng);
        build_kpm(m->spin, M_prec, Mq_prec);
        double e = energy / m->n_sites;
        
        std::stringstream fname;
        fname << "dump" << std::setfill('0') << std::setw(4) << step/steps_per_dump << ".json";
        cout << "Dumping file " << base_dir/"dump"/fname.str() << ", time=" << m->time << ", energy=" << e << ", n=" << filling << ", mu=" << mu << endl;
        boost::filesystem::ofstream dump_file(base_dir/"dump"/fname.str());
        dump_file << "{\n" <<
        "\"n_steps\":" << dynamics->n_steps << ",\n" <<
        "\"time\":" << m->time << ",\n" <<
        "\"action\":" << e << ",\n" <<
        "\"filling\":" << filling << ",\n" <<
        "\"mu\":" << mu << ",\n" <<
        "\"eig\":[]" << ",\n" <<
        "\"moments\":[";
        for (int i = 0; i < moments.size(); i++) {
            dump_file << moments[i];
            if (i < moments.size()-1) dump_file << ",";
        }
        dump_file << "],\n";
        dump_file << "\"rng_state\":\"";
        serialize_to_hex(rng, dump_file);
        dump_file << "\",\n";
        dump_file << "\"spin\":[";
        for (int i = 0; i < m->n_sites; i++) {
            dump_file << m->spin[i].x << ",";
            dump_file << m->spin[i].y << ",";
            dump_file << m->spin[i].z;
            if (i < m->n_sites-1) dump_file << ",";
        }
        dump_file << "]\n" <<
        "}\n";
        dump_file.close();
    };
    
    auto write_json_vis = [&]() {
        boost::filesystem::ofstream json_file(base_dir/"cfg.json");
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
                auto lattice = g.get_unwrap<std::string>("model.lattice");
                json_file << "    \"type\": \"" << lattice << "\",\n";
                if (lattice == "cubic") {
                    json_file << "    \"w\": " << g.get_unwrap<int64_t>("model.lx") << ",\n";
                    json_file << "    \"h\": " << g.get_unwrap<int64_t>("model.ly")*g.get_unwrap<int64_t>("model.lz") << ",";
                }
                else {
                    json_file << "    \"w\": " << g.get_unwrap<int64_t>("model.w") << ",\n";
                    json_file << "    \"h\": " << g.get_unwrap<int64_t>("model.h") << ",";
                }
                json_file << R"(
                "t": 0,
                "t1": 0,
                "t2": 0,
                "J_H": 0,
                "B_n": 0
            }
        })";
        json_file.close();
    };
    
    if (!boost::filesystem::is_directory(base_dir/"dump")) {
        boost::filesystem::create_directory(base_dir/"dump");
    }
    boost::filesystem::directory_iterator it(base_dir/"dump"), end_it;
    if(it != end_it) {
        // non-empty dump directory -- resume previous simulation
        read_state_from_dump(base_dir/"dump", rng, *m, *dynamics);
    }
    else {
        // empty dump directory -- start from scratch
        auto init_spins_type = g.get_unwrap<std::string>("init_spins.type");
        if (init_spins_type == "random") {
            m->set_spins_random(rng, m->spin);
        } else {
            m->set_spins(init_spins_type, *g.get_group("init_spins"), m->spin);
        }
        
        write_json_vis();
        dump(dynamics->n_steps);
        
        engine->set_R_correlated(groups, rng);
        dynamics->init(calc_force, rng, *m);
    }
    
    while (dynamics->n_steps < max_steps) {
        engine->set_R_correlated(groups, rng);
        dynamics->step(calc_force, rng, *m);
        if (dynamics->n_steps % steps_per_dump == 0) {
            dump(dynamics->n_steps);
        }
    }
    
    return EXIT_SUCCESS;
}
