#include "kondo.h"
#include "iostream_util.h"

#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
//using boost::property_tree::write_json;

template <typename T>
std::vector<T> as_vector(ptree const& pt, ptree::key_type const& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}




void triangular(int argc, char *argv[]) {
    auto engine = fkpm::mk_engine<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);
    if (argc != 6) {
        cout << "Usage: " << argv[0] << " <base_dir>  dumpXXXX.json  M  colors  seed\n";
        std::exit(EXIT_SUCCESS);
    }
    
    std::string base_dir(argv[1]);
    auto toml_name = base_dir + "/config.toml";
    std::ifstream toml_file(toml_name);
    if (!toml_file.is_open()) {
        cerr << "Unable to open file `" << toml_name << "`!\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "using toml file `" << toml_name << "`!\n";
    cpptoml::parser p{toml_file};
    cpptoml::toml_group g = p.parse();
    
    auto m = SimpleModel::mk_triangular(g.get_unwrap<int64_t>("model.w"), g.get_unwrap<int64_t>("model.h"));
    m->J  = g.get_unwrap<double>("model.J");
    m->t1 = g.get_unwrap<double>("model.t1", 0);
    m->t2 = g.get_unwrap<double>("model.t2", 0);
    m->t3 = g.get_unwrap<double>("model.t3", 0);
    m->kT_init  = g.get_unwrap<double>("model.kT");
    //std::cout << "using T=0, change back later!" << std::endl;
    //m->kT_init = 0.0;
    m->kT_decay = g.get_unwrap<double>("model.kT_decay", 0);
    m->B_zeeman = {g.get_unwrap<double>("model.zeeman_x", 0), g.get_unwrap<double>("model.zeeman_y", 0), g.get_unwrap<double>("model.zeeman_z", 0)};
    m->easy_z   = g.get_unwrap<double>("model.easy_z", 0);
    m->s1       = g.get_unwrap<double>("model.s1", 0);
    m->s2       = g.get_unwrap<double>("model.s2", 0);
    m->s3       = g.get_unwrap<double>("model.s3", 0);
    
    std::string json_name(argv[2]);
    json_name = base_dir + "/dump/" + json_name;
    std::ifstream json_file(json_name);
    if (!json_file.is_open()) {
        cerr << "Unable to open file `" << json_name << "`!\n";
        std::exit(EXIT_FAILURE);
    }
    cout << "Using json file `" << json_name << "`!\n";
    ptree pt_json;        // used for reading the json file
    std::string json_contents;
    std::getline(json_file, json_contents);
    std::istringstream is (json_contents);
    read_json (is, pt_json);
    auto time    = pt_json.get<int>    ("time");
    auto action  = pt_json.get<double> ("action");
    auto filling = pt_json.get<double> ("filling");
    auto mu      = pt_json.get<double> ("mu");
    auto spin    = as_vector<double>(pt_json, "spin");
    
    std::cout << "lattice: " << g.get_unwrap<std::string>("model.lattice") << std::endl;
    std::cout << "time:    " << time << std::endl;
    std::cout << "action:  " << action << std::endl;
    std::cout << "filling: " << filling << std::endl;
    std::cout << "mu:      " << mu << std::endl;
    std::cout << "J:       " << m->J << std::endl;
    std::cout << "t:       " << m->t1 << std::endl;
    std::cout << "kT:      " << m->kT() << std::endl;
    
    
    assert(spin.size() == m->n_sites * 3);              // build spin configuration from dump file
    for (int i = 0; i < m->n_sites; i++) {
        m->spin[i] = vec3(spin[3*i],spin[3*i+1],spin[3*i+2]);
    }
    m->set_hamiltonian(m->spin);
    
    //fkpm::EnergyScale es{g.get_unwrap<double>("kpm.energy_scale_lo"), g.get_unwrap<double>("kpm.energy_scale_hi")};
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = energy_scale(m->H, extra, tolerance);
    
    
    std::stringstream convert_M(argv[3]);
    int M;
    assert(convert_M >> M);
    std::cout << "M:       " << M << std::endl;
    int Mq = 2*M;
    std::cout << "Mq:      " << Mq << std::endl;
    std::stringstream convert_colors(argv[4]);
    int n_colors;
    assert(convert_colors >> n_colors);
    m->groups(n_colors);
    std::cout << "colors:  " << n_colors << std::endl;
    std::stringstream convert_seed(argv[5]);
    int seed;
    assert(convert_seed >> seed);
    std::cout << "seed:    " << seed << std::endl;
    
    fkpm::RNG rng(seed);
    //engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    engine->set_R_correlated(m->groups(n_colors), rng);
    auto kernel = fkpm::jackson_kernel(M);
    engine->set_H(m->H, es);
    
    double area = m->n_sites*sqrt(3.0)/2.0;
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    jx.scale(1/sqrt(area));
    jy.scale(1/sqrt(area));
    
    toml_file.close();
    json_file.close();
    
    auto mu_dos = engine->moments(M);
    auto gamma = fkpm::moment_transform(mu_dos, Mq);
    fkpm::Vec<double> mu_list, rho;
    fkpm::density_function(gamma, es, mu_list, rho);
    for (int i = 0; i < rho.size(); i++) {
        rho[i]/=m->n_sites;
    }
    
    std::cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    auto mu_xy = engine->moments2_v1(M, jx, jy);
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    cout << "calculating xy conductivities... " << std::flush;
    std::ofstream fout2("full_time"+std::to_string(time)+"_M"+std::to_string(M)+
                        "_color"+std::to_string(n_colors)+"_seed"+std::to_string(seed)+".dat", std::ios::out);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20) << "mu"
          << std::setw(20) << "rho" << std::setw(20) << "sigma_xy" << std::endl;
    arma::Col<double> sigma_xy(Mq);
    sigma_xy.zeros();
    for (int i = 0; i < Mq; i++) {
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu_list[i], 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/m->kT() << std::setw(20) << mu_list[i]
              << std::setw(20) << rho[i] << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    double mu_extra=-5.0;
    std::ifstream fin0("quarter_time"+std::to_string(time)+".dat");
    if (! fin0.good()) {
        fin0.close();
        std::ofstream fout3("quarter_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
        fout3 << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)"
              << std::setw(20) << "(4)" << std::setw(20) << "(5)" << std::setw(20) << "(6)"
              << std::setw(20) << "(7)" << std::setw(20) << "(8)" << std::endl;
        fout3 << std::setw(20) << "#M" << std::setw(20) << "mu" << std::setw(20)
              << "mu_extra" << std::setw(20) << "colors" << std::setw(20) << "seed"
              << std::setw(20) << "F" << std::setw(20) << "sigma_xy"
              << std::setw(20) << "sigma_xy_extra" << std::endl;
        fout3.close();
    }
    std::ofstream fout3("quarter_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
    fout3 << std::scientific << std::right;
    auto cmn       = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
    auto cmn_extra = electrical_conductivity_coefficients(M, Mq, m->kT(), mu_extra, 0.0, es, kernel);
    fout3 << std::setw(20) << M << std::setw(20) << mu << std::setw(20) << mu_extra
          << std::setw(20) << n_colors << std::setw(20) << seed
          << std::setw(20) << fkpm::electronic_energy(gamma, es, m->kT(), filling, mu)/m->n_sites
          << std::setw(20) << std::real(fkpm::moment_product(cmn, mu_xy))
          << std::setw(20) << std::real(fkpm::moment_product(cmn_extra, mu_xy)) << std::endl;
    fout3.close();
    
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    triangular(argc, argv);
}