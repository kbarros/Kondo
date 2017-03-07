#include "kondo.h"
#include "iostream_util.h"
#include <cstdio>
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
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <base_dir>\n";
        std::exit(EXIT_SUCCESS);
    }
    
    std::string base_dir(argv[1]);
    auto input_name = base_dir + "/config.toml";
    std::cout << "using toml file `" << input_name << "`!\n";
    toml_ptr g = toml_from_file(input_name);
    
    auto m = SimpleModel::mk_triangular(toml_get<int64_t>(g, "model.w"), toml_get<int64_t>(g, "model.h"));
    m->J  = toml_get<double>(g, "model.J");
    m->t1 = toml_get(g, "model.t1", 0.0);
    m->t2 = toml_get(g, "model.t2", 0.0);
    m->t3 = toml_get(g, "model.t3", 0.0);
    m->kT_init  = toml_get<double>(g, "model.kT");
    m->kT_decay = toml_get(g, "model.kT_decay", 0.0);
    m->zeeman   = {toml_get(g, "model.zeeman_x", 0.0), toml_get(g, "model.zeeman_y", 0.0), toml_get(g, "model.zeeman_z", 0.0)};
    m->easy_z   = toml_get(g, "model.easy_z", 0.0);
    m->s1       = toml_get(g, "model.s1", 0.0);
    m->s2       = toml_get(g, "model.s2", 0.0);
    m->s3       = toml_get(g, "model.s3", 0.0);
    
    std::string json_name;
    std::cout << "Input dumpfile name:" << std::endl;
    std::cin >> json_name;
    std::cout << json_name << std::endl;
    json_name = base_dir + "/dump/" + json_name;
    std::ifstream json_file(json_name);
    if (!json_file.is_open()) {
        cerr << "Unable to open file `" << json_name << "`!\n";
        std::exit(EXIT_FAILURE);
    }
    cout << "Using json file `" << json_name << "`!\n";
    ptree pt_json;        // used for reading the json file
    std::string json_eachline, json_contents;
    while (std::getline(json_file, json_eachline)) {
        json_contents += json_eachline;
    }
    std::istringstream is (json_contents);
    read_json (is, pt_json);
    auto time    = pt_json.get<int>    ("time");
    auto action  = pt_json.get<double> ("action");
    auto filling = pt_json.get<double> ("filling");
    auto mu      = pt_json.get<double> ("mu");
    auto spin    = as_vector<double>(pt_json, "spin");
    
    std::cout << "lattice: " << toml_get<std::string>(g, "model.lattice") << std::endl;
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
    
    int purpose, M, Mq, use_correlated, n_colors, seed;
    std::cout << "Inut 0/1/2 for calculating sigma_xx/sigma_xy/both: " << std::endl;
    std::cin >> purpose;
    fkpm::EnergyScale es;
    es = engine->energy_scale(m->H, 0.1);
    std::cout << "energyscale: [" << es.lo << ", " << es.hi << "]" << std::endl;
    std::cout << "Input M:" << std::endl;
    std::cin >> M;
    std::cout << M << std::endl;
    std::cout << "Input Mq:" << std::endl;
    std::cin >> Mq;
    std::cout << Mq << std::endl;
    std::cout << "Input 0/1 for using uncorrelated/correlated random numbers:" << std::endl;
    std::cin >> use_correlated;
    std::cout << use_correlated << std::endl;
    std::cout << "Input n_colors:" << std::endl;
    std::cin >> n_colors;
    std::cout << n_colors << std::endl;
    std::cout << "Input seed:" << std::endl;
    std::cin >> seed;
    std::cout << seed << std::endl;
    
    fkpm::RNG rng(seed);
    if (use_correlated) {
        engine->set_R_correlated(m->groups(n_colors), rng);
    } else {
        engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    }
    
    auto kernel = fkpm::jackson_kernel(M);
    engine->set_H(m->H, es);
    
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    
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
    Vec<Vec<fkpm::cx_double>> mu_xx, mu_xy;
    if (purpose == 0) {
        mu_xx = engine->moments2_v1(M, jx, jx, 10, 16);
        mu_xy = mu_xx; // invalid
    } else if (purpose == 1){
        mu_xy = engine->moments2_v1(M, jx, jy, 10, 16);
        mu_xx = mu_xy; // invalid
    } else {
        mu_xx = engine->moments2_v1(M, jx, jx, 10, 16);
        mu_xy = engine->moments2_v1(M, jx, jy, 10, 16);
    }
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    cout << "calculating dc conductivities... " << std::flush;
    std::ofstream fout2("full_time"+std::to_string(time)+"_M"+std::to_string(M)+"_corr"+std::to_string(use_correlated)+
                        "_color"+std::to_string(n_colors)+"_seed"+std::to_string(seed)+".dat", std::ios::out);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)"
          << std::setw(20) << "(4)" << std::setw(20) << "(5)" << std::setw(20) << "(6)" << std::endl;
    fout2 << std::setw(20) << "M" << std::setw(20) << "kT" << std::setw(20) << "mu"
          << std::setw(20) << "rho" << std::setw(20) << "sigma_xx" << std::setw(20) << "sigma_xy" << std::endl;
    int interval = std::max(Mq/400,5);
    for (int i = 0; i < Mq; i+=interval) {
        auto cmn = electrical_conductivity_coefficients_v2(M, Mq, m->kT(), mu_list[i], 0.0, es, kernel);
        auto sigma_xx = std::real(fkpm::moment_product(cmn, mu_xx));
        auto sigma_xy = std::real(fkpm::moment_product(cmn, mu_xy));
        fout2 << std::setw(20) << M << std::setw(20) << m->kT() << std::setw(20) << mu_list[i]
              << std::setw(20) << rho[i] << std::setw(20) << sigma_xx << std::setw(20) << sigma_xy << std::endl;
    }
    fout2.close();
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    std::ifstream fin0("result_time"+std::to_string(time)+".dat");
    if (! fin0.good()) {
        fin0.close();
        std::ofstream fout3("result_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
        fout3 << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)"
              << std::setw(20) << "(4)" << std::setw(20) << "(5)" << std::setw(20) << "(6)"
              << std::setw(20) << "(7)" << std::setw(20) << "(8)" << std::endl;
        fout3 << std::setw(20) << "M" << std::setw(20) << "correlated" << std::setw(20) << "colors"
              << std::setw(20) << "seed" << std::setw(20) << "F" << std::setw(20) << "mu"
              << std::setw(20) << "sigma_xx" << std::setw(20) << "sigma_xy" << std::endl;
        fout3.close();
    }
    std::ofstream fout3("result_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
    fout3 << std::scientific << std::right;
    auto cmn       = electrical_conductivity_coefficients_v2(M, Mq, m->kT(), mu, 0.0, es, kernel);

    fout3 << std::setw(20) << M << std::setw(20) << use_correlated << std::setw(20) << n_colors
          << std::setw(20) << seed
          << std::setw(20) << fkpm::electronic_energy(gamma, es, m->kT(), filling, mu)/m->n_sites
          << std::setw(20) << mu
          << std::setw(20) << std::real(fkpm::moment_product(cmn, mu_xx))
          << std::setw(20) << std::real(fkpm::moment_product(cmn, mu_xy))
          << std::endl;
    fout3.close();
    std::cout << std::endl;
    std::remove("dump_device0.dat");
    std::remove("dump_device1.dat");
}



int main(int argc, char *argv[]) {
    triangular(argc, argv);
}
