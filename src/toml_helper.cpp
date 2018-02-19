#include "cpptoml.h"
#include "iostream_util.h"


typedef std::shared_ptr<cpptoml::table> toml_ptr;

toml_ptr toml_from_str(std::string const& str) {
    std::istringstream is(str);
    return cpptoml::parser(is).parse();
}

toml_ptr toml_from_file(std::string const& filename) {
    return cpptoml::parse_file(filename);
}


// Convert a TOML template parameter into a string
template <typename T> std::string toml_type_name();
template <> inline std::string toml_type_name<int64_t>()     { return "int64_t"; }
template <> inline std::string toml_type_name<double>()      { return "double"; }
template <> inline std::string toml_type_name<bool>()        { return "bool"; }
template <> inline std::string toml_type_name<std::string>() { return "std::string"; }


// Attempt to get table value with type corresponding to the template parameter.
// If not found, will print an error and exit.
template <class T>
T toml_get(const toml_ptr table, const std::string& key)
{
    auto v = table->get_qualified_as<T>(key);
    if (v) {
        return *v;
    }
    std::cerr << "Could not find key '" << key << "' of type '" << toml_type_name<T>() << "' in TOML group:\n{\n" << *table << "}" << std::endl;
    std::exit(1);
}
// Specific template instantations
template int64_t        toml_get<int64_t>       (const toml_ptr table, const std::string& key);
template double         toml_get<double>        (const toml_ptr table, const std::string& key);
template bool           toml_get<bool>          (const toml_ptr table, const std::string& key);
template std::string    toml_get<std::string>   (const toml_ptr table, const std::string& key);
// Separate logic for returning another TOML table
template <>
toml_ptr toml_get<toml_ptr>(const toml_ptr table, const std::string& key) {
    return table->get_table_qualified(key);
}


// Attempt to get table value with type corresponding to the template parameter.
// If not found, will return default value.
template <class T>
T toml_get(const toml_ptr table, const std::string& key, T default_value)
{
    auto v = table->get_qualified_as<T>(key);
    if (v) {
        return *v;
    }
    else {
        try {
            if (table->get_qualified(key)) {
                std::cerr << "Key '" << key << "' does not have expected type of '" << toml_type_name<T>() << "' in TOML group:\n{\n" << *table << "}" << std::endl;
                std::exit(1);
            }
        }
        catch (const std::out_of_range&) {}
        return default_value;
    }
}
template int64_t        toml_get<int64_t>       (const toml_ptr table, const std::string& key, int64_t);
template double         toml_get<double>        (const toml_ptr table, const std::string& key, double);
template bool           toml_get<bool>          (const toml_ptr table, const std::string& key, bool);
template std::string    toml_get<std::string>   (const toml_ptr table, const std::string& key, std::string);

