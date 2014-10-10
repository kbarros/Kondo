//
//  vec3.cpp
//  tibidy
//
//  Created by Kipton Barros on 6/19/14.
//
//

#include <cmath>

template<typename T>
class Vec3 {
public:
    T x=0, y=0, z=0;
    
    Vec3(T x, T y, T z): x(x), y(y), z(z) {}
    
    Vec3<T> operator-() const {
        return {-x, -y, -z};
    }
    
    Vec3<T> operator+(Vec3<T> that) const {
        return {x+that.x, y+that.y, z+that.z};
    }

    Vec3<T> operator-(Vec3<T> that) const {
        return {x-that.x, y-that.y, z-that.z};
    }

    Vec3<T> operator*(T a) const {
        return {x*a, y*a, z*a};
    }
    
    friend Vec3<T> operator *(T x, Vec3<T> y) {
        return y*x;
    }
    
    Vec3<T> operator/(T a) const {
        return {x/a, y/a, z/a};
    }
    
    T dot(Vec3<T> const& that) const {
        return x*std::conj(that.x) + y*std::conj(that.y) + z*std::conj(that.z);
    }
    
    auto norm() const -> decltype(std::norm(x)) {
        return sqrt(std::norm(x) + std::norm(y) + std::norm(z));
    }
    
    Vec3<T> normalize() const {
        return *this / norm();
    }
    
    void operator+=(Vec3<T> that) {
        x += that.x;
        y += that.y;
        z += that.z;
    }
    
    void operator-=(Vec3<T> that) {
        x -= that.x;
        y -= that.y;
        z -= that.z;
    }
    
    void operator*=(T a) {
        x *= a;
        y *= a;
        z *= a;
    }
    
    void operator/=(T a) {
        x /= a;
        y /= a;
        z /= a;
    }
    
    template <typename S>
    operator Vec3<S>() const {
        return Vec3<S>(x, y, z);
    }
    
    template <typename S>
    friend Vec3<S> real(Vec3<std::complex<S>> v) {
        return {std::real(v.x), std::real(v.y), std::real(v.z)};
    }
    
    template <typename S>
    friend Vec3<S> imag(Vec3<std::complex<S>> v) {
        return {std::imag(v.x), std::imag(v.y), std::imag(v.z)};
    }
    
    friend std::ostream& operator<< (std::ostream& os, Vec3<T> const& v) {
        return os << "<x=" << v.x << ", y=" << v.y << ", z=" << v.z << ">";
    }
};

typedef Vec3<float> fvec3;
typedef Vec3<double> vec3;
