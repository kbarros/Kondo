//
//  vec3.cpp
//  tibidy
//
//  Created by Kipton Barros on 6/19/14.
//
//

#include <cmath>

template<typename T>
class vec3 {
    T x=0, y=0, z=0;
    
    vec3<T> operator+(vec3<T> that) const {
        return {x+that.x, y+that.y, z+that.z};
    }

    vec3<T> operator-(vec3<T> that) const {
        return {x-that.x, y-that.y, z-that.z};
    }

    vec3<T> operator*(T a) const {
        return {x*a, y*a, z*a};
    }
    
    /*
    Vec3 Vec3::operator/(double a) const {
        return Vec3 {x/a, y/a, z/a};
    }
    
    double Vec3::dotProduct(Vec3 that) const {
        return x*that.x + y*that.y + z*that.z;
    }
    
    double Vec3::norm() const {
        return sqrt(this->dotProduct(*this));
    }
    
    Vec3 Vec3::normalize() const {
        double m = norm();
        if (m == 0) {
            return *this;
        }
        else {
            return *this / m;
        }
    }
    
    void Vec3::operator+=(Vec3 that) {
        x += that.x;
        y += that.y;
        z += that.z;
    }
    
    void Vec3::operator-=(Vec3 that) {
        x -= that.x;
        y -= that.y;
        z -= that.z;
    }
    
    void Vec3::operator*=(double a) {
        x *= a;
        y *= a;
        z *= a;
    }
    
    void Vec3::operator/=(double a) {
        x /= a;
        y /= a;
        z /= a;
    }
    
*/
};

template<typename T>
std::ostream& operator<< (std::ostream& os, vec3<T> const& v) {
    return os << "<x=" << v.x << ", y=" << v.y << ", z=" << v.z << ">";
}
