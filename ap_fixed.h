#ifndef AP_FIXED_H
#define AP_FIXED_H

#include <iostream>

// Mock ap_fixed for simulation
template<int W, int I>
class ap_fixed_base {
public:
    float value;
    
    ap_fixed_base() : value(0.0f) {}
    ap_fixed_base(float v) : value(v) {}
    ap_fixed_base(double v) : value(static_cast<float>(v)) {}
    ap_fixed_base(int v) : value(static_cast<float>(v)) {}
    
    operator float() const { return value; }
    operator double() const { return static_cast<double>(value); }
    
    ap_fixed_base& operator=(float v) { value = v; return *this; }
    ap_fixed_base& operator=(double v) { value = static_cast<float>(v); return *this; }
    ap_fixed_base& operator=(int v) { value = static_cast<float>(v); return *this; }
    
    ap_fixed_base operator+(const ap_fixed_base& other) const { return ap_fixed_base(value + other.value); }
    ap_fixed_base operator-(const ap_fixed_base& other) const { return ap_fixed_base(value - other.value); }
    ap_fixed_base operator*(const ap_fixed_base& other) const { return ap_fixed_base(value * other.value); }
    ap_fixed_base operator/(const ap_fixed_base& other) const { return ap_fixed_base(value / other.value); }
    
    bool operator>(const ap_fixed_base& other) const { return value > other.value; }
    bool operator<(const ap_fixed_base& other) const { return value < other.value; }
    bool operator>=(const ap_fixed_base& other) const { return value >= other.value; }
    bool operator<=(const ap_fixed_base& other) const { return value <= other.value; }
    bool operator==(const ap_fixed_base& other) const { return value == other.value; }
    bool operator!=(const ap_fixed_base& other) const { return value != other.value; }
    
    bool operator>(float other) const { return value > other; }
    bool operator<(float other) const { return value < other; }
    bool operator>=(float other) const { return value >= other; }
    bool operator<=(float other) const { return value <= other; }
    bool operator==(float other) const { return value == other; }
    bool operator!=(float other) const { return value != other; }
    
    ap_fixed_base& operator+=(const ap_fixed_base& other) { value += other.value; return *this; }
    ap_fixed_base& operator-=(const ap_fixed_base& other) { value -= other.value; return *this; }
    ap_fixed_base& operator*=(const ap_fixed_base& other) { value *= other.value; return *this; }
    ap_fixed_base& operator/=(const ap_fixed_base& other) { value /= other.value; return *this; }
};

template<int W, int I>
using ap_fixed = ap_fixed_base<W, I>;

template<int W, int I>
using ap_ufixed = ap_fixed_base<W, I>;

// Stream output operator
template<int W, int I>
std::ostream& operator<<(std::ostream& os, const ap_fixed_base<W, I>& val) {
    return os << val.value;
}

#endif