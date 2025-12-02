#ifndef AP_INT_H
#define AP_INT_H

// Mock ap_int for simulation
template<int W>
class ap_int_base {
public:
    int value;
    
    ap_int_base() : value(0) {}
    ap_int_base(int v) : value(v) {}
    
    operator int() const { return value; }
    ap_int_base& operator=(int v) { value = v; return *this; }
    ap_int_base& operator+=(int v) { value += v; return *this; }
    ap_int_base& operator-=(int v) { value -= v; return *this; }
    ap_int_base& operator*=(int v) { value *= v; return *this; }
    ap_int_base& operator/=(int v) { value /= v; return *this; }
    
    // Pre/post increment operators
    ap_int_base& operator++() { ++value; return *this; }
    ap_int_base operator++(int) { ap_int_base tmp(*this); ++value; return tmp; }
    ap_int_base& operator--() { --value; return *this; }
    ap_int_base operator--(int) { ap_int_base tmp(*this); --value; return tmp; }
    
    // Comparison operators
    bool operator<(const ap_int_base& other) const { return value < other.value; }
    bool operator<(int other) const { return value < other; }
    bool operator>=(const ap_int_base& other) const { return value >= other.value; }
    bool operator>=(int other) const { return value >= other; }
};

template<int W>
using ap_int = ap_int_base<W>;

template<int W>
using ap_uint = ap_int_base<W>;

#endif