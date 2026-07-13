#ifndef SCALER_HPP
#define SCALER_HPP

#include "../../include/common.hpp"

extern "C" void scaler(
    data_t* scaler_mean,
    data_t* inv_scale,
    hls::stream<data_t>& features_in,
    hls::stream<data_t>& scaled_features_out,
    int_t num_features
);

#endif // SCALER_HPP
