#ifndef FEATURE_EXTRACTION_HPP
#define FEATURE_EXTRACTION_HPP

#include "../../include/common.hpp"

extern "C" void feature_extraction(
    data_t* time_series_input,
    int_t*  dilations,
    int_t*  num_features_per_dilation,
    data_t* biases,
    hls::stream<data_t>& features_out,
    int_t time_series_length,
    int_t num_features,
    int_t num_dilations
);

void apply_kernel_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t convolutions[MAX_TIME_SERIES_LENGTH],
    int_t kernel_idx,
    int_t dilation,
    int_t time_series_length,
    int_t* output_length
);

void minirocket_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    int_t dilations[MAX_DILATIONS],
    int_t num_features_per_dilation[MAX_DILATIONS],
    data_t biases[MAX_FEATURES],
    int_t time_series_length,
    int_t num_dilations,
    int_t num_features
);

#endif // FEATURE_EXTRACTION_HPP
