#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include "../../include/common.hpp"

extern "C" void classifier(
    data_t* coefficients,
    data_t* intercept,
    data_t* prediction_output,
    hls::stream<data_t>& scaled_features_in,
    int_t num_features,
    int_t num_classes
);

void linear_classifier_predict_hls(
    data_t scaled_features[MAX_FEATURES],
    data_t predictions[MAX_CLASSES],
    data_t coefficients[MAX_CLASSES][MAX_FEATURES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes
);

#endif // CLASSIFIER_HPP
