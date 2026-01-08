#ifndef HYDRA_INFERENCE_HLS_H
#define HYDRA_INFERENCE_HLS_H

#include <cmath>
#include <cstring>

// Data types for HLS synthesis
typedef float data_t;
typedef int int_t;

// HYDRA Algorithm Constants
#define MAX_TIME_SERIES_LENGTH 5120  // Updated for UCR datasets (InsectSound:600, MosquitoSound:3750, FruitFlies:5000)
#define NUM_KERNELS 512
#define NUM_GROUPS 8
#define KERNELS_PER_GROUP 64  // NUM_KERNELS / NUM_GROUPS
#define KERNEL_SIZE 9
#define MAX_FEATURES 2048
#define MAX_DILATIONS 8
#define MAX_CLASSES 10
#define POOLING_OPERATORS 2  // Max + Mean

// Model parameters structure for HLS
struct HydraModelParams_HLS {
    // Dictionary kernel weights: [NUM_KERNELS][KERNEL_SIZE]
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE];

    // Bias values for each kernel
    data_t biases[NUM_KERNELS];

    // Group assignments for kernels (0 to NUM_GROUPS-1)
    int_t group_assignments[NUM_KERNELS];

    // Dilations per kernel
    int_t dilations[NUM_KERNELS];

    // StandardScaler parameters
    data_t scaler_mean[MAX_FEATURES];
    data_t scaler_scale[MAX_FEATURES];

    // Ridge classifier coefficients: [num_features][num_classes]
    data_t coefficients[MAX_FEATURES * MAX_CLASSES];

    // Ridge classifier intercept: [num_classes]
    data_t intercept[MAX_CLASSES];

    // Metadata
    int_t num_features;
    int_t num_classes;
    int_t time_series_length;
};

// Function prototypes

/**
 * Main HYDRA inference kernel (OpenCL entry point)
 *
 * Performs dictionary-based time series classification using HYDRA algorithm:
 * 1. Applies 512 convolutional kernels to input time series
 * 2. Extracts max pooling + global mean features
 * 3. Normalizes features using StandardScaler
 * 4. Classifies using Ridge linear classifier
 *
 * @param time_series_input Input time series data [time_series_length]
 * @param prediction_output Output class predictions [num_classes]
 * @param coefficients Ridge classifier weights [num_features * num_classes]
 * @param intercept Ridge classifier bias [num_classes]
 * @param scaler_mean StandardScaler mean values [num_features]
 * @param scaler_scale StandardScaler scale values [num_features]
 * @param kernel_weights Dictionary kernels [NUM_KERNELS * KERNEL_SIZE]
 * @param biases Kernel bias values [NUM_KERNELS]
 * @param dilations Dilation values for each kernel [NUM_KERNELS]
 * @param time_series_length Length of input time series
 * @param num_features Total number of features extracted
 * @param num_classes Number of output classes
 * @param num_groups Number of kernel groups
 */
extern "C" void hydra_inference(
    data_t* time_series_input,
    data_t* prediction_output,
    data_t* coefficients,
    data_t* intercept,
    data_t* scaler_mean,
    data_t* scaler_scale,
    data_t* kernel_weights,
    data_t* biases,
    int_t* dilations,
    int_t time_series_length,
    int_t num_features,
    int_t num_classes,
    int_t num_groups
);

/**
 * Apply single convolutional kernel with dilation
 *
 * @param time_series Input time series
 * @param weights Kernel weights [KERNEL_SIZE]
 * @param bias Kernel bias value
 * @param dilation Dilation factor
 * @param length Time series length
 * @param output Convolution output buffer
 * @param output_length Output length after convolution
 */
void apply_kernel_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t weights[KERNEL_SIZE],
    data_t bias,
    int_t dilation,
    int_t length,
    data_t output[MAX_TIME_SERIES_LENGTH],
    int_t& output_length
);

/**
 * Extract features from time series using HYDRA dictionary
 *
 * For each kernel:
 *   - Apply convolution
 *   - Extract max pooling
 *   - Extract global mean
 *
 * @param time_series Input time series
 * @param features Output feature vector
 * @param kernel_weights Dictionary kernels
 * @param biases Kernel biases
 * @param dilations Dilation values
 * @param length Time series length
 * @param num_features Output number of features
 */
void hydra_feature_extraction_hls(
    data_t time_series[MAX_TIME_SERIES_LENGTH],
    data_t features[MAX_FEATURES],
    data_t kernel_weights[NUM_KERNELS * KERNEL_SIZE],
    data_t biases[NUM_KERNELS],
    int_t dilations[NUM_KERNELS],
    int_t length,
    int_t& num_features
);

/**
 * Compute max pooling from convolution output
 *
 * @param values Convolution output
 * @param length Number of values
 * @param max_val Maximum value (output)
 */
void compute_max_pooling(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& max_val
);

/**
 * Compute global mean from convolution output
 *
 * @param values Convolution output
 * @param length Number of values
 * @param mean_val Mean value (output)
 */
void compute_global_mean(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& mean_val
);

/**
 * Compute both max and mean pooling in single pass
 *
 * @param values Convolution output
 * @param length Number of values
 * @param max_val Maximum value (output)
 * @param mean_val Mean value (output)
 */
void compute_two_pooling_operators(
    data_t values[MAX_TIME_SERIES_LENGTH],
    int_t length,
    data_t& max_val,
    data_t& mean_val
);

/**
 * Apply StandardScaler normalization to features
 *
 * scaled = (features - mean) / scale
 *
 * @param features Input features [num_features]
 * @param scaler_mean Mean values [num_features]
 * @param scaler_scale Scale values [num_features]
 * @param num_features Number of features
 */
void apply_scaler_hls(
    data_t features[MAX_FEATURES],
    data_t scaler_mean[MAX_FEATURES],
    data_t scaler_scale[MAX_FEATURES],
    int_t num_features
);

/**
 * Ridge linear classifier prediction
 *
 * prediction = features @ coefficients + intercept
 *
 * @param features Normalized input features [num_features]
 * @param coefficients Classifier weights [num_features * num_classes]
 * @param intercept Classifier bias [num_classes]
 * @param num_features Number of input features
 * @param num_classes Number of output classes
 * @param prediction Output predictions [num_classes]
 */
void linear_classifier_predict_hls(
    data_t features[MAX_FEATURES],
    data_t coefficients[MAX_FEATURES * MAX_CLASSES],
    data_t intercept[MAX_CLASSES],
    int_t num_features,
    int_t num_classes,
    data_t prediction[MAX_CLASSES]
);

#endif // HYDRA_INFERENCE_HLS_H
