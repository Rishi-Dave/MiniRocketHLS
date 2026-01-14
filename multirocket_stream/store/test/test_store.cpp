#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../include/store.hpp"

int main(int argc, char* argv[]) {
    std::cout << "STORE TESTBED" << std::endl; 

    /*
    
    call and test store
    
    */

    data_t test_array[MAX_TIME_SERIES_LENGTH][MAX_CLASSES];
    data_t prediction_output_array[MAX_TIME_SERIES_LENGTH * MAX_CLASSES];
    hls::stream<data_t> prediction_input;

    for (int j = 0;  j < MAX_TIME_SERIES_LENGTH; j++) {
        for (int i = 0; i < MAX_CLASSES; i++) {
            test_array[j][i] = static_cast<data_t>(i * MAX_TIME_SERIES_LENGTH + j) * 1.5f; // Example data
            prediction_input.write(test_array[j][i]);
        }
    }

    store(prediction_input, prediction_output_array, MAX_TIME_SERIES_LENGTH, MAX_CLASSES);
    
    bool match = true;
    int count = 0; 
    for (int i = 0; i < MAX_TIME_SERIES_LENGTH * MAX_CLASSES; i++) {
        if (prediction_output_array[i] != test_array[i / MAX_CLASSES][i % MAX_CLASSES]) {
            match = false;
            std::cout << "Mismatch at index " << i << ": expected " << test_array[i / MAX_CLASSES][i % MAX_CLASSES] << ", got " << prediction_output_array[i] << std::endl;
        } else {
            count++;
        }
    }
    std::cout << "STORE test completed. " << count << " out of " << MAX_CLASSES * MAX_TIME_SERIES_LENGTH << " values matched." << std::endl;
    return match ? 0 : 1;
}