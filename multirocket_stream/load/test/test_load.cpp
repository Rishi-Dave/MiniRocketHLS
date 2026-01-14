#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "../include/load.hpp"

int main(int argc, char* argv[]) {
    std::cout << "LOAD TESTBED" << std::endl; 

    /*
    
    call and test load
    
    */

    data_t test_array[MAX_TIME_SERIES_LENGTH];
    hls::stream<data_t> test_stream;

    for (int i = 0; i < MAX_TIME_SERIES_LENGTH; i++) {
        test_array[i] = static_cast<data_t>(i) * 1.5f; // Example data
    }

    load(test_array, test_stream, MAX_TIME_SERIES_LENGTH);
    bool match = true;
    int count = 0; 
    for (int i = 0; i < MAX_TIME_SERIES_LENGTH; i++) {
        data_t val = test_stream.read();
        if (val != test_array[i]) {
            match = false;
            std::cout << "Mismatch at index " << i << ": expected " << test_array[i] << ", got " << val << std::endl;
        } else {
            count++;
        }
    }
    std::cout << "LOAD test completed. " << count << " out of " << MAX_TIME_SERIES_LENGTH << " values matched." << std::endl;
    return match ? 0 : 1;
}