#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>

/**
 * Simple .npy file reader for NumPy binary format
 * Supports float32 and int32 arrays
 * Does NOT support complex types, object arrays, or compressed files
 */
class NpyLoader {
public:
    /**
     * Load a 2D float32 array from .npy file
     * @param filename Path to .npy file
     * @param data Output vector of vectors
     * @param rows Output number of rows
     * @param cols Output number of columns
     * @return true if successful
     */
    static bool load_float32_2d(
        const std::string& filename,
        std::vector<std::vector<float>>& data,
        int& rows,
        int& cols
    );

    /**
     * Load a 1D int32 array from .npy file
     * @param filename Path to .npy file
     * @param data Output vector
     * @param size Output number of elements
     * @return true if successful
     */
    static bool load_int32_1d(
        const std::string& filename,
        std::vector<int>& data,
        int& size
    );

private:
    struct NpyHeader {
        std::string dtype;
        bool fortran_order;
        std::vector<size_t> shape;
    };

    static bool parse_npy_header(std::ifstream& file, NpyHeader& header);
    static bool read_magic_and_version(std::ifstream& file, uint8_t& major, uint8_t& minor);
};

#endif // NPY_LOADER_H
