#include "../include/npy_loader.h"
#include <sstream>
#include <algorithm>

bool NpyLoader::read_magic_and_version(std::ifstream& file, uint8_t& major, uint8_t& minor) {
    // NPY magic: 0x93 'N' 'U' 'M' 'P' 'Y'
    uint8_t magic[6];
    file.read(reinterpret_cast<char*>(magic), 6);

    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        std::cerr << "Error: Invalid NPY magic number" << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    return true;
}

bool NpyLoader::parse_npy_header(std::ifstream& file, NpyHeader& header) {
    uint8_t major, minor;
    if (!read_magic_and_version(file, major, minor)) {
        return false;
    }

    // Read header length
    uint16_t header_len = 0;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else if (major == 2 || major == 3) {
        uint32_t header_len_32;
        file.read(reinterpret_cast<char*>(&header_len_32), 4);
        header_len = static_cast<uint16_t>(header_len_32);
    } else {
        std::cerr << "Error: Unsupported NPY version " << (int)major << "." << (int)minor << std::endl;
        return false;
    }

    // Read header dictionary
    std::string header_str(header_len, ' ');
    file.read(&header_str[0], header_len);

    // Parse dtype
    size_t dtype_pos = header_str.find("'descr':");
    if (dtype_pos == std::string::npos) dtype_pos = header_str.find("\"descr\":");
    if (dtype_pos != std::string::npos) {
        size_t quote1 = header_str.find("'", dtype_pos + 8);
        if (quote1 == std::string::npos) quote1 = header_str.find("\"", dtype_pos + 8);
        size_t quote2 = header_str.find("'", quote1 + 1);
        if (quote2 == std::string::npos) quote2 = header_str.find("\"", quote1 + 1);

        if (quote1 != std::string::npos && quote2 != std::string::npos) {
            header.dtype = header_str.substr(quote1 + 1, quote2 - quote1 - 1);
        }
    }

    // Parse fortran_order
    size_t fortran_pos = header_str.find("'fortran_order':");
    if (fortran_pos == std::string::npos) fortran_pos = header_str.find("\"fortran_order\":");
    if (fortran_pos != std::string::npos) {
        size_t true_pos = header_str.find("True", fortran_pos);
        size_t false_pos = header_str.find("False", fortran_pos);
        header.fortran_order = (true_pos != std::string::npos &&
                               (false_pos == std::string::npos || true_pos < false_pos));
    }

    // Parse shape
    size_t shape_pos = header_str.find("'shape':");
    if (shape_pos == std::string::npos) shape_pos = header_str.find("\"shape\":");
    if (shape_pos != std::string::npos) {
        size_t paren1 = header_str.find("(", shape_pos);
        size_t paren2 = header_str.find(")", paren1);

        if (paren1 != std::string::npos && paren2 != std::string::npos) {
            std::string shape_str = header_str.substr(paren1 + 1, paren2 - paren1 - 1);
            std::istringstream iss(shape_str);
            std::string token;

            while (std::getline(iss, token, ',')) {
                // Remove whitespace
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty()) {
                    header.shape.push_back(std::stoull(token));
                }
            }
        }
    }

    return true;
}

bool NpyLoader::load_float32_2d(
    const std::string& filename,
    std::vector<std::vector<float>>& data,
    int& rows,
    int& cols
) {
    std::cout << "Loading NPY file: " << filename << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    NpyHeader header;
    if (!parse_npy_header(file, header)) {
        return false;
    }

    std::cout << "  dtype: " << header.dtype << std::endl;
    std::cout << "  shape: (";
    for (size_t i = 0; i < header.shape.size(); i++) {
        std::cout << header.shape[i];
        if (i < header.shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
    std::cout << "  fortran_order: " << (header.fortran_order ? "True" : "False") << std::endl;

    // Validate dtype
    if (header.dtype != "<f4" && header.dtype != "=f4" && header.dtype != "|f4") {
        std::cerr << "Error: Expected float32 dtype, got " << header.dtype << std::endl;
        return false;
    }

    // Validate shape
    if (header.shape.size() != 2) {
        std::cerr << "Error: Expected 2D array, got " << header.shape.size() << "D" << std::endl;
        return false;
    }

    if (header.fortran_order) {
        std::cerr << "Error: Fortran order not supported" << std::endl;
        return false;
    }

    rows = static_cast<int>(header.shape[0]);
    cols = static_cast<int>(header.shape[1]);

    std::cout << "  Loading " << rows << " x " << cols << " float32 array..." << std::endl;

    // Read data
    data.resize(rows);
    for (int i = 0; i < rows; i++) {
        data[i].resize(cols);
        file.read(reinterpret_cast<char*>(data[i].data()), cols * sizeof(float));

        if (file.fail()) {
            std::cerr << "Error: Failed to read row " << i << std::endl;
            return false;
        }

        // Progress indicator for large files
        if (rows > 1000 && i % (rows / 10) == 0) {
            std::cout << "  Progress: " << (i * 100 / rows) << "%" << std::endl;
        }
    }

    std::cout << "  NPY file loaded successfully!" << std::endl;
    return true;
}

bool NpyLoader::load_int32_1d(
    const std::string& filename,
    std::vector<int>& data,
    int& size
) {
    std::cout << "Loading NPY file: " << filename << std::endl;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    NpyHeader header;
    if (!parse_npy_header(file, header)) {
        return false;
    }

    std::cout << "  dtype: " << header.dtype << std::endl;
    std::cout << "  shape: (";
    for (size_t i = 0; i < header.shape.size(); i++) {
        std::cout << header.shape[i];
        if (i < header.shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;

    // Validate dtype
    if (header.dtype != "<i4" && header.dtype != "=i4" && header.dtype != "|i4") {
        std::cerr << "Error: Expected int32 dtype, got " << header.dtype << std::endl;
        return false;
    }

    // Validate shape
    if (header.shape.size() != 1) {
        std::cerr << "Error: Expected 1D array, got " << header.shape.size() << "D" << std::endl;
        return false;
    }

    size = static_cast<int>(header.shape[0]);

    std::cout << "  Loading " << size << " int32 elements..." << std::endl;

    // Read data
    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), size * sizeof(int32_t));

    if (file.fail()) {
        std::cerr << "Error: Failed to read data" << std::endl;
        return false;
    }

    std::cout << "  NPY file loaded successfully!" << std::endl;
    return true;
}
