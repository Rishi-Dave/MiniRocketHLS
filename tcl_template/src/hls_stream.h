#ifndef HLS_STREAM_H
#define HLS_STREAM_H

#include <queue>

// Mock hls::stream for simulation
namespace hls {
    template<typename T>
    class stream {
    private:
        std::queue<T> data;
        
    public:
        void write(const T& value) {
            data.push(value);
        }
        
        T read() {
            if (data.empty()) {
                return T();
            }
            T value = data.front();
            data.pop();
            return value;
        }
        
        bool empty() const {
            return data.empty();
        }
        
        bool full() const {
            return false; // Unlimited for simulation
        }
    };
}

#endif