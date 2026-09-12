CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17
NVCC = nvcc
# sm_75 = RTX 2080 Ti (cerebro). Override: make CUDA_ARCH=sm_80 ...
CUDA_ARCH ?= sm_75
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH)

TARGET = minirocket_cpu
GPU_UDP_SERVER = minirocket_gpu_udp_server
UDP_CLIENT = minirocket_udp_client

MODELS_DIR ?= ../fpga-network/minirocket_fused
RESULTS_DIR = ../results

.PHONY: all clean gpu-stream stream-server stream-client \
        minirocket_gpu minirocket_gpu_streams minirocket_cpu_mt \
        run-insectsound run-fruitflies run-mosquitosound run-all

all: $(TARGET)

$(TARGET): minirocket_cpu_inference.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# OpenMP multithreaded CPU
minirocket_cpu_mt: minirocket_cpu_inference_mt.cpp
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $<

# Existing single-node GPU binaries
minirocket_gpu: minirocket_gpu_inference.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

minirocket_gpu_streams: minirocket_gpu_inference_streams.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Naive streaming UDP (phase a) — NAIVE full-recompute, no DP-Reuse
stream-server: $(GPU_UDP_SERVER)
stream-client: $(UDP_CLIENT)
gpu-stream: stream-server stream-client

$(GPU_UDP_SERVER): minirocket_gpu_udp_server.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(UDP_CLIENT): minirocket_udp_client.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run-insectsound: $(TARGET)
	./$(TARGET) $(MODELS_DIR)/InsectSound_minirocket_model.json \
	            $(MODELS_DIR)/InsectSound_test_data_compact.json \
	            $(RESULTS_DIR)/MiniRocket_CPU_cpp_InsectSound_per_sample.csv

run-fruitflies: $(TARGET)
	./$(TARGET) $(MODELS_DIR)/FruitFlies_minirocket_model.json \
	            $(MODELS_DIR)/FruitFlies_test_data_compact.json \
	            $(RESULTS_DIR)/MiniRocket_CPU_cpp_FruitFlies_per_sample.csv

run-mosquitosound: $(TARGET)
	./$(TARGET) $(MODELS_DIR)/MosquitoSound_minirocket_model.json \
	            $(MODELS_DIR)/MosquitoSound_test_data_compact.json \
	            $(RESULTS_DIR)/MiniRocket_CPU_cpp_MosquitoSound_per_sample.csv

run-all: run-insectsound run-fruitflies run-mosquitosound

clean:
	rm -f $(TARGET) minirocket_cpu_mt minirocket_gpu minirocket_gpu_streams \
	      $(GPU_UDP_SERVER) $(UDP_CLIENT)
