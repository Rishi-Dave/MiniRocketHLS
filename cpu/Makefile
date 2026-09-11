CXX = g++
CXXFLAGS = -O3 -march=native -std=c++17
TARGET = minirocket_cpu

MODELS_DIR ?= ../fpga-network/minirocket_fused
RESULTS_DIR = ../results

.PHONY: all clean run-insectsound run-fruitflies run-mosquitosound run-all

all: $(TARGET)

$(TARGET): minirocket_cpu_inference.cpp
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
	rm -f $(TARGET)
