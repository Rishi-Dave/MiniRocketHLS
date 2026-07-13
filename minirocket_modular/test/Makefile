#
# Standalone module tests for MiniRocket Modular pipeline
# Pure C++ (no FPGA, no HLS tools needed)
#

CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -I../include -I../host/include
LDFLAGS :=

# Model/test data (default: GunPoint)
MODEL_JSON ?= ../minirocket_model.json
TEST_JSON  ?= ../minirocket_model_test_data.json
NUM_SAMPLES ?= 0

.PHONY: all clean test test-mosquito test-insect test-fruitflies

all: test_fe_csim test_v16_fixed

test_fe_csim: test_feature_extraction_csim.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

test_v16_fixed: test_v16_fixed_csim.cpp
	$(CXX) $(CXXFLAGS) -Wno-misleading-indentation -o $@ $< $(LDFLAGS)

# Run with default dataset
test: test_fe_csim
	./test_fe_csim $(MODEL_JSON) $(TEST_JSON) $(NUM_SAMPLES)

# Run with benchmark datasets
test-mosquito: test_fe_csim
	./test_fe_csim ../MosquitoSound_minirocket_model.json ../MosquitoSound_minirocket_model_test_data.json 100

test-insect: test_fe_csim
	./test_fe_csim ../InsectSound_minirocket_model.json ../InsectSound_minirocket_model_test_data.json 100

test-fruitflies: test_fe_csim
	./test_fe_csim ../FruitFlies_minirocket_model.json ../FruitFlies_minirocket_model_test_data.json 100

# Run all benchmark datasets
test-all: test-mosquito test-insect test-fruitflies

# Validate v16_fixed restructuring matches original
test-v16: test_v16_fixed
	./test_v16_fixed $(MODEL_JSON) $(TEST_JSON)

clean:
	rm -f test_fe_csim test_v16_fixed feature_dump_csim.csv
