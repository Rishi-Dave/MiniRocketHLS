# Makefile for MultiRocket C++ Testbench

CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -Wextra
INCLUDES = -I../third_party
LDFLAGS =

TARGET = multirocket_testbench
SOURCES = multirocket_testbench.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Model and test data files
MODEL_FILE = ../multirocket84_gunpoint_model.json
TEST_FILE = ../multirocket84_gunpoint_test.json

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	@echo ""
	@echo "Running testbench..."
	@echo ""
	./$(TARGET) $(MODEL_FILE) $(TEST_FILE)

test: run

clean:
	rm -f $(OBJECTS) $(TARGET)

help:
	@echo "MultiRocket C++ Testbench Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build testbench (default)"
	@echo "  run     - Build and run testbench"
	@echo "  test    - Same as run"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  make              # Build testbench"
	@echo "  make run          # Build and run"
	@echo "  make clean        # Clean build"
