# Optimized native C++ baseline for HYDRA inference.
# Compiles the same kernel sources used by HLS as plain C++ with -O3.

CXX      ?= g++
CXXSTD   ?= -std=c++17
OPT      ?= -O3 -march=native -ffast-math -funroll-loops -DNDEBUG
WARN     ?= -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare
INCLUDES := -I../hydra/include

KERNEL_SRCS := ../hydra/src/hydra.cpp ../hydra/src/hydra_pooling.cpp
HOST_SRC    := hydra_cpp_baseline.cpp
BIN         := hydra_cpp_baseline

.PHONY: all clean

all: $(BIN)

$(BIN): $(HOST_SRC) $(KERNEL_SRCS) ../hydra/include/hydra.hpp
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(INCLUDES) -o $@ $(HOST_SRC) $(KERNEL_SRCS)

clean:
	rm -f $(BIN)
