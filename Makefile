CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -I/opt/homebrew/Cellar/jsoncpp/1.9.6/include
LDFLAGS = -L/opt/homebrew/Cellar/jsoncpp/1.9.6/lib -ljsoncpp

TARGET = minirocket_test
SOURCE = minirocket_inference.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET) minirocket_model.json minirocket_model_test_data.json

.PHONY: all clean test