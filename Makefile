CXX = g++
CXXFLAGS = -O3 -std=c++17 -DNDEBUG

TARGET = simplelogics

all:
	$(CXX) $(CXXFLAGS) simplelogics.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
