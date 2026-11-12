CC=g++
CXXFLAGS= -Wall -Wvla -Wextra -Werror -g -std=c++20
LDFLAGS= -lm
HEADERS= Matrix.h Activation.h Dense.h MlpNetwork.h
OBJS= Matrix.o Activation.o Dense.o MlpNetwork.o main.o

# the main Target
all: mlpnetwork

# Linkage
mlpnetwork: $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o mlpnetwork

#compilation
%.o: %.cpp $(HEADERS)
	$(CC) $(CXXFLAGS) -c $< -o $@

#cleaning
.PHONY: clean
clean:
	rm -rf *.o mlpnetwork