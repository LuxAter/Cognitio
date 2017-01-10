CPP_FILES = $(wildcard *.cpp)
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))
TOTAL_OBJ_FILES = $(wildcard */*.o) $(wildcard */*/*.o) $(wildcard */*/*/*.o)
COMPILER_FLAGS = -MMD -std=c++11 -w -c

all: subsystem top_obj

%.o: %.cpp
	$(CC) $(COMPILER_FLAGS) -o $(notdir $*).o $*.cpp

.PHONY : top_obj
top_obj:$(OBJ_FILES)

.PHONY : subsystem
subsystem:
	setterm --foreground blue
	cd genetic_algorithm && $(MAKE)
	setterm --foreground green
	cd neural_network && $(MAKE)
