CPP_FILES = $(wildcard *.cpp)
OBJ_FILES = $(notdir $(CPP_FILES:.cpp=.o))

all: $(OBJ_FILES) subsystem

%.o: %.cpp %.hpp
	@printf "Compiling $*.cpp...\n"
	@$(COMPILER) $(FLAGS) -o $(notdir $*).o $*.cpp

.PHONY : subsystem
subsystem:
	@setterm -fore blue; printf "=====Neural Network=====\n"; setterm -fore 7
	@cd neural_network && $(MAKE)
	# @cd classes && $(MAKE)
	# @cd equation && $(MAKE)
	# @cd vector && $(MAKE)
