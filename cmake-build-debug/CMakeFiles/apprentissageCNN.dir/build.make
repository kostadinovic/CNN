# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/nemanja/CLionProjects/apprentissageCNN

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/apprentissageCNN.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/apprentissageCNN.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/apprentissageCNN.dir/flags.make

CMakeFiles/apprentissageCNN.dir/main.c.o: CMakeFiles/apprentissageCNN.dir/flags.make
CMakeFiles/apprentissageCNN.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/apprentissageCNN.dir/main.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/apprentissageCNN.dir/main.c.o   -c /Users/nemanja/CLionProjects/apprentissageCNN/main.c

CMakeFiles/apprentissageCNN.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/apprentissageCNN.dir/main.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/nemanja/CLionProjects/apprentissageCNN/main.c > CMakeFiles/apprentissageCNN.dir/main.c.i

CMakeFiles/apprentissageCNN.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/apprentissageCNN.dir/main.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/nemanja/CLionProjects/apprentissageCNN/main.c -o CMakeFiles/apprentissageCNN.dir/main.c.s

CMakeFiles/apprentissageCNN.dir/matrice.c.o: CMakeFiles/apprentissageCNN.dir/flags.make
CMakeFiles/apprentissageCNN.dir/matrice.c.o: ../matrice.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/apprentissageCNN.dir/matrice.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/apprentissageCNN.dir/matrice.c.o   -c /Users/nemanja/CLionProjects/apprentissageCNN/matrice.c

CMakeFiles/apprentissageCNN.dir/matrice.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/apprentissageCNN.dir/matrice.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/nemanja/CLionProjects/apprentissageCNN/matrice.c > CMakeFiles/apprentissageCNN.dir/matrice.c.i

CMakeFiles/apprentissageCNN.dir/matrice.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/apprentissageCNN.dir/matrice.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/nemanja/CLionProjects/apprentissageCNN/matrice.c -o CMakeFiles/apprentissageCNN.dir/matrice.c.s

CMakeFiles/apprentissageCNN.dir/load_mnist.c.o: CMakeFiles/apprentissageCNN.dir/flags.make
CMakeFiles/apprentissageCNN.dir/load_mnist.c.o: ../load_mnist.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/apprentissageCNN.dir/load_mnist.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/apprentissageCNN.dir/load_mnist.c.o   -c /Users/nemanja/CLionProjects/apprentissageCNN/load_mnist.c

CMakeFiles/apprentissageCNN.dir/load_mnist.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/apprentissageCNN.dir/load_mnist.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/nemanja/CLionProjects/apprentissageCNN/load_mnist.c > CMakeFiles/apprentissageCNN.dir/load_mnist.c.i

CMakeFiles/apprentissageCNN.dir/load_mnist.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/apprentissageCNN.dir/load_mnist.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/nemanja/CLionProjects/apprentissageCNN/load_mnist.c -o CMakeFiles/apprentissageCNN.dir/load_mnist.c.s

CMakeFiles/apprentissageCNN.dir/conv_nn.c.o: CMakeFiles/apprentissageCNN.dir/flags.make
CMakeFiles/apprentissageCNN.dir/conv_nn.c.o: ../conv_nn.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/apprentissageCNN.dir/conv_nn.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/apprentissageCNN.dir/conv_nn.c.o   -c /Users/nemanja/CLionProjects/apprentissageCNN/conv_nn.c

CMakeFiles/apprentissageCNN.dir/conv_nn.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/apprentissageCNN.dir/conv_nn.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/nemanja/CLionProjects/apprentissageCNN/conv_nn.c > CMakeFiles/apprentissageCNN.dir/conv_nn.c.i

CMakeFiles/apprentissageCNN.dir/conv_nn.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/apprentissageCNN.dir/conv_nn.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/nemanja/CLionProjects/apprentissageCNN/conv_nn.c -o CMakeFiles/apprentissageCNN.dir/conv_nn.c.s

# Object files for target apprentissageCNN
apprentissageCNN_OBJECTS = \
"CMakeFiles/apprentissageCNN.dir/main.c.o" \
"CMakeFiles/apprentissageCNN.dir/matrice.c.o" \
"CMakeFiles/apprentissageCNN.dir/load_mnist.c.o" \
"CMakeFiles/apprentissageCNN.dir/conv_nn.c.o"

# External object files for target apprentissageCNN
apprentissageCNN_EXTERNAL_OBJECTS =

apprentissageCNN: CMakeFiles/apprentissageCNN.dir/main.c.o
apprentissageCNN: CMakeFiles/apprentissageCNN.dir/matrice.c.o
apprentissageCNN: CMakeFiles/apprentissageCNN.dir/load_mnist.c.o
apprentissageCNN: CMakeFiles/apprentissageCNN.dir/conv_nn.c.o
apprentissageCNN: CMakeFiles/apprentissageCNN.dir/build.make
apprentissageCNN: CMakeFiles/apprentissageCNN.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking C executable apprentissageCNN"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/apprentissageCNN.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/apprentissageCNN.dir/build: apprentissageCNN

.PHONY : CMakeFiles/apprentissageCNN.dir/build

CMakeFiles/apprentissageCNN.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/apprentissageCNN.dir/cmake_clean.cmake
.PHONY : CMakeFiles/apprentissageCNN.dir/clean

CMakeFiles/apprentissageCNN.dir/depend:
	cd /Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/nemanja/CLionProjects/apprentissageCNN /Users/nemanja/CLionProjects/apprentissageCNN /Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug /Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug /Users/nemanja/CLionProjects/apprentissageCNN/cmake-build-debug/CMakeFiles/apprentissageCNN.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/apprentissageCNN.dir/depend

