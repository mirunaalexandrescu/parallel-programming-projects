# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mirunaalexandrescu/Desktop/kmeans

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/kmeans.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kmeans.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kmeans.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kmeans.dir/flags.make

CMakeFiles/kmeans.dir/main.cpp.o: CMakeFiles/kmeans.dir/flags.make
CMakeFiles/kmeans.dir/main.cpp.o: /Users/mirunaalexandrescu/Desktop/kmeans/main.cpp
CMakeFiles/kmeans.dir/main.cpp.o: CMakeFiles/kmeans.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kmeans.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans.dir/main.cpp.o -MF CMakeFiles/kmeans.dir/main.cpp.o.d -o CMakeFiles/kmeans.dir/main.cpp.o -c /Users/mirunaalexandrescu/Desktop/kmeans/main.cpp

CMakeFiles/kmeans.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mirunaalexandrescu/Desktop/kmeans/main.cpp > CMakeFiles/kmeans.dir/main.cpp.i

CMakeFiles/kmeans.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mirunaalexandrescu/Desktop/kmeans/main.cpp -o CMakeFiles/kmeans.dir/main.cpp.s

# Object files for target kmeans
kmeans_OBJECTS = \
"CMakeFiles/kmeans.dir/main.cpp.o"

# External object files for target kmeans
kmeans_EXTERNAL_OBJECTS =

kmeans: CMakeFiles/kmeans.dir/main.cpp.o
kmeans: CMakeFiles/kmeans.dir/build.make
kmeans: /opt/homebrew/opt/libomp/lib/libomp.dylib
kmeans: CMakeFiles/kmeans.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kmeans"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kmeans.dir/build: kmeans
.PHONY : CMakeFiles/kmeans.dir/build

CMakeFiles/kmeans.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kmeans.dir/clean

CMakeFiles/kmeans.dir/depend:
	cd /Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mirunaalexandrescu/Desktop/kmeans /Users/mirunaalexandrescu/Desktop/kmeans /Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug /Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug /Users/mirunaalexandrescu/Desktop/kmeans/cmake-build-debug/CMakeFiles/kmeans.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/kmeans.dir/depend

