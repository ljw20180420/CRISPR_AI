# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap

# Include any dependencies generated for this target.
include CMakeFiles/indelmh.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/indelmh.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/indelmh.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/indelmh.dir/flags.make

CMakeFiles/indelmh.dir/indel.cpp.o: CMakeFiles/indelmh.dir/flags.make
CMakeFiles/indelmh.dir/indel.cpp.o: indel.cpp
CMakeFiles/indelmh.dir/indel.cpp.o: CMakeFiles/indelmh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/indelmh.dir/indel.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/indelmh.dir/indel.cpp.o -MF CMakeFiles/indelmh.dir/indel.cpp.o.d -o CMakeFiles/indelmh.dir/indel.cpp.o -c /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indel.cpp

CMakeFiles/indelmh.dir/indel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/indelmh.dir/indel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indel.cpp > CMakeFiles/indelmh.dir/indel.cpp.i

CMakeFiles/indelmh.dir/indel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/indelmh.dir/indel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indel.cpp -o CMakeFiles/indelmh.dir/indel.cpp.s

CMakeFiles/indelmh.dir/oligo.cpp.o: CMakeFiles/indelmh.dir/flags.make
CMakeFiles/indelmh.dir/oligo.cpp.o: oligo.cpp
CMakeFiles/indelmh.dir/oligo.cpp.o: CMakeFiles/indelmh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/indelmh.dir/oligo.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/indelmh.dir/oligo.cpp.o -MF CMakeFiles/indelmh.dir/oligo.cpp.o.d -o CMakeFiles/indelmh.dir/oligo.cpp.o -c /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/oligo.cpp

CMakeFiles/indelmh.dir/oligo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/indelmh.dir/oligo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/oligo.cpp > CMakeFiles/indelmh.dir/oligo.cpp.i

CMakeFiles/indelmh.dir/oligo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/indelmh.dir/oligo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/oligo.cpp -o CMakeFiles/indelmh.dir/oligo.cpp.s

CMakeFiles/indelmh.dir/readmap.cpp.o: CMakeFiles/indelmh.dir/flags.make
CMakeFiles/indelmh.dir/readmap.cpp.o: readmap.cpp
CMakeFiles/indelmh.dir/readmap.cpp.o: CMakeFiles/indelmh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/indelmh.dir/readmap.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/indelmh.dir/readmap.cpp.o -MF CMakeFiles/indelmh.dir/readmap.cpp.o.d -o CMakeFiles/indelmh.dir/readmap.cpp.o -c /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/readmap.cpp

CMakeFiles/indelmh.dir/readmap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/indelmh.dir/readmap.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/readmap.cpp > CMakeFiles/indelmh.dir/readmap.cpp.i

CMakeFiles/indelmh.dir/readmap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/indelmh.dir/readmap.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/readmap.cpp -o CMakeFiles/indelmh.dir/readmap.cpp.s

CMakeFiles/indelmh.dir/gen.cpp.o: CMakeFiles/indelmh.dir/flags.make
CMakeFiles/indelmh.dir/gen.cpp.o: gen.cpp
CMakeFiles/indelmh.dir/gen.cpp.o: CMakeFiles/indelmh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/indelmh.dir/gen.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/indelmh.dir/gen.cpp.o -MF CMakeFiles/indelmh.dir/gen.cpp.o.d -o CMakeFiles/indelmh.dir/gen.cpp.o -c /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/gen.cpp

CMakeFiles/indelmh.dir/gen.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/indelmh.dir/gen.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/gen.cpp > CMakeFiles/indelmh.dir/gen.cpp.i

CMakeFiles/indelmh.dir/gen.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/indelmh.dir/gen.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/gen.cpp -o CMakeFiles/indelmh.dir/gen.cpp.s

CMakeFiles/indelmh.dir/indelmh.cpp.o: CMakeFiles/indelmh.dir/flags.make
CMakeFiles/indelmh.dir/indelmh.cpp.o: indelmh.cpp
CMakeFiles/indelmh.dir/indelmh.cpp.o: CMakeFiles/indelmh.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/indelmh.dir/indelmh.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/indelmh.dir/indelmh.cpp.o -MF CMakeFiles/indelmh.dir/indelmh.cpp.o.d -o CMakeFiles/indelmh.dir/indelmh.cpp.o -c /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelmh.cpp

CMakeFiles/indelmh.dir/indelmh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/indelmh.dir/indelmh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelmh.cpp > CMakeFiles/indelmh.dir/indelmh.cpp.i

CMakeFiles/indelmh.dir/indelmh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/indelmh.dir/indelmh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelmh.cpp -o CMakeFiles/indelmh.dir/indelmh.cpp.s

# Object files for target indelmh
indelmh_OBJECTS = \
"CMakeFiles/indelmh.dir/indel.cpp.o" \
"CMakeFiles/indelmh.dir/oligo.cpp.o" \
"CMakeFiles/indelmh.dir/readmap.cpp.o" \
"CMakeFiles/indelmh.dir/gen.cpp.o" \
"CMakeFiles/indelmh.dir/indelmh.cpp.o"

# External object files for target indelmh
indelmh_EXTERNAL_OBJECTS =

indelmh: CMakeFiles/indelmh.dir/indel.cpp.o
indelmh: CMakeFiles/indelmh.dir/oligo.cpp.o
indelmh: CMakeFiles/indelmh.dir/readmap.cpp.o
indelmh: CMakeFiles/indelmh.dir/gen.cpp.o
indelmh: CMakeFiles/indelmh.dir/indelmh.cpp.o
indelmh: CMakeFiles/indelmh.dir/build.make
indelmh: CMakeFiles/indelmh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable indelmh"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/indelmh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/indelmh.dir/build: indelmh
.PHONY : CMakeFiles/indelmh.dir/build

CMakeFiles/indelmh.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/indelmh.dir/cmake_clean.cmake
.PHONY : CMakeFiles/indelmh.dir/clean

CMakeFiles/indelmh.dir/depend:
	cd /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/CMakeFiles/indelmh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/indelmh.dir/depend
