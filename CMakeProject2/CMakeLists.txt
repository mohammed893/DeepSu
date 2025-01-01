# CMakeList.txt : CMake project for CMakeProject2, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (CMakeProject2 "CMakeProject2.cpp" "CMakeProject2.h" "src/graph.cpp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeProject2 PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
