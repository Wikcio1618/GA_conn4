cmake_minimum_required(VERSION 3.16)
project(main)

# Find LibTorch (adjust the path accordingly)
find_package(Torch CONFIG REQUIRED PATHS "./libtorch/share/cmake/Torch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add your source files and headers
set(SOURCE_FILES
    src/main.cpp
    src/conn4model.cpp
    src/population.cpp
    src/board.cpp
)

set(HEADER_FILES
    include/board.h
    include/population.h
    include/conn4model.h
)

# Create an executable
add_executable(main ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(main "${TORCH_LIBRARIES}")

# Include directories for headers
target_include_directories(main PRIVATE include)

# Set additional compiler options if needed
target_compile_options(main PRIVATE -O2 -Wall -Wextra)
