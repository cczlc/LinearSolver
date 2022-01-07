cmake_minimum_required(VERSION 3.10)

find_package(CUDA)

if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

#list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O2;-DVERBOSE") 

# 配置一个头文件，将版本号、配置信息传递给源代码
configure_file(LinearSolverConfig.h.in LinearSolverConfig.h)

# Cuda文件筛选
file(GLOB_RECURSE CUDA_SOURCE "src/*.h" "src/*.cu")

# cpp文件筛选
file(GLOB_RECURSE CXX_SOURCE "src/*.cpp")

# 编译
cuda_add_executable(${PROJECT_NAME} ${CUDA_SOURCE} ${CXX_SOURCE} OPTIONS -arch=sm_72 -compute_72)

# 包含头文件搜索路径
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")