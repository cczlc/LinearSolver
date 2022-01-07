# 定义最小版本
cmake_minimum_required(VERSION 3.10)

# 定义工程名字
project(LinearSolver)

# 版本号
set(LinearSolver_VERSION_MAJOR 1)
set(LinearSolver_VERSION_MINOR 0)

# 库文件存放位置
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/library)
# 执行文件存放位置
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

# 设定c++14标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加编译选项
add_compile_options("-wall -O2 -release")
set(CMAKE_BUILD_TYPE "Release")

# 包含子 CMakeLists
add_subdirectory("${PROJECT_NAME}")
