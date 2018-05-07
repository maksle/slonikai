cmake_minimum_required(VERSION 3.5.1)
project(slonikai)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")

set (SOURCES bb.cpp features.cpp magics.cpp main.cpp material.cpp mcts.cpp movecalc.cpp movegen.cpp nn.cpp position.cpp search.cpp sts.cpp template.cpp tt.cpp uci.cpp zobrist.cpp)

set (TENSORFLOW_DIR /home/maksle/projects/cpp_tensorflow/tensorflow)
set (LD_LIBRARY_PATH ${TENSORFLOW_DIR}/tensorflow/loader)
set (PROJECT_LINK_LIBS libprotobuf.so libtensorflow_cc.so)

include_directories (${TENSORFLOW_DIR} ${TENSORFLOW_DIR}/bazel-genfiles ${TENSORFLOW_DIR}/protobuf/protobuf-3.5.0/src ${TENSORFLOW_DIR}/eigen)

link_directories (${LD_LIBRARY_PATH} ${TENSORFLOW_DIR}/protobuf/protobuf-3.5.0/src/.libs)

#file(GLOB SOURCES "src/*.cpp")
add_executable (main ${SOURCES})
target_link_libraries (main ${PROJECT_LINK_LIBS})
