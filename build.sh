CPP="bb.cpp features.cpp magics.cpp main.cpp material.cpp mcts.cpp movecalc.cpp movegen.cpp nn.cpp position.cpp search.cpp sts.cpp template.cpp tt.cpp uci.cpp zobrist.cpp"
TENSORFLOW_DIR=/home/maksle/projects/cpp_tensorflow/tensorflow

LD_LIBRARY_PATH=${TENSORFLOW_DIR}/tensorflow/loader
g++ --std=c++11 ${CPP} -o main -O3 -Wfatal-errors \
-I ${TENSORFLOW_DIR}/ \
-I ${TENSORFLOW_DIR}/bazel-genfiles \
-I ${TENSORFLOW_DIR}/protobuf/protobuf-3.5.0/src \
-I ${TENSORFLOW_DIR}/eigen \
-L ${LD_LIBRARY_PATH} -Wl,-R,${LD_LIBRARY_PATH} -ltensorflow_cc \
-L ${TENSORFLOW_DIR}/protobuf/protobuf-3.5.0/src/.libs -lprotobuf
