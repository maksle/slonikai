# g++ --std=c++11 *.cpp -o main -Wfatal-errors -I package -lpthread -lmxnet
g++ --std=c++11 ./libmxnet.so *.cpp -o main -Wfatal-errors -I package -lpthread
