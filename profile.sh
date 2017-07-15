# run with -pg option
g++ -g -O0 -pg -Wall -o main -std=c++11 *.cpp -I package -lpthread -lmxnet
# run the program normally
./main
# run gprof
gprof main > prof
