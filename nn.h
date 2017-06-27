#ifndef SLONIK_NN_H_GUARD
#define SLONIK_NN_H_GUARD

#include <string>
#include <map>
#include "package/mxnet-cpp/MxNetCpp.h"

class Position;

namespace Slonik {
  struct NNMapsContainer {
    std::map<std::string, mxnet::cpp::NDArray> args_map;
    std::map<std::string, mxnet::cpp::NDArray> arg_grad_store;
    std::map<std::string, mxnet::cpp::OpReqType> grad_req_type;
    std::map<std::string, mxnet::cpp::NDArray> aux_map;
  };
}

class SlonikNet {
 private:
  int input_global_size = 22;
  int hidden_global_size = 22;

  int input_pawn_size = 18;
  int hidden_pawn_size = 14;

  int input_piece_size = 68;
  int hidden_piece_size = 24;

  int input_square_size = 148;
  int hidden_square_size = 32;

  int hidden_shared_size = 64;
  
  mxnet::cpp::Symbol v;
  mxnet::cpp::Symbol target;
  mxnet::cpp::Symbol loss;
  
  Slonik::NNMapsContainer predict_maps;
  Slonik::NNMapsContainer train_maps;
  Slonik::NNMapsContainer other_maps;
  
  mxnet::cpp::Symbol build_net();
  mxnet::cpp::Symbol feature_group(std::string name, int hidden_size);
  void make_args_map(int batch_size, Slonik::NNMapsContainer& maps);
  void load_args_map(std::map<std::string, mxnet::cpp::NDArray>& args_map);
  void load_inputs_from_position(const Position& pos,
                                 std::vector<float> targets = std::vector<float>());

 public:
  SlonikNet();

  void train();
  void train(Slonik::NNMapsContainer& maps);
  float forward_only();
  float evaluate(const Position& pos);
};

#endif
