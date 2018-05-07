#ifndef SLONIK_NN_H_GUARD
#define SLONIK_NN_H_GUARD

#include <string>
#include <map>
#include <vector>
/* #include "eval.h" */
/* #include "package/mxnet-cpp/MxNetCpp.h" */

class Position;

typedef std::vector<std::vector<float>> Features;

/* namespace Slonik { */
/*   struct NNMapsContainer { */
/*     std::map<std::string, mxnet::cpp::NDArray> args_map; */
/*     std::map<std::string, mxnet::cpp::NDArray> arg_grad_store; */
/*     std::map<std::string, mxnet::cpp::OpReqType> grad_req_type; */
/*     std::map<std::string, mxnet::cpp::NDArray> aux_map; */

/*     /\* For executor *\/ */
/*     std::vector<mxnet::cpp::NDArray> arg_arrays; */
/*     std::vector<mxnet::cpp::NDArray> grad_arrays; */
/*     std::vector<mxnet::cpp::OpReqType> grad_reqs; */
/*     std::vector<mxnet::cpp::NDArray> aux_arrays; */
/*   }; */
/* } */

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
  
  /* mxnet::cpp::Symbol v; */
  /* mxnet::cpp::Symbol target; */
  /* mxnet::cpp::Symbol loss; */
  
  /* mxnet::cpp::Symbol build_net(); */
  /* mxnet::cpp::Symbol feature_group(std::string name, int hidden_size); */
  void load_inputs_from_position(const Position& pos,
                                 std::vector<float> targets = std::vector<float>());

  void load_data(std::vector<Features> features, std::vector<float> targets);
  
 public:

  /* std::unique_ptr<mxnet::cpp::Executor> executor; */
  /* std::unique_ptr<mxnet::cpp::Optimizer> optimizer; */
  
  SlonikNet();
  
  void save(std::string name);
  float validate(std::vector<Features> features, std::vector<float> targets);
  void fit(std::vector<Features> features, std::vector<float> targets);
  float forward_only();
  float evaluate(const Position& pos);
};

#endif
