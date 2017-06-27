#include "nn.h"
#include <sys/stat.h>
#include "types.h"
#include "position.h"
#include "features.h"

using namespace mxnet::cpp;
using namespace std;

Symbol SlonikNet::feature_group(std::string name, int hidden_size) {
    auto input_data = Symbol::Variable(name + "_data");

    auto fc = Operator("FullyConnected")
        .SetParam("num_hidden", hidden_size)
        .SetInput("data", input_data)
        .CreateSymbol(name + "_hidden_layer");

    auto output = Operator("Activation")
        .SetParam("act_type", "relu")
        .SetInput("data", fc)
        .CreateSymbol(name + "_activation");

    return output;
}

Symbol SlonikNet::build_net() {
    Symbol hidden_global = feature_group("global", hidden_global_size);
    Symbol hidden_pawn = feature_group("pawn", hidden_pawn_size);
    Symbol hidden_piece = feature_group("piece", hidden_piece_size);
    Symbol hidden_square = feature_group("square", hidden_square_size);
    
    vector<Symbol> parts { hidden_global, hidden_pawn, hidden_piece, hidden_square };
    auto merged = Concat("merged", parts, parts.size(), 1);
    auto fc = Operator("FullyConnected")
        .SetParam("num_hidden", hidden_shared_size)
        .SetInput("data", merged)
        .CreateSymbol("fully_connected");
    auto fc_act = Operator("Activation")
        .SetParam("act_type", "relu")
        .SetInput("data", fc)
        .CreateSymbol("fully_connected_activation");
    auto V_prelim = Operator("FullyConnected")
        .SetParam("num_hidden", 1)
        .SetInput("data", fc_act)
        .CreateSymbol("V_prelim");
    // auto V = Operator("Activation")
    //     .SetParam("act_type", "tanh")
    //     .SetInput("data", V_prelim)
    //     .CreateSymbol("V");
    return V_prelim;
}

SlonikNet::SlonikNet() {
    v = build_net();
    target = Symbol::Variable("label");

    loss = Operator("LinearRegressionOutput")
        .SetInput("data", v)
        .SetInput("label", target)
        .CreateSymbol("loss");
    
    make_args_map(1, predict_maps);
    load_args_map(predict_maps.args_map);

    make_args_map(32, train_maps);
    load_args_map(train_maps.args_map);
}

void SlonikNet::make_args_map(int batch_size,
                              Slonik::NNMapsContainer& maps) {
    
    maps.args_map["global_data"] = NDArray(Shape(batch_size, input_global_size), Context::cpu());
    maps.args_map["pawn_data"] = NDArray(Shape(batch_size, input_pawn_size), Context::cpu());
    maps.args_map["piece_data"] = NDArray(Shape(batch_size, input_piece_size), Context::cpu());
    maps.args_map["square_data"] = NDArray(Shape(batch_size, input_square_size), Context::cpu());
    maps.args_map["label"] = NDArray(Shape(batch_size), Context::cpu());
    
    for (const auto &name : loss.ListArguments()) {
        if (name == "label")
            maps.grad_req_type[name] = OpReqType::kNullOp;
        else
            maps.grad_req_type[name] = OpReqType::kWriteTo;
    } 
    
    v.InferArgsMap(Context::cpu(), &(maps.args_map), maps.args_map);
}

void SlonikNet::load_args_map(std::map<std::string, NDArray>& args_map) {
    string name = "./model/slonik_net";
    struct stat buffer;
    if (stat(name.c_str(), &buffer) == 0)
    {
        cout << "Loading args_map from file\n";
        NDArray::Load(name, nullptr, &args_map);
    }
    else
    {
        cout << "Loading args_map with Normal distribution\n";
        Normal dist = Normal(0, 1);
        for (auto &arg : args_map) {
            dist(arg.first, &arg.second);
        }
    }
}

void SlonikNet::load_inputs_from_position(const Position& pos, vector<float> targets) {
    auto fe = FeatureExtractor();
    fe.set_position(pos);
    auto features = fe.extract();

    std::vector<float> global = features[0];
    std::vector<float> pawns = features[1];
    std::vector<float> pieces = features[2];
    std::vector<float> squares = features[3];

    cout << "[";
    for (auto& f : pieces) {
        cout << f << ", ";
    }
    cout << "]\n";

    LG << "global size " << global.size();
    LG << "pawns size " << pawns.size();
    LG << "pieces size " << pieces.size();
    LG << "squares size " << squares.size();
    
    predict_maps.args_map["global_data"].SyncCopyFromCPU(global.data(), global.size());
    predict_maps.args_map["pawn_data"].SyncCopyFromCPU(pawns.data(), pawns.size());
    predict_maps.args_map["piece_data"].SyncCopyFromCPU(pieces.data(), pieces.size());
    predict_maps.args_map["square_data"].SyncCopyFromCPU(squares.data(), squares.size());

    // if (targets.size() > 0) 
    //     train_maps.args_map["label"].SyncCopyFromCPU(targets.data(), targets.size());

    NDArray::WaitAll();
}

float SlonikNet::evaluate(const Position& pos) {
    load_inputs_from_position(pos);
    return forward_only();
}

void SlonikNet::train() {
    train(train_maps);
}

void SlonikNet::train(Slonik::NNMapsContainer& maps) {
    Optimizer* opt = OptimizerRegistry::Find("adam");
    unique_ptr<Executor> exec { loss.SimpleBind(Context::cpu(), maps.args_map, maps.arg_grad_store, maps.grad_req_type, maps.aux_map) };

    float learning_rate = 3e-5;
    float weight_decay = 1e-4; // what should this be
    
    exec->Forward(true);
    exec->Backward();
    exec->UpdateAll(opt, learning_rate, weight_decay);
    NDArray::WaitAll();
}

float SlonikNet::forward_only() {
    Optimizer* opt = OptimizerRegistry::Find("adam");
    unique_ptr<Executor> exec { v.SimpleBind(Context::cpu(),
                                             predict_maps.args_map, predict_maps.arg_grad_store,
                                             predict_maps.grad_req_type, predict_maps.aux_map) };
    float res[1];
    
    // LG << args_map["global_data"];
    // LG << args_map["pawn_data"];

    // for (auto& arg: v.ListArguments()) {
    //     LG << arg;
    // }
    
    // LG << exec->DebugStr();
    
    exec->Forward(false);
    (exec->outputs[0]).SyncCopyToCPU(res, 1);
    // NDArray::WaitAll();
    return *res;
}
