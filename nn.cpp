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

SlonikNet::SlonikNet()
{
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

void SlonikNet::set_batch_size(int batch_size)
{
    make_args_map(batch_size, train_maps);
    load_args_map(train_maps.args_map);
}

void SlonikNet::compile()
{
    Optimizer* opt = OptimizerRegistry::Find("adam");
    this->optimizer = std::unique_ptr<Optimizer>(opt);
    this->executor = std::unique_ptr<Executor> { loss.Bind(Context::cpu(), train_maps.arg_arrays, train_maps.grad_arrays, train_maps.grad_reqs, train_maps.aux_arrays) };
}

float SlonikNet::validate(vector<Features> features, vector<float> targets)
{
    vector<float> global;
    vector<float> pawn;
    vector<float> piece;
    vector<float> square;
    for (auto& pos_features : features)
    {
        move(pos_features[0].begin(), pos_features[0].end(), back_inserter(global));
        move(pos_features[1].begin(), pos_features[1].end(), back_inserter(pawn));
        move(pos_features[2].begin(), pos_features[2].end(), back_inserter(piece));
        move(pos_features[3].begin(), pos_features[3].end(), back_inserter(square));
    }
    
    // Optimizer* opt = OptimizerRegistry::Find("adam");
    // unique_ptr<Executor> exec { v.Bind(Context::cpu(),
    //                                    train_maps.arg_arrays, train_maps.grad_arrays, train_maps.grad_reqs, train_maps.aux_arrays) };
    
    mx_uint batch_size = train_maps.args_map["label"].GetShape()[0];

    mx_float rmse = 0.0f;
    int num_batches = 0;
    
    int i = 0, j = batch_size;
    while (j + batch_size <= targets.size())
    {
        train_maps.args_map["global_data"].SyncCopyFromCPU(global.data() + i, input_global_size * batch_size);
        train_maps.args_map["pawn_data"].SyncCopyFromCPU(pawn.data() + i, input_pawn_size * batch_size);
        train_maps.args_map["piece_data"].SyncCopyFromCPU(piece.data() + i, input_piece_size * batch_size);
        train_maps.args_map["square_data"].SyncCopyFromCPU(square.data() + i, input_square_size * batch_size);
        train_maps.args_map["label"].SyncCopyFromCPU(targets.data() + i, batch_size);
        
        executor->Forward(false);
        NDArray::WaitAll();
        
        NDArray preds = executor->outputs[0];
        std::vector<mx_float> preds_data;
        preds.SyncCopyToCPU(&preds_data);
        
        mx_float sum = 0;
        for (size_t k = 0; k < batch_size; ++k) {
            mx_float diff = preds_data[k] - targets[i + k];
            sum += diff * diff;
        }
        rmse += std::sqrt(sum / batch_size);
        ++num_batches;
        
        i = j;
        j += batch_size;
    }

    float accuracy = rmse / num_batches;
    return accuracy;
}

void SlonikNet::make_args_map(int batch_size, Slonik::NNMapsContainer& maps)
{
    maps.args_map["global_data"] = NDArray(Shape(batch_size, input_global_size), Context::cpu());
    maps.args_map["pawn_data"] = NDArray(Shape(batch_size, input_pawn_size), Context::cpu());
    maps.args_map["piece_data"] = NDArray(Shape(batch_size, input_piece_size), Context::cpu());
    maps.args_map["square_data"] = NDArray(Shape(batch_size, input_square_size), Context::cpu());
    maps.args_map["label"] = NDArray(Shape(batch_size), Context::cpu());
    
    for (const auto &name : loss.ListArguments()) {
        if (name == "label"/* || name.rfind("data") == name.length() - 4*/)
            maps.grad_req_type[name] = OpReqType::kNullOp;
        else
            maps.grad_req_type[name] = OpReqType::kWriteTo;
    } 
    
    // v.InferArgsMap(Context::cpu(), &(maps.args_map), maps.args_map);
    
    v.InferExecutorArrays(Context::cpu(),
                          &maps.arg_arrays, &maps.grad_arrays, &maps.grad_reqs, &maps.aux_arrays,
                          maps.args_map, maps.arg_grad_store, maps.grad_req_type, maps.aux_map);
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

    // cout << "[";
    // for (auto& f : pieces) {
    //     cout << f << ", ";
    // }
    // cout << "]\n";

    // LG << "global size " << global.size();
    // LG << "pawns size " << pawns.size();
    // LG << "pieces size " << pieces.size();
    // LG << "squares size " << squares.size();
    
    predict_maps.args_map["global_data"].SyncCopyFromCPU(global.data(), global.size());
    predict_maps.args_map["pawn_data"].SyncCopyFromCPU(pawns.data(), pawns.size());
    predict_maps.args_map["piece_data"].SyncCopyFromCPU(pieces.data(), pieces.size());
    predict_maps.args_map["square_data"].SyncCopyFromCPU(squares.data(), squares.size());

    // if (targets.size() > 0) 
    //     train_maps.args_map["label"].SyncCopyFromCPU(targets.data(), targets.size());

    NDArray::WaitAll();
}

void SlonikNet::load_data(std::vector<Features> features, std::vector<float> targets, Slonik::NNMapsContainer& maps) {
    vector<float> global;
    vector<float> pawn;
    vector<float> piece;
    vector<float> square;

    for (auto& pos_features : features)
    {
        move(pos_features[0].begin(), pos_features[0].end(), back_inserter(global));
        move(pos_features[1].begin(), pos_features[1].end(), back_inserter(pawn));
        move(pos_features[2].begin(), pos_features[2].end(), back_inserter(piece));
        move(pos_features[3].begin(), pos_features[3].end(), back_inserter(square));
    }

    maps.args_map["global_data"].SyncCopyFromCPU(global.data(), global.size());
    maps.args_map["pawn_data"].SyncCopyFromCPU(pawn.data(), pawn.size());
    maps.args_map["piece_data"].SyncCopyFromCPU(piece.data(), piece.size());
    maps.args_map["square_data"].SyncCopyFromCPU(square.data(), square.size());
    maps.args_map["label"].SyncCopyFromCPU(targets.data(), targets.size());
    NDArray::WaitAll();
}

void SlonikNet::train(vector<Features> features, vector<float> targets) {
    load_data(features, targets, train_maps);
    fit(train_maps);
}

float SlonikNet::evaluate(const Position& pos) {
    load_inputs_from_position(pos);
    return forward_only();
}

void SlonikNet::fit() { fit(train_maps); }

void SlonikNet::fit(Slonik::NNMapsContainer& maps)
{
    // Optimizer* opt = OptimizerRegistry::Find("adam");
    
    // unique_ptr<Executor> exec { loss.SimpleBind(Context::cpu(), maps.args_map, maps.arg_grad_store, maps.grad_req_type, maps.aux_map) };

    float learning_rate = 1e-4;
    float weight_decay = 0; // what should this be
    
    executor->Forward(true);
    executor->Backward();
    executor->UpdateAll(optimizer.get(), learning_rate, weight_decay);
    NDArray::WaitAll();
}

float SlonikNet::forward_only() {
    Optimizer* opt = OptimizerRegistry::Find("adam");
    unique_ptr<Executor> exec { v.Bind(Context::cpu(),
                                       predict_maps.arg_arrays, predict_maps.grad_arrays, predict_maps.grad_reqs, predict_maps.aux_arrays) };
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
