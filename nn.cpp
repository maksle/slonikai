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
    auto V = Operator("Activation")
        .SetParam("act_type", "tanh")
        .SetInput("data", V_prelim)
        .CreateSymbol("V");
    return V;
}

SlonikNet::SlonikNet()
{
    v = build_net();
    target = Symbol::Variable("label");

    loss = Operator("LinearRegressionOutput")
        .SetInput("data", v)
        .SetInput("label", target)
        .CreateSymbol("loss");
    
    map<string, NDArray> args_map;
    map<string, NDArray> aux_map;

    mx_uint batch_size = 32;
    
    args_map["global_data"] = NDArray(Shape(batch_size, input_global_size), Context::cpu());
    args_map["pawn_data"] = NDArray(Shape(batch_size, input_pawn_size), Context::cpu());
    args_map["piece_data"] = NDArray(Shape(batch_size, input_piece_size), Context::cpu());
    args_map["square_data"] = NDArray(Shape(batch_size, input_square_size), Context::cpu());
    args_map["label"] = NDArray(Shape(batch_size), Context::cpu());
    
    this->executor = std::unique_ptr<Executor> { loss.SimpleBind(Context::cpu(), args_map) };

    args_map = this->executor->arg_dict();
    aux_map = this->executor->aux_dict();
    
    auto arg_names = loss.ListArguments();
    
    string name = "./model/slonik_net.params";
    struct stat buffer;
    if (stat(name.c_str(), &buffer) == 0)
    {
        cout << "Loading args_map from file\n";
        NDArray::Load(name, nullptr, &args_map);
    }
    else
    {
        cout << "Loading args_map with Normal distribution\n";
        // Normal dist = Normal(0, 1);
        // auto he_init = Xavier(Xavier::RandType::gaussian, Xavier::FactorType::in, 2);
        auto he_init = Xavier();
        for (auto &arg : args_map) {
            he_init(arg.first, &arg.second);
        }
    }
    
    // for (const auto &s : arg_names) {
    //     LG << s;
    //     const auto &k = args_map[s].GetShape();
    //     for (const auto &i : k) {
    //         cout << i << " ";
    //     }
    //     cout << endl;
    // }
    
    Optimizer* opt = OptimizerRegistry::Find("adam");
    // Optimizer* opt = OptimizerRegistry::Find("ccsgd");
    opt->SetParam("lr", 1e-5)
        ->SetParam("wd", 1e-10)
        ->SetParam("rescale_grad", 1.0/batch_size);
    // opt->SetParam("lr", 1e-4)
    //     ->SetParam("momentum", 0.9)
    //     ->SetParam("rescale_grad", 1.0/batch_size)
    //     ->SetParam("clip_gradient", 10)
    //     ->SetParam("wd", 1e-9);
    
    this->optimizer = std::unique_ptr<Optimizer>(opt);
    
    // for (int i = 0; i < arg_names.size(); ++i) {
    //     LG << arg_names[i];
    //     LG << "arg_arrays: " << executor->arg_arrays[i];
    //     LG << "grad_arrays: " << executor->grad_arrays[i];
    // }
}

void SlonikNet::save(string name) {
    map<string, NDArray> params;
    for (const auto &n : this->executor->arg_dict()) {
        string name = n.first;
        if (name.rfind("data") == name.length() - 4 || name.rfind("label") == name.length() - 5)
            continue;
        params.insert(n);
    }
    string save_path = "./model/" + name;
    NDArray::Save(save_path, params);
}


float SlonikNet::validate(vector<Features> features, vector<float> targets)
{
    vector<float> global;
    vector<float> pawn;
    vector<float> piece;
    vector<float> square;
    // std::cout << "features:\n";
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << features[0][2][i] << " ";
    // }
    // std::cout << std::endl;
    
    for (auto& pos_features : features)
    {
        move(pos_features[0].begin(), pos_features[0].end(), back_inserter(global));
        move(pos_features[1].begin(), pos_features[1].end(), back_inserter(pawn));
        move(pos_features[2].begin(), pos_features[2].end(), back_inserter(piece));
        move(pos_features[3].begin(), pos_features[3].end(), back_inserter(square));
    }

    auto args_map = this->executor->arg_dict();
    
    mx_uint batch_size = args_map["label"].GetShape()[0];

    mx_float rmse = 0.0f;
    int num_batches = 0;
    
    int i = 0, j = batch_size;
    while (j + batch_size <= targets.size())
    {
        args_map["global_data"].SyncCopyFromCPU(global.data() + i, input_global_size * batch_size);
        args_map["pawn_data"].SyncCopyFromCPU(pawn.data() + i, input_pawn_size * batch_size);
        args_map["piece_data"].SyncCopyFromCPU(piece.data() + i, input_piece_size * batch_size);
        args_map["square_data"].SyncCopyFromCPU(square.data() + i, input_square_size * batch_size);
        args_map["label"].SyncCopyFromCPU(targets.data() + i, batch_size);
        
        executor->Forward(false);
        NDArray::WaitAll();
        NDArray preds = executor->outputs[0];

        // LG << "size " <<(executor->outputs).size();
        
        if ((i + j) == batch_size) {

            // LG << args_map["global_data"][0];
            // LG << args_map["pawn_data"][0];
            // LG << args_map["piece_data"][0];
            // LG << args_map["square_data"][0];
            // LG << args_map["label"][0];
            
            // LG << "preds:" << preds;

            // const auto& arg_names = loss.ListArguments();
            // for (int i = 0; i < arg_names.size(); ++i) {
            //     LG << arg_names[i];
            //     LG << "arg_arrays: " << executor->arg_arrays[i];
            //     LG << "grad_arrays: " << executor->grad_arrays[i];
            // }
            // LG << train_maps.args_map["label"];
        }
        
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

void SlonikNet::load_inputs_from_position(const Position& pos, vector<float> targets) {
    auto fe = FeatureExtractor();
    fe.set_position(pos);
    auto features = fe.extract();

    std::vector<float> global = features[0];
    std::vector<float> pawns = features[1];
    std::vector<float> pieces = features[2];
    std::vector<float> squares = features[3];
    
    auto args_map = this->executor->arg_dict();
    
    args_map["global_data"].SyncCopyFromCPU(global.data(), global.size());
    args_map["pawn_data"].SyncCopyFromCPU(pawns.data(), pawns.size());
    args_map["piece_data"].SyncCopyFromCPU(pieces.data(), pieces.size());
    args_map["square_data"].SyncCopyFromCPU(squares.data(), squares.size());
    
    NDArray::WaitAll();
}

void SlonikNet::load_data(std::vector<Features> features, std::vector<float> targets) {
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

    auto args_map = this->executor->arg_dict();
    
    args_map["global_data"].SyncCopyFromCPU(global.data(), global.size());
    args_map["pawn_data"].SyncCopyFromCPU(pawn.data(), pawn.size());
    args_map["piece_data"].SyncCopyFromCPU(piece.data(), piece.size());
    args_map["square_data"].SyncCopyFromCPU(square.data(), square.size());
    args_map["label"].SyncCopyFromCPU(targets.data(), targets.size());
    NDArray::WaitAll();
}

float SlonikNet::evaluate(const Position& pos) {
    load_inputs_from_position(pos);
    return forward_only();
}

void SlonikNet::fit(vector<Features> features, vector<float> targets)
{
    load_data(features, targets);
    // Optimizer* opt = OptimizerRegistry::Find("adam");
    
    // unique_ptr<Executor> exec { loss.SimpleBind(Context::cpu(), maps.args_map, maps.arg_grad_store, maps.grad_req_type, maps.aux_map) };

    // float lr = 1e-3;
    // float wd = 0; // what should this be
    
    executor->Forward(true);
    executor->Backward();
    
    // executor->UpdateAll(optimizer.get(), learning_rate, weight_decay);
    auto arg_names = loss.ListArguments();

    // for (int i = 0; i < arg_names.size(); ++i) {
    //     LG << arg_names[i];
    //     LG << "arg_arrays: " << executor->arg_arrays[i];
    //     LG << "grad_arrays: " << executor->grad_arrays[i];
    // }
    
    for (int i = 0; i < arg_names.size(); ++i) {
        std::string name = arg_names[i];
        if (name.rfind("data") == name.length() - 4 || name.rfind("label") == name.length() - 5)
            continue;
        // LG << "GRAD ARRAYS " << name << " " << executor->grad_arrays[i];
        optimizer->Update(i, executor->arg_arrays[i], executor->grad_arrays[i]);
    }
    NDArray::WaitAll();
}

float SlonikNet::forward_only() {
    // auto args_map = this->executor->arg_dict();
    // Optimizer* opt = OptimizerRegistry::Find("adam");
    // unique_ptr<Executor> exec { v.Bind(Context::cpu(), arg_arrays, grad_arrays, grad_reqs, aux_arrays) };
    float res[1];
    
    // LG << args_map["global_data"];
    // LG << args_map["pawn_data"];

    // for (auto& arg: v.ListArguments()) {
    //     LG << arg;
    // }
    
    // LG << exec->DebugStr();
    
    this->executor->Forward(false);
    (this->executor->outputs[0]).SyncCopyToCPU(res, 1);
    // NDArray::WaitAll();
    return *res;
}
