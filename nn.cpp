#include "nn.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/graph/default_device.h"
// #include <omp.h>

using namespace std;
using namespace tensorflow;
using namespace Eigen;

typedef Eigen::Tensor<int, 2, RowMajor> Tensor2i;
typedef Eigen::Tensor<int, 1, RowMajor> Tensor1i;

std::function<int(int,int)> mod = [](int a, int b) { return a % b; };

Tensor1i my_ravel_multi_index(Tensor2i inds, Tensor1i dims) {
    Eigen::array<bool, 1> reverse({true});
    Eigen::array<int, 2> reshape{{dims.size(), 1}};
    Eigen::array<int, 2> bcast{{1, inds.dimension(1)}};
    Eigen::array<int, 1> sum_dim({0});

    auto elems = dims.reverse(reverse).cumprod(0, true).reverse(reverse)
        .reshape(reshape).broadcast(bcast);
    return (inds * elems).sum(sum_dim);
}

Tensor2i my_unravel_index(Tensor1i inds, Tensor1i dims) {
    // np.unravel_index?
    // >>> np.unravel_index([22, 41, 37], (7,6))
    // (array([3, 6, 6]), array([4, 5, 1]))
    Tensor1i res(inds);

    Eigen::array<bool, 1> reverse({true});
    Eigen::array<int, 2> reshape{{1, inds.size()}};
    Eigen::array<int, 2> bcast{{dims.size(), 1}};
    Eigen::array<int, 2> elems_reshape{{dims.size(), 1}};
    Eigen::array<int, 2> elems_bcast{{1, inds.size()}};

    auto elems = dims.reverse(reverse).cumprod(0, false).reverse(reverse)
        .reshape(elems_reshape).broadcast(elems_bcast);
    auto elems_shift = dims.reverse(reverse).cumprod(0, true).reverse(reverse)
        .reshape(elems_reshape).broadcast(elems_bcast);
    return res.reshape(reshape).broadcast(bcast).binaryExpr(elems, mod)  / elems_shift;
}

// {
//     Tensor2i inds(4,1);
//     inds.setValues({{4},{6},{5},{7}});
//     Tensor1i dims(4);
//     dims.setValues({8, 8, 11, 8});
//     LOG(INFO) << my_ravel_multi_index(inds, dims) << endl;
// }

// {
//     Tensor1i a(3);
//     Tensor1i b(3);
//     a.setValues({1,2,3});
//     b.setValues({4,5,6});
//     LOG(INFO) << a.dot(b);
// }

// >>> np.unravel_index([22, 41, 37], (7,6))
// (array([3, 6, 6]), array([4, 5, 1]))
// {
//     Tensor1i inds(4);
//     inds.setValues({22, 41, 37, 5});
//     Tensor1i dims(2);
//     dims.setValues({7, 6});
//     LOG(INFO) << my_unravel_index(inds, dims) << endl;
// }

// {
//     Tensor1i inds(1);
//     inds.setValues({3391});
//     Tensor1i dims(4);
//     dims.setValues({8,8,11,8});
//     LOG(INFO) << my_unravel_index(inds, dims) << endl;
// }

SlonikNet::SlonikNet() {
    Session* session;
    GraphDef graph_def;
    SessionOptions opts;

    TF_CHECK_OK(ReadBinaryProto(Env::Default(), "/home/maksle/projects/test/models/train3.pb", &graph_def));
    TF_CHECK_OK(NewSession(opts, &session));
    TF_CHECK_OK(session->Create(graph_def));
    TF_CHECK_OK(session->Run({}, {}, {"init"}, nullptr));

    tensorflow::Tensor x(DT_FLOAT, TensorShape({100, 32}));
    tensorflow::Tensor y(DT_FLOAT, TensorShape({100, 8}));
    // auto _XTensor = x.matrix<float>();
    // auto _YTensor = y.matrix<float>();
    auto _XTensor = x.tensor<float, 2>();
    auto _YTensor = y.tensor<float, 2>();

    _XTensor.setRandom();
    _YTensor.setRandom();

    // #pragma omp parallel for
    // for (int i = 0; i < 25; ++i) {
    //     std::vector<Tensor> outputs;
    //     TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs));
    //     float cost = outputs[0].scalar<float>()(0);
    //     std::cout << "Cost: " << cost << std::endl;
    //     TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr));
    //     outputs.clear();
    // }

    // session->Close();
    // delete session;
    // return 0;
}
