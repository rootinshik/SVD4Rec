#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>

namespace py = pybind11;

py::tuple LFM_SGD(const py::array_t<double>&, double, int, int, double, double);
double scal_prod(const double*, const double*, int, int, int);

py::tuple LFM_SGD(const py::array_t<double>& R, double eps = .01, 
        int num_iter = 5000, int k = 5, double lr = .0003, double reg = .5, int batch_size = 50) {
    py::buffer_info bufInfoR = R.request();
    int num_users = bufInfoR.shape[0];
    int num_items = bufInfoR.shape[1];

    py::array_t<double> P({num_users, k});
    py::array_t<double> Q({num_items, k});

    py::buffer_info bufInfoP = P.request();
    py::buffer_info bufInfoQ = Q.request();

    double* ptrR = static_cast<double*>(bufInfoR.ptr);
    double* ptrP = static_cast<double*>(bufInfoP.ptr);
    double* ptrQ = static_cast<double*>(bufInfoQ.ptr);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 5);
    for (int i = 0; i < num_users * k; i++) {
        ptrP[i] = dis(gen);
    }
    for (int i = 0; i < num_items * k; i++) {
        ptrQ[i] = dis(gen);
    }

    for (int iter = 0; iter < num_iter; iter++) {
        double loss = 0;
        for (int batch = 0; batch < num_users / batch_size; batch++) {

            std::vector<int> user_indices;
            for (int i = 0; i < batch_size; i++) {
                int user = batch * batch_size + i;
                if (user >= num_users) break;
                user_indices.push_back(user);
            }
            std::shuffle(user_indices.begin(), user_indices.end(), gen);

            for (int i = 0; i < user_indices.size(); i++) {
                int user = user_indices[i];

                for (int item = 0; item < num_items; item++) {
                    int rating_idx = user * num_items + item;

                    if (std::isnan(ptrR[rating_idx])) continue;
                    double err_ui = ptrR[rating_idx] - scal_prod(ptrP, ptrQ, user, item, k);

                    for (int j = 0; j < k; j++) {
                        ptrP[user * k + j] += lr * (2 * err_ui * ptrQ[item * k + j] - reg/2 * ptrP[user * k + j]);
                        ptrQ[item * k + j] += lr * (2 * err_ui * ptrP[user * k + j] - reg/2 * ptrQ[item * k + j]);
                    }

                    loss += std::pow((ptrR[rating_idx] - scal_prod(ptrP, ptrQ, user, item, k)), 2);
                }
            }
        }
        if (loss < eps) break;
    }

    return py::make_tuple(P, Q);
}


double scal_prod(const double* ptrP, const double* ptrQ, int user, int item, int k) {
    double result = 0.0;
    for (int i = 0; i < k; i++) {
        result += ptrP[user * k + i] * ptrQ[item * k + i];
    }
    return result;
}

PYBIND11_MODULE(SVD4Rec, m) {
    m.def("LFM_SGD", static_cast<py::tuple(*)(const py::array_t<double>&, double, int, int, double, double, int)>(&LFM_SVD),
          "Latent Factor Model with Singular Value Decomposition",
          py::arg("R"), py::arg("eps") = .01, py::arg("num_iter") = 5000, 
          py::arg("k") = 2, py::arg("lr") = .0003, py::arg("reg") = .5, py::arg("batch_size") = 50);
}