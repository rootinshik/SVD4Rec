#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple> 

namespace py = pybind11;

// Structure to store hyperparameters
struct Hyperparameters {
    double epsilon;
    int numIterations;
    int latentFactors;
    double learningRate;
    double regularization;
    int batchSize;
    double loss; 
};

// Function to compute the dot product of two vectors
double dotProduct(const double* vectorA, const double* vectorB, int user, int item, int latentFactors) {
    double result = 0.0;
    for (int i = 0; i < latentFactors; i++) {
        result += vectorA[user * latentFactors + i] * vectorB[item * latentFactors + i];
    }
    return result;
}

// Function to calculate Mean Squared Error (MSE) loss
double calculateLoss(const py::array_t<double>& R, const py::array_t<double>& P, const py::array_t<double>& Q) {
    auto R_info = R.request();
    auto P_info = P.request();
    auto Q_info = Q.request();

    int numUsers = R_info.shape[0];
    int numItems = R_info.shape[1];
    int latentFactors = P_info.shape[1];

    const double* ptrR = static_cast<const double*>(R_info.ptr);
    const double* ptrP = static_cast<const double*>(P_info.ptr);
    const double* ptrQ = static_cast<const double*>(Q_info.ptr);

    double totalLoss = 0.0;
    int numRatings = 0;

    for (int user = 0; user < numUsers; user++) {
        for (int item = 0; item < numItems; item++) {
            int ratingIndex = user * numItems + item;
            double actualRating = ptrR[ratingIndex];

            if (!std::isnan(actualRating)) {
                double predictedRating = dotProduct(ptrP, ptrQ, user, item, latentFactors);
                double error = actualRating - predictedRating;
                totalLoss += error * error;
                numRatings++;
            }
        }
    }

    if (numRatings > 0) {
        double mse = totalLoss / numRatings;
        return mse;
    } else {
        // No valid ratings found
        return 0.0;
    }
}

// Function to perform LFM_SGD
std::tuple<py::array_t<double>, py::array_t<double>, Hyperparameters> LFM_SGD(const py::array_t<double>& R, double epsilon = 0.01,
                  int numIterations = 5000, int latentFactors = 5, double learningRate = 0.0003,
                  double regularization = 0.5, int batchSize = 50) {
    auto R_info = R.request();
    int numUsers = R_info.shape[0];
    int numItems = R_info.shape[1];

    py::array_t<double> P({numUsers, latentFactors});
    py::array_t<double> Q({numItems, latentFactors});

    auto P_info = P.request();
    auto Q_info = Q.request();

    double* ptrR = static_cast<double*>(R_info.ptr);
    double* ptrP = static_cast<double*>(P_info.ptr);
    double* ptrQ = static_cast<double*>(Q_info.ptr);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 5);

    for (int i = 0; i < numUsers * latentFactors; i++) {
        ptrP[i] = dis(gen);
    }
    for (int i = 0; i < numItems * latentFactors; i++) {
        ptrQ[i] = dis(gen);
    }

    double loss = 0;
    for (int iter = 0; iter < numIterations; iter++) {
        loss = 0;

        std::vector<int> userIndices(numUsers);
        for (int i = 0; i < numUsers; i++) {
            userIndices[i] = i;
        }
        std::shuffle(userIndices.begin(), userIndices.end(), gen);

        for (int start = 0; start < numUsers; start += batchSize) {
            int end = std::min(start + batchSize, numUsers);

            for (int batch = start; batch < end; batch++) {
                int user = userIndices[batch];

                for (int item = 0; item < numItems; item++) {
                    int ratingIndex = user * numItems + item;
                    double actualRating = ptrR[ratingIndex];

                    if (!std::isnan(actualRating)) {
                        double error_ui = actualRating - dotProduct(ptrP, ptrQ, user, item, latentFactors);

                        for (int j = 0; j < latentFactors; j++) {
                            ptrP[user * latentFactors + j] += learningRate * (2 * error_ui * ptrQ[item * latentFactors + j] - regularization / 2 * ptrP[user * latentFactors + j]);
                            ptrQ[item * latentFactors + j] += learningRate * (2 * error_ui * ptrP[user * latentFactors + j] - regularization / 2 * ptrQ[item * latentFactors + j]);
                        }

                        loss += error_ui * error_ui;
                    }
                }
            }
        }
        loss /= (numUsers * numItems);
        if (loss < epsilon) break;
    }

    Hyperparameters hyperparameters{epsilon, numIterations, latentFactors, learningRate, regularization, batchSize, loss};

    return std::make_tuple(P, Q, hyperparameters);
}

// Function for hyperparameter tuning using randomized cross-validation
std::tuple<py::array_t<double>, py::array_t<double>, Hyperparameters> tuneHyperparameters(const py::array_t<double>& R,
                               int numTrials = 10,
                               int numIterationsRange = 1000,
                               int latentFactorsRange = 10,
                               double learningRateMin = 0.0001,
                               double learningRateMax = 0.001,
                               double regularizationMin = 0.1,
                               double regularizationMax = 1.0,
                               int batchSizeMin = 10,
                               int batchSizeMax = 100) {
    auto R_info = R.request();
    int numUsers = R_info.shape[0];
    int numItems = R_info.shape[1];

    double bestLoss = std::numeric_limits<double>::max();
    std::tuple<py::array_t<double>, py::array_t<double>, Hyperparameters> bestResult;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int trial = 0; trial < numTrials; trial++) {
        int numIterations = std::uniform_int_distribution<>(1, numIterationsRange)(gen);
        int latentFactors = std::uniform_int_distribution<>(1, latentFactorsRange)(gen);
        double learningRate = std::uniform_real_distribution<>(learningRateMin, learningRateMax)(gen);
        double regularization = std::uniform_real_distribution<>(regularizationMin, regularizationMax)(gen);
        int batchSize = std::uniform_int_distribution<>(batchSizeMin, batchSizeMax)(gen);

        auto result = LFM_SGD(R, 0.0001, numIterations, latentFactors, learningRate, regularization, batchSize);
        double loss = calculateLoss(R, std::get<0>(result), std::get<1>(result));

        if (loss < bestLoss) {
            bestLoss = loss;
            bestResult = result;
        }
    }

    return bestResult;
}

// Define the Python module
PYBIND11_MODULE(SVD4Rec, m) {
    m.def("LFM_SGD", &LFM_SGD, "Latent Factor Model with Singular Value Decomposition",
          py::arg("R"), py::arg("epsilon") = 0.01, py::arg("numIterations") = 5000,
          py::arg("latentFactors") = 5, py::arg("learningRate") = 0.0003, py::arg("regularization") = 0.5, py::arg("batchSize") = 50);

    m.def("tuneHyperparameters", &tuneHyperparameters, "Hyperparameter tuning using randomized cross-validation",
          py::arg("R"), py::arg("numTrials") = 10, py::arg("numIterationsRange") = 1000,
          py::arg("latentFactorsRange") = 10, py::arg("learningRateMin") = 0.0001, py::arg("learningRateMax") = 0.001,
          py::arg("regularizationMin") = 0.1, py::arg("regularizationMax") = 1.0,
          py::arg("batchSizeMin") = 10, py::arg("batchSizeMax") = 100);
}
