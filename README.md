# SVD4Rec: Singular Value Decomposition for Recommender Systems

SVD4Rec is a C++ library that implements a Latent Factor Model (LFM) with Singular Value Decomposition (SVD) for building recommender systems. This library provides a Python interface using Pybind11, allowing you to efficiently train recommendation models and perform hyperparameter tuning.
Compilation

Before using SVD4Rec, you need to compile the C++ code and create a Python module. Follow these steps:

```
git clone https://github.com/rootinshik/SVD4Rec.git
cd SVD4Rec
```
Compile the Code:

```
python setup.py build_ext -i
```

## Python Usage

Once you've compiled the library, you can use it in Python as follows:

```

import SVD4Rec

# Load your user-item interaction matrix R as a numpy array
# Example:
# R = your_data_loading_function()

# Train the recommender system using LFM_SGD
P, Q, hyperparameters = SVD4Rec.LFM_SGD(R)

# You can also specify hyperparameters
# P, Q, hyperparameters = SVD4Rec.LFM_SGD(R, epsilon=0.01, numIterations=5000, latentFactors=5, learningRate=0.0003, regularization=0.5, batchSize=50)

# Perform hyperparameter tuning using randomized cross-validation
# This will return the best hyperparameters and corresponding P and Q matrices
best_P, best_Q, best_hyperparameters = SVD4Rec.tuneHyperparameters(R, numTrials=10, numIterationsRange=1000,
                                                                  latentFactorsRange=10, learningRateMin=0.0001,
                                                                  learningRateMax=0.001, regularizationMin=0.1,
                                                                  regularizationMax=1.0, batchSizeMin=10,
                                                                  batchSizeMax=100)
```

## Example Data

You should prepare your user-item interaction data as a numpy array `R` where rows represent users, columns represent items, and the values represent user-item interactions. Replace your_data_loading_function() with your actual data loading code.
Library Documentation

Enjoy using SVD4Rec for building and tuning recommender systems!