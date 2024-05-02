# SVD4Rec: Метод сингулярного разложения для рекомендательных систем

VD4Rec - это библиотека на C++, которая реализует модель латентных факторов (LFM) с использованием метода сингулярного разложения (SVD) для создания рекомендательных систем. Эта библиотека предоставляет интерфейс на Python с использованием Pybind11, позволяя вам эффективно обучать модели рекомендаций и проводить настройку гиперпараметров.

Прежде чем использовать SVD4Rec, вам необходимо скомпилировать C++ код и создать модуль на Python. Следуйте этим шагам:

1. Склонировать репозийторий

```
git clone https://github.com/rootinshik/SVD4Rec.git
cd SVD4Rec
```
2. Скомпилировать код:

```
python setup.py build_ext -i
```

## Использование на Python

После компиляции библиотеки, вы можете использовать ее на Python следующим образом:

```
import SVD4Rec

# Загрузите матрицу взаимодействия user-item (numpy.ndarray)
# Пример:
R = user_item_matrix 

# Обучите систему рекомендаций с использованием LFM_SGD
P, Q, params = SVD4Rec.LFM_SGD(R)

# Вы также можете указать гиперпараметры
# P, Q, params = SVD4Rec.LFM_SGD(R, epsilon=0.01, numIterations=5000, latentFactors=5, learningRate=0.0003, regularization=0.5, batchSize=50)

# Проведите настройку гиперпараметров с использованием grid search cv
# Это вернет лучшие гиперпараметры и соответствующие матрицы P и Q
best_P, best_Q, best_params = SVD4Rec.tuneHyperparameters(R, numTrials=10, numIterationsRange=1000,
                                                          latentFactorsRange=10, learningRateMin=0.0001,
                                                          learningRateMax=0.001, regularizationMin=0.1,
                                                          regularizationMax=1.0, batchSizeMin=10,
                                                          batchSizeMax=100)
```
