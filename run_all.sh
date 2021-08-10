#!/usr/bin/env bash

python3 compare_ml_models.py we sgd "balanced_accuracy" &  python3 compare_ml_models.py we rf "balanced_accuracy" &  python3 compare_ml_models.py we lg "balanced_accuracy"  &  python3 compare_ml_models.py we svm "balanced_accuracy" &

python3 compare_ml_models.py we sgd -f "balanced_accuracy" &  python3 compare_ml_models.py we rf -f "balanced_accuracy" &  python3 compare_ml_models.py we lg -f "balanced_accuracy"  &  python3 compare_ml_models.py we svm -f "balanced_accuracy" &

python3 compare_ml_models.py fea sgd "balanced_accuracy" &  python3 compare_ml_models.py fea rf "balanced_accuracy" &  python3 compare_ml_models.py fea lg "balanced_accuracy"  &  python3 compare_ml_models.py fea svm "balanced_accuracy" &

wait
