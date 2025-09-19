PCA + KNN on MNIST

A compact reproducible implementation demonstrating how Principal Component Analysis (PCA) can be used to compress MNIST digit images before classification with K-Nearest Neighbors (KNN) — with direct dataset loading from OpenML (and a tensorflow.keras fallback). The repository contains a ready-to-run Python script / notebook that:

1.) loads MNIST automatically (OpenML → fallback to Keras),
2.) builds a scikit-learn Pipeline (StandardScaler → PCA → KNN),
3.) compares KNN with and without PCA,
4.) visualizes PCA reconstructions and the top PCA components,
5.) reports accuracy, classification reports and confusion matrices,
6.) saves best models and GridSearchCV results.
