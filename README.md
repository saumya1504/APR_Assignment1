PCA + KNN on MNIST

A compact reproducible implementation demonstrating how Principal Component Analysis (PCA) can be used to compress MNIST digit images before classification with K-Nearest Neighbors (KNN) — with direct dataset loading from OpenML (and a tensorflow.keras fallback). 

How it works (short)

Load MNIST
fetch_openml('mnist_784') is used to fetch MNIST (70k samples). If OpenML is unavailable, the code falls back to tensorflow.keras.datasets.mnist.

Pipelines

Baseline pipeline: StandardScaler() → KNeighborsClassifier()

PCA pipeline: StandardScaler() → PCA(n_components=k) → KNeighborsClassifier()

Model selection
GridSearchCV is used to tune KNN hyperparameters (n_neighbors, weights) for both pipelines. For speed, the default grid is small; expand as desired.

Evaluation & visualization

Accuracy, classification report, and confusion matrix are printed for both approaches.

A small set of test images is shown original vs PCA-reconstructed (using pca.inverse_transform) so you can visually inspect information loss.

PCA components (reshaped to 28×28) are plotted to show the principal directions learned.

What you will see / outputs

Console output containing:

Best GridSearchCV parameters and CV accuracy

Test accuracy for both models

Classification reports

Saved files in results/ (or results_visual/ depending on the script version):

cv_results.csv

best_pipeline.joblib (serialized best estimator)

confusion_matrix.png and explained_variance_cumulative.png (plots)

knn_no_pca.joblib, knn_pca_k{K}.joblib (visual experiment script)

Visual plots:

Original images (from test set) vs PCA reconstructions — useful to see how compression affects digits.

First N PCA components displayed as 28×28 images (they look like “basis strokes”).

Confusion matrices for both pipelines (per-class errors).
