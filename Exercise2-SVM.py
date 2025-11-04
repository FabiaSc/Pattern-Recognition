""" Simple SVM for Exercise 2 (MNIST). """

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Global variables
TRAIN_TSV = "gt-train.tsv"
TEST_TSV  = "gt-test.tsv"
IMAGE_SIZE = (28, 28)
SUBSET_SIZE = 10000       # subset for hyperparameter tuning 
PCA_COMPONENTS = 100      # PCA dimensions for RBF 
CV_FOLDS_SUBSET = 3       # CV when tuning on subset 
N_JOBS = 1                
RANDOM_STATE = 0

# Load Data
def load_indexed_images(tsv_path, image_size=(28,28)):
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['relpath','label'])
    base = Path(tsv_path).parent
    imgs = []
    labels = []
    for rel, lab in zip(df['relpath'], df['label']):
        p = base / rel
        with Image.open(p) as im:
            im = im.convert('L').resize(image_size)
            imgs.append(np.asarray(im, dtype=np.float32).ravel())
            labels.append(int(lab))
    return np.stack(imgs), np.array(labels, dtype=np.int32)

def stratified_subset(X, y, n_samples, random_state=RANDOM_STATE):
    if n_samples is None or n_samples >= len(y):
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]

# Main
def main():
    print("Loading data...")
    X_train, y_train = load_indexed_images(TRAIN_TSV, IMAGE_SIZE)
    X_test, y_test   = load_indexed_images(TEST_TSV, IMAGE_SIZE)
    print(f"Train {X_train.shape}, Test {X_test.shape}")

    # Scale Data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, "Results - SVM/scaler.joblib")

    # Make subset for fast tuning
    X_sub, y_sub = stratified_subset(X_train_s, y_train, SUBSET_SIZE)
    print(f"Using subset for CV: {X_sub.shape}")

    # linear tuning: SGDClassifier 
    print("\n1) Fast tuning linear SVM using SGDClassifier on subset...")
    sgd = SGDClassifier(loss='hinge', max_iter=3000, tol=1e-4, random_state=RANDOM_STATE)
    # alpha relates to regularization; alpha ≈ 1/(n_samples * C)
    sgd_grid = {'alpha': [1e-5, 5e-5, 1e-4, 5e-4]}
    sgd_gs = GridSearchCV(sgd, sgd_grid, cv=CV_FOLDS_SUBSET, n_jobs=N_JOBS, verbose=2, scoring='accuracy')
    sgd_gs.fit(X_sub, y_sub)
    best_alpha = sgd_gs.best_params_['alpha']
    print("SGD best alpha:", best_alpha, "CV score:", sgd_gs.best_score_)

    # Save CV results and plot (alpha vs mean CV accuracy)
    df_sgd = pd.DataFrame(sgd_gs.cv_results_)
    df_sgd.to_csv("cv_results_sgd.csv", index=False)
    plt.figure()
    alpha_vals = df_sgd['param_alpha'].astype(float)
    # aggregate by alpha (mean over splits)
    agg = df_sgd.groupby(alpha_vals)['mean_test_score'].mean().reset_index()
    plt.plot(agg.iloc[:,0], agg['mean_test_score'], marker='o')
    plt.xscale('log')
    plt.xlabel("alpha (log scale)")
    plt.ylabel("Mean CV Accuracy")
    plt.title("Linear SVM (SGD) CV Accuracy vs alpha")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cv_linear_alpha.png")
    plt.close()
    print("Saved cv_linear_alpha.png and cv_results_sgd.csv")

    # Convert alpha -> approx C for LinearSVC refit on full training set
    approx_C = 1.0 / (len(X_sub) * best_alpha)
    print(f"Approximate C computed from alpha: {approx_C:.6g}")

    # Refit LinearSVC on FULL training data 
    print("\n2) Refit LinearSVC on FULL training set with approx C (this may take a few minutes)...")
    final_linear = LinearSVC(C=approx_C, dual=False, max_iter=50000, tol=1e-4, random_state=RANDOM_STATE)
    final_linear.fit(X_train_s, y_train)
    joblib.dump(final_linear, "Results - SVM/svm_linear_final_full.joblib")
    print("Saved final LinearSVC: svm_linear_final_full.joblib")

    # RBF tuning: PCA on subset, tune RBF on PCA-subset 
    print(f"\n3) PCA (n_components={PCA_COMPONENTS}) on subset and RBF tuning on PCA-subset...")
    pca_sub = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', random_state=RANDOM_STATE)
    X_sub_p = pca_sub.fit_transform(X_sub)
    joblib.dump(pca_sub, "Results - SVM/pca_tuning.joblib")

    rbf = SVC(kernel='rbf')
    rbf_grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 0.01, 0.1]}
    rbf_gs = GridSearchCV(rbf, rbf_grid, cv=CV_FOLDS_SUBSET, n_jobs=N_JOBS, verbose=2, scoring='accuracy')
    rbf_gs.fit(X_sub_p, y_sub)
    print("RBF best params (on PCA-subset):", rbf_gs.best_params_, "CV score:", rbf_gs.best_score_)

    # Save CV results and plots for RBF
    df_rbf = pd.DataFrame(rbf_gs.cv_results_)
    df_rbf.to_csv("cv_results_rbf.csv", index=False)

    # Plot CV accuracy vs C (averaging over gamma)
    try:
        df_rbf['param_C_float'] = df_rbf['param_C'].astype(float)
        aggC = df_rbf.groupby('param_C_float')['mean_test_score'].mean().reset_index()
        plt.figure()
        plt.plot(aggC['param_C_float'], aggC['mean_test_score'], marker='o')
        plt.xscale('log')
        plt.xlabel("C (log scale)")
        plt.ylabel("Mean CV Accuracy")
        plt.title("RBF SVM (PCA) CV Accuracy vs C")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cv_rbf_C.png")
        plt.close()
    except Exception:
        # fallback: plot as string categories
        aggC = df_rbf.groupby('param_C')['mean_test_score'].mean().reset_index()
        plt.figure()
        plt.plot(aggC['param_C'].astype(str), aggC['mean_test_score'], marker='o')
        plt.xlabel("C")
        plt.ylabel("Mean CV Accuracy")
        plt.title("RBF SVM (PCA) CV Accuracy vs C")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cv_rbf_C.png")
        plt.close()

    # Plot CV accuracy vs gamma (averaging over C)
    agg_gamma = df_rbf.groupby('param_gamma')['mean_test_score'].mean().reset_index()
    plt.figure()
    plt.plot(agg_gamma['param_gamma'].astype(str), agg_gamma['mean_test_score'], marker='o')
    plt.xlabel("gamma")
    plt.ylabel("Mean CV Accuracy")
    plt.title("RBF SVM (PCA) CV Accuracy vs gamma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cv_rbf_gamma.png")
    plt.close()

    print("Saved cv_rbf_C.png, cv_rbf_gamma.png and cv_results_rbf.csv")

    # Refit PCA on FULL training and train RBF on full PCA features 
    print("\n4) Refit PCA on FULL training data and train final RBF on full PCA features...")
    pca_full = PCA(n_components=PCA_COMPONENTS, svd_solver='randomized', random_state=RANDOM_STATE)
    X_train_p_full = pca_full.fit_transform(X_train_s)
    X_test_p_full  = pca_full.transform(X_test_s)
    joblib.dump(pca_full, "Results - SVM/pca_full.joblib")

    best_rbf = SVC(kernel='rbf', C=rbf_gs.best_params_['C'], gamma=rbf_gs.best_params_['gamma'])
    best_rbf.fit(X_train_p_full, y_train)
    joblib.dump(best_rbf, "Results - SVM/svm_rbf_final_full.joblib")
    print("Saved final RBF: svm_rbf_final_full.joblib")

    # Evaluate final models on test set
    print("\n5) Evaluating final models on the test set...")

    preds_lin = final_linear.predict(X_test_s)
    acc_lin = accuracy_score(y_test, preds_lin)
    print(f"Linear test acc: {acc_lin:.4f}")
    print(classification_report(y_test, preds_lin, digits=4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds_lin)
    plt.title("Linear SVM (final) Confusion")
    plt.tight_layout(); plt.savefig("confusion_linear_final.png"); plt.close()

    preds_rbf = best_rbf.predict(X_test_p_full)
    acc_rbf = accuracy_score(y_test, preds_rbf)
    print(f"RBF (PCA) test acc: {acc_rbf:.4f}")
    print(classification_report(y_test, preds_rbf, digits=4))
    ConfusionMatrixDisplay.from_predictions(y_test, preds_rbf)
    plt.title("RBF (PCA) SVM (final) Confusion")
    plt.tight_layout(); plt.savefig("confusion_rbf_final.png"); plt.close()

    # Accuracy summary 
    print("\n=== Final test accuracies summary ===")
    print(f"Linear SVM (LinearSVC refit on full): {acc_lin:.4f}")
    print(f"RBF SVM (PCA, refit on full PCA features): {acc_rbf:.4f}")
    print("\nArtifacts saved (in current folder):")
    print(" - scaler.joblib")
    print(" - svm_linear_final_full.joblib")
    print(" - pca_tuning.joblib, pca_full.joblib")
    print(" - svm_rbf_final_full.joblib")
    print(" - confusion_linear_final.png, confusion_rbf_final.png")
    print(" - cv_linear_alpha.png, cv_rbf_C.png, cv_rbf_gamma.png")
    print(" - cv_results_sgd.csv, cv_results_rbf.csv")
    print("\nDone.")

if __name__ == "__main__":
    main()

