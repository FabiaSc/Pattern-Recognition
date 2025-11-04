""" Simple SVM for Exercise 2 (MNIST). """

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# load dataset from tsv files
def load_indexed_images(tsv_path, base_dir=None, image_size=(28,28), as_gray=True):
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['relpath','label'])
    base = Path(tsv_path).parent if base_dir is None else Path(base_dir)
    images = []
    labels = []
    for rel, label in zip(df['relpath'], df['label']):
        p = base / rel
        # Open image, convert to grayscale, resize if necessary, convert to numpy
        with Image.open(p) as im:
            if as_gray:
                im = im.convert('L')
            if image_size is not None:
                im = im.resize(image_size)  # MNIST is 28x28 but this is safe
            arr = np.asarray(im, dtype=np.float32)
            images.append(arr.ravel())  # flatten to 1D
            labels.append(int(label))
    X = np.stack(images, axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y

# Main pipeline
def main():
    # Paths
    train_tsv = 'gt-train.tsv'
    test_tsv  = 'gt-test.tsv'

    print("Loading training data...")
    X_train, y_train = load_indexed_images(train_tsv)
    print(f"Train: {X_train.shape[0]} samples, feature dim {X_train.shape[1]}")

    print("Loading test data...")
    X_test, y_test = load_indexed_images(test_tsv)
    print(f"Test: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Save scaler for later use
    joblib.dump(scaler, "svm_scaler.joblib")

    # Grid search for Linear SVM
    print("\nGrid search: LinearSVC")
    linear_params = {'C': [0.1, 1, 10]}
    linear_svc = LinearSVC(max_iter=5000, dual=False)
    linear_gs = GridSearchCV(linear_svc, linear_params, cv=5, n_jobs=1, verbose=2, scoring='accuracy')
    linear_gs.fit(X_train_s, y_train)
    print("Best linear params:", linear_gs.best_params_, "CV:", linear_gs.best_score_)
    # save model
    joblib.dump(linear_gs.best_estimator_, "svm_linear_best.joblib")

    # Grid search for RBF SVM
    print("\nGrid search: RBF SVM")
    rbf_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1]
    }
    rbf_svc = SVC(kernel='rbf', decision_function_shape='ovr')
    rbf_gs = GridSearchCV(rbf_svc, rbf_params, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    rbf_gs.fit(X_train_s, y_train)
    print("Best RBF params:", rbf_gs.best_params_)
    print("Best RBF CV score:", rbf_gs.best_score_)

    joblib.dump(rbf_gs.best_estimator_, "svm_rbf_best.joblib")

    # Evaluate best models on test set
    print("\nEvaluating on test set...")

    def eval_model(model, Xs, ys, name):
        preds = model.predict(Xs)
        acc = accuracy_score(ys, preds)
        print(f"\n{name} Test accuracy: {acc:.4f}")
        print(classification_report(ys, preds, digits=4))
        # Confusion matrix plot
        disp = ConfusionMatrixDisplay.from_predictions(ys, preds, cmap='viridis', xticks_rotation='vertical')
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{name.replace(' ', '_')}_confusion.png")
        plt.close()
        return acc

    best_linear = linear_gs.best_estimator_
    best_rbf = rbf_gs.best_estimator_

    acc_lin = eval_model(best_linear, X_test_s, y_test, "SVM Linear")
    acc_rbf = eval_model(best_rbf, X_test_s, y_test, "SVM RBF")

    # Plot CV results
    def plot_grid_results(gs, param_name, title):
        res = gs.cv_results_
        params = res['params']
        mean_scores = res['mean_test_score']
        # Build dataframe
        df = pd.DataFrame(params)
        df['score'] = mean_scores
        # Aggregate by param_name taking mean over other params
        agg = df.groupby(param_name)['score'].mean().reset_index()
        plt.figure()
        plt.plot(agg[param_name].astype(str), agg['score'], marker='o')
        plt.xlabel(param_name)
        plt.ylabel('Mean CV accuracy')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        fname = title.replace(' ', '_') + '.png'
        plt.savefig(fname)
        plt.close()

    plot_grid_results(linear_gs, 'C', 'Linear SVM: CV accuracy vs C')

    # Plot vs C (averaging over gamma) and vs gamma (averaging over C)
    res_r = pd.DataFrame(rbf_gs.cv_results_['params'])
    res_r['score'] = rbf_gs.cv_results_['mean_test_score']

    # Plot aggregated by C
    aggC = res_r.groupby('C')['score'].mean().reset_index()
    plt.figure(); plt.plot(aggC['C'].astype(str), aggC['score'], marker='o')
    plt.xlabel('C'); plt.ylabel('Mean CV accuracy'); plt.title('RBF SVM: CV accuracy vs C'); plt.grid(True)
    plt.tight_layout(); plt.savefig('RBF_C_vs_CVaccuracy.png'); plt.close()

    # Plot aggregated by gamma
    aggg = res_r.groupby('gamma')['score'].mean().reset_index()
    plt.figure(); plt.plot(aggg['gamma'].astype(str), aggg['score'], marker='o')
    plt.xlabel('gamma'); plt.ylabel('Mean CV accuracy'); plt.title('RBF SVM: CV accuracy vs gamma'); plt.grid(True)
    plt.tight_layout(); plt.savefig('RBF_gamma_vs_CVaccuracy.png'); plt.close()

    print("\nDone. Models and plots saved to current folder.")
    print(f"Linear test acc: {acc_lin:.4f}, RBF test acc: {acc_rbf:.4f}")

if __name__ == "__main__":
    main()
