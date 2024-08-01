import numpy as np
from sklearn.impute import KNNImputer
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
    sparse_matrix_predictions,
    evaluate
)
from knn import knn_impute_by_user
from item_response import irt, sigmoid
from matrix_factorization import als


def resample(data):
    # samples
    n_samples = len(data["user_id"])
    # resample indices
    indices = np.random.choice(range(n_samples), size=n_samples, replace=True)

    # Create resampled data using the indices
    resampled_data = {
        "user_id": np.array(data["user_id"])[indices],
        "question_id": np.array(data["question_id"])[indices],
        "is_correct": np.array(data["is_correct"])[indices]
    }
    return resampled_data

def irt_pred(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return pred

def ensemble(train_matrix, train_data, val_data, test_data, hp):
    val_pred = []
    test_pred = []

    # KNN (user-based)
    # resampling with replacement
    knn_train_data = resample(train_data)
    # convert to sparse matrix
    knn_sparse_matrix = np.empty(train_matrix.shape)
    knn_sparse_matrix[:] = np.nan
    for i in range(len(knn_train_data["is_correct"])):
        knn_sparse_matrix[knn_train_data["user_id"][i], knn_train_data["question_id"][i]] \
            = knn_train_data["is_correct"][i]
    nbrs = KNNImputer(n_neighbors=hp["knn_user_k"])
    knn_mat = nbrs.fit_transform(knn_sparse_matrix)
    knn_acc = sparse_matrix_evaluate(val_data, knn_mat)
    print(f"KNN: Validation Accuracy: {knn_acc}")
    val_pred.append(sparse_matrix_predictions(val_data, knn_mat))
    test_pred.append(sparse_matrix_predictions(test_data, knn_mat))

    # IRT
    irt_train_data = resample(train_data)
    theta, beta, _, _, _ = irt(irt_train_data, val_data, hp["irt_lr"], hp["irt_iter"])
    irt_acc = evaluate(val_data, irt_pred(val_data, theta, beta))
    print(f"IRT: Validation Accuracy: {irt_acc}")
    val_pred.append(irt_pred(val_data, theta, beta))
    test_pred.append(irt_pred(test_data, theta, beta))

    # ALS
    als_train_data = resample(train_data)
    _, _, als_mat = als(als_train_data, val_data, hp["als_k"], hp["als_lr"], hp["als_iter"])
    als_acc = sparse_matrix_evaluate(val_data, als_mat)
    print(f"ALS: Validation Accuracy: {als_acc}")
    val_pred.append(sparse_matrix_predictions(val_data, als_mat))
    test_pred.append(sparse_matrix_predictions(test_data, als_mat))

    return val_pred, test_pred


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # hyper-parameters
    hp = {
        # KNN
        "knn_user_k": 11,
        # IRT
        "irt_lr": 0.006,
        "irt_iter": 100,
        # ALS
        "als_lr": 0.12,
        "als_iter": 50000,
        "als_k": 6
    }

    # ensemble
    val_pred, test_pred = ensemble(train_matrix, train_data, val_data, test_data, hp)
    # the mean of predictions from the 3 models
    mean_val_pred = np.mean(np.array(val_pred), axis=0)
    mean_test_pred = np.mean(np.array(test_pred), axis=0)
    # accuracy of combined prediction
    val_acc = evaluate(val_data, mean_val_pred)
    test_acc = evaluate(test_data, mean_test_pred)
    print("Final Ensembled Results:")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
