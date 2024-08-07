import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from part_b_zengzixuan import load_question_meta_csv, als
import part_b_liuhao
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
    sparse_matrix_predictions,
    evaluate
)


def knn_impute_by_user_item(matrix, valid_data, k_user, k_item, user_weight=0.5, item_weight=0.5):
    """Fill in the missing values using a hybrid k-Nearest Neighbors approach
    based on both student and question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param k_user: int, number of neighbors for user-based KNN
    :param k_item: int, number of neighbors for item-based KNN
    :param user_weight: float, weight for user-based KNN
    :param item_weight: float, weight for item-based KNN
    :return: float
    """
    # User-based KNN
    user_nbrs = KNNImputer(n_neighbors=k_user)
    user_mat = user_nbrs.fit_transform(matrix)

    # Item-based KNN
    item_nbrs = KNNImputer(n_neighbors=k_item)
    item_mat = item_nbrs.fit_transform(matrix.T).T

    # Combine user and item matrices
    combined_mat = user_weight * user_mat + item_weight * item_mat

    acc = sparse_matrix_evaluate(valid_data, combined_mat)
    print("Validation Accuracy (Hybrid KNN): {}".format(acc))
    return acc, combined_mat


def main():
    # data import
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_meta = load_question_meta_csv("./data")
    train_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in train_data["question_id"]]
    val_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in val_data["question_id"]]
    test_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in test_data["question_id"]]

    # Hybrid KNN where k*=11
    print("Training Hybrid KNN with k*=11...")
    hybrid_acc, knn_mat = knn_impute_by_user_item(train_matrix, val_data, 11, 11, 0.5, 0.5)
    knn_val_pred = sparse_matrix_predictions(val_data, knn_mat)
    knn_test_pred = sparse_matrix_predictions(test_data, knn_mat)

    # ALS Matrix Factorization
    print("Training Modified ALS Matrix Factorization...")
    lr = 0.022
    num_iteration = 300000
    k = 11
    train_losses, val_losses, als_mat = als(train_data, val_data, k, lr, num_iteration)
    als_val_acc = sparse_matrix_evaluate(val_data, als_mat)
    print("Validation Accuracy (Modified ALS): {}".format(als_val_acc))
    als_val_pred = sparse_matrix_predictions(val_data, als_mat)
    als_test_pred = sparse_matrix_predictions(test_data, als_mat)

    # the mean of predictions from the 3 models
    mean_val_pred = np.mean([knn_val_pred, als_val_pred], axis=0)
    mean_test_pred = np.mean([knn_test_pred, als_test_pred], axis=0)
    # accuracy of combined prediction
    val_acc = evaluate(val_data, mean_val_pred)
    test_acc = evaluate(test_data, mean_test_pred)
    print("Final Ensembled Results:")
    print(f"Validation Accuracy: {val_acc}")
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()
