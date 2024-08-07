import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
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
    print("Validation Accuracy (Hybrid): {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    k_values = [1, 6, 11, 16, 21, 26]
    hybrid_accuracies = []

    for k in k_values:
        print(f"Evaluating k={k} for hybrid KNN...")
        hybrid_acc = knn_impute_by_user_item(sparse_matrix, val_data, k, k, 0.5, 0.5)
        hybrid_accuracies.append(hybrid_acc)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, hybrid_accuracies, label='Hybrid KNN', marker='o')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. k for Hybrid KNN Imputation')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Find the best k for hybrid KNN
    best_k_hybrid = k_values[np.argmax(hybrid_accuracies)]
    print(f"Best k for hybrid KNN: {best_k_hybrid}")
    test_acc_hybrid = knn_impute_by_user_item(sparse_matrix, test_data, best_k_hybrid, best_k_hybrid, 0.5, 0.5)
    print(f"Test Accuracy for hybrid KNN with k={best_k_hybrid}: {test_acc_hybrid}")


if __name__ == "__main__":
    main()
