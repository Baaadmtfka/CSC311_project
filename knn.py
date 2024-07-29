import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = (nbrs.fit_transform(matrix.T)).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # init parameters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ks = [1, 6, 11, 16, 21, 26]

    # knn_impute_by_user
    print("\nknn_impute_by_user:")
    accs = []
    # KNN
    for k in ks:
        accs.append(knn_impute_by_user(sparse_matrix, val_data, k))
    # plot
    ax1.plot(ks, accs, 'bo-')
    ax1.set_xlabel("k")
    ax1.set_ylabel("Accuracy")
    # report argmax k* and test result
    max_acc_idx = np.argmax(accs)
    max_k = ks[max_acc_idx]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, max_k)
    print("Chosen argmax k*:", max_k, ", Test accuracy:", test_acc)

    # knn_impute_by_item
    print("\nknn_impute_by_item:")
    accs = []
    # KNN
    for k in ks:
        accs.append(knn_impute_by_item(sparse_matrix, val_data, k))
    # plot
    ax2.plot(ks, accs, 'bo-')
    ax2.set_xlabel("k")
    ax2.set_ylabel("Accuracy")
    # report argmax k* and test result
    max_acc_idx = np.argmax(accs)
    max_k = ks[max_acc_idx]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, max_k)
    print("Chosen argmax k*:", max_k, ", Test accuracy:", test_acc)

    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
