import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def svd_reconstruct(matrix, k):
    """Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i] - np.sum(u[data["user_id"][i]] * z[q])) ** 2.0
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # gradients
    error = c - np.dot(u[n].T, z[q])
    u_grad = error * -z[q]
    z_grad = error * -u[n]

    # update user and item latent factors
    u[n] -= lr * u_grad
    z[q] -= lr * z_grad

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """Performs ALS algorithm, here we use the iterative solution - SGD
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["user_id"])), k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(len(set(train_data["question_id"])), k)
    )

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    train_losses = []
    val_losses = []

    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        if i % 1000 == 0:
            #print(i)
            train_loss = squared_error_loss(train_data, u, z)
            val_loss = squared_error_loss(val_data, u, z)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    mat = np.dot(u, z.T)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return train_losses, val_losses, mat


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # SVD
    # init parameters
    ks = [1, 3, 5, 7, 8, 9, 10, 11, 13, 15, 50, 100, 200, 500]
    mats = []
    accs = []

    for k in ks:
        mat = svd_reconstruct(train_matrix, k)
        mats.append(mat)
        acc = sparse_matrix_evaluate(val_data, mat)
        accs.append(acc)

    # results with argmax k*
    max_acc_idx = np.argmax(accs)
    test_acc = sparse_matrix_evaluate(test_data, mats[max_acc_idx])
    print("SVD:"
          f"\nChosen argmax k*: {ks[max_acc_idx]}, "
          f"\nValidation accuracy: {accs[max_acc_idx]}, "
          f"\nTest accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # ALS
    # init hyperparameters
    lr = 0.12
    num_iteration = 50000
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #ks = [6]
    mats = []
    accs = []
    losses = []

    for k in ks:
        train_losses, val_losses, mat = als(train_data, val_data, k, lr, num_iteration)
        mats.append(mat)
        losses.append((train_losses, val_losses))
        acc = sparse_matrix_evaluate(val_data, mat)
        accs.append(acc)
        #train_acc = sparse_matrix_evaluate(train_data, mat)
        #print(f"k={k}, Training Accuracy={train_acc}")
        #print(f"k={k}, Validation Accuracy={acc}")

    max_acc_idx = np.argmax(accs)
    test_acc = sparse_matrix_evaluate(test_data, mats[max_acc_idx])
    print("ALS:"
          f"\nChosen argmax k*: {ks[max_acc_idx]},"
          f"\nValidation Accuracy: {accs[max_acc_idx]}"
          f"\nTest Accuracy: {test_acc}")

    # plot losses
    train_loss, val_loss = losses[max_acc_idx]
    # Computing loss for each iteration is expensive, so once every 1000 iterations.
    # still takes about 2 mins
    plt.plot(list(range(num_iteration // 1000)), train_loss, label='Training Loss')
    plt.plot(list(range(num_iteration // 1000)), val_loss, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Squared-Error Loss')
    plt.title(f'ALS Squared-Error Loss vs Iterations (k={ks[max_acc_idx]})')
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
