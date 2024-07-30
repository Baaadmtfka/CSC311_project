from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    log_lklihood = 0.0

    for i in range(len(data["is_correct"])):
        # user
        u = data["user_id"][i]
        theta_u = theta[u]
        # question
        q = data["question_id"][i]
        beta_q = beta[q]
        # c_ij
        c = data["is_correct"][i]
        # llk
        log_lklihood += c * (theta_u - beta_q) - np.log(1 + np.exp(theta_u - beta_q))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # init derivatives
    theta_deriv = np.zeros_like(theta)
    beta_deriv = np.zeros_like(beta)

    for i in range(len(data["is_correct"])):
        # user
        u = data["user_id"][i]
        theta_u = theta[u]
        # question
        q = data["question_id"][i]
        beta_q = beta[q]
        # c_ij
        c = data["is_correct"][i]
        # update derivatives
        theta_deriv[u] += c - sigmoid(theta_u - beta_q)
        beta_deriv[q] += -c + sigmoid(theta_u - beta_q)

    theta += lr * theta_deriv
    beta += lr * beta_deriv

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.

    # init with constants assuming operating on current fixed dataset
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    train_llk_lst = []
    val_llk_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk_lst.append(-neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        val_llk_lst.append(-val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_llk_lst, val_llk_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # hyper-parameters
    lr = 0.008
    iterations = 100

    # irt
    theta, beta, val_acc_lst, train_llk_lst, val_llk_lst = irt(train_data, val_data, lr, iterations)

    # plot training curve
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(list(range(iterations)), train_llk_lst, label="Training Log-Likelihood")
    ax1.plot(list(range(iterations)), val_llk_lst, label="Validation Log-Likelihood")
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("Log-Likelihood")
    ax1.legend()

    # final validation and test acc
    val_score = evaluate(val_data, theta, beta)
    test_score = evaluate(test_data, theta, beta)
    print("Validation Accuracy: {}".format(val_score))
    print("Test Accuracy: {}".format(test_score))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    # sample quesitons
    j1 = 1
    j2 = 1452
    j3 = 666
    #print(beta[j1], beta[j2], beta[j3])

    # prob
    p_j1 = sigmoid(theta - beta[j1])
    p_j2 = sigmoid(theta - beta[j2])
    p_j3 = sigmoid(theta - beta[j3])

    ax2.plot(theta, p_j1, "o", label=f'Question {j1}')
    ax2.plot(theta, p_j2, "o", label=f'Question {j2}')
    ax2.plot(theta, p_j3, "o", label=f'Question {j3}')
    ax2.set_xlabel("Theta (Student Ability)")
    ax2.set_xlabel("Probability of Correct Response")
    ax2.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
