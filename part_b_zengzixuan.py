import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from matrix_factorization import als as base_als
import csv
import os

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def load_question_meta_csv(root_dir="./data"):
    """Load question metadata.
    :param root_dir: str, path to the question metadata file
    :return: dict, mapping of question_id to a dictionary of question metadata
    """
    path = os.path.join(root_dir, "question_meta.csv")
    question_meta = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            question_id = int(row[0])
            subject_ids = eval(row[1])  # Convert string representation of list to list
            question_meta[question_id] = {"subject_ids": subject_ids}
    return question_meta

def create_subject_matrix(num_questions, num_subjects, data):
    """
    Create a sparse matrix representation of subjects for each question.
    :param data: DataFrame with question data including 'question_id'
    :return: Sparse matrix with questions as rows and subjects as columns
    """

    # create matrix
    subject_matrix = np.zeros((num_questions, num_subjects))

    # populate the sparse matrix
    for i, q in enumerate(data["question_id"]):
        subject_ids = data["subjects"][q]
        for subj in subject_ids:
            subject_matrix[q, subj] = 1 / np.sqrt(num_subjects)
    return subject_matrix


def squared_error_loss(data, u, z):
    """Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :param g: Gender biases
    :param p: Premium pupil biases
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        user_id = data["user_id"][i]
        is_correct = data["is_correct"][i]
        pred = u[user_id].dot(z[q])
        loss += (is_correct - pred) ** 2
    return 0.5 * loss


def update_lf(train_data, lr, u, z, u_s, z_s):
    """Update latent factors using stochastic gradient descent.
    :param train_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :param u_s: 2D matrix
    :param z_s: 2D matrix
    :return: Updated u, z, u_s, z_s
    """
    # Randomly select a pair (user_id, question_id).
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # error
    error = c - (np.dot(u_s[n].T, z_s[q]) + np.dot(u[n].T, z[q]))

    # update subject factors
    subjects = train_data["subjects"][q]
    #error = c - np.dot(u_s[n].T, z_s[q])
    u_s_grad = error * -z_s[q]
    z_s_grad = error * -u_s[n]
    '''for s in subjects:
        u_s[n, s] -= lr * u_s_grad[s]
        z_s[q, s] -= lr * z_s_grad[s]
    '''
    u_s[n] -= lr * u_s_grad
    z_s[q] -= lr * z_s_grad

    # update latent factors
    #error = c - np.dot(u[n].T, z[q])
    u_grad = error * -z[q]
    z_grad = error * -u[n]
    u[n] -= lr * u_grad
    z[q] -= lr * z_grad

    return u, z, u_s, z_s


def als(train_data, val_data, k, lr, num_iteration):
    """Performs ALS algorithm with additional factors.
    :param train_data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param val_data: Validation data
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    num_users = len(set(train_data["user_id"]))
    num_questions = len(set(train_data["question_id"]))
    num_subjects = 388

    # init latent factors ###and biases
    u = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(num_users, k)
    )
    z = np.random.uniform(
        low=0, high=1 / np.sqrt(k), size=(num_questions, k)
    )
    # subject factors
    u_s = np.random.uniform(
        low=0, high=1 / np.sqrt(num_subjects), size=(num_users, num_subjects)
    )
    # subject sparse matrix for Z, with entries set at 1 / np.sqrt(num_subjects), otw 0.
    z_s = create_subject_matrix(num_questions, num_subjects, train_data)
    u_con = np.concatenate((u, u_s), axis=1)
    z_con = np.concatenate((z, z_s), axis=1)

    train_losses = []
    val_losses = []

    for i in range(num_iteration):
        #print("iteration: ", i)
        u, z, u_s, z_s = update_lf(
            train_data, lr, u, z, u_s, z_s)
        if i % 1000 == 0:
            u_con = np.concatenate((u, u_s), axis=1)
            z_con = np.concatenate((z, z_s), axis=1)
            train_loss = squared_error_loss(train_data, u_con, z_con) / len(train_data["question_id"])
            val_loss = squared_error_loss(val_data, u_con, z_con) / len(val_data["question_id"])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    #print(u, z, u_s, z_s)

    mat = np.dot(u_con, z_con.T)
    return train_losses, val_losses, mat


def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")
    question_meta = load_question_meta_csv("./data")

    # map questions to their subjects
    train_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in train_data["question_id"]]
    val_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in val_data["question_id"]]
    test_data["subjects"] = [list(map(int, question_meta[q]["subject_ids"])) for q in test_data["question_id"]]

    # Modified ALS
    # Hyperparameters
    lr = 0.022
    num_iteration = 300000
    k = 11
    # Train
    train_losses, val_losses, mat = als(train_data, val_data, k, lr, num_iteration)
    # Evaluate
    val_acc = sparse_matrix_evaluate(val_data, mat)
    test_acc = sparse_matrix_evaluate(test_data, mat)
    print(f"Modified ALS Validation Accuracy: {val_acc}")
    print(f"Modified ALS Test Accuracy: {test_acc}")
    plt.plot(train_losses, label='Modified ALS Training Loss')
    plt.plot(val_losses, label='Modified ALS Validation Loss')
    # Plot training and validation loss

    # Base Model ALS
    '''lr = 0.012
    num_iteration = 300000
    k = 2
    # Train
    base_train_losses, base_val_losses, base_mat = base_als(train_data, val_data, k, lr, num_iteration)
    # Evaluate
    base_train_losses = [loss / len(train_data["question_id"]) for loss in base_train_losses]
    base_val_losses = [loss / len(val_data["question_id"]) for loss in base_val_losses]
    base_val_acc = sparse_matrix_evaluate(val_data, base_mat)
    base_test_acc = sparse_matrix_evaluate(test_data, base_mat)
    print(f"Base ALS Validation Accuracy: {base_val_acc}")
    print(f"Base ALS Test Accuracy: {base_test_acc}")
    plt.plot(base_train_losses, label='Base ALS Training Loss')
    plt.plot(base_val_losses, label='Base ALS Validation Loss')'''
    
    plt.xlabel('Iterations (in thousands)')
    plt.ylabel('avg Loss')
    plt.title('ALS avg Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
