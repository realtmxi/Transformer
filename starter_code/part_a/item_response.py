from starter_code.utils import *

import numpy as np
import pandas as pd


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

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


    # Masking
    positive_mask = data == 1
    negative_mask = data == 0

    m = theta.shape[0]
    n = beta.shape[0]

    theta = np.expand_dims(theta, axis=1)
    beta = np.expand_dims(beta, axis=1)

    theta_mat = np.repeat(theta, n, axis=1)
    beta_mat = np.repeat(beta, m, axis=1).T

    x = theta_mat - beta_mat
    p = sigmoid(x)

    # Project the mask and compute the sum
    log_lklihood = np.sum(np.log(p[positive_mask])) + np.sum(np.log((1 - p))[negative_mask])

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

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

    m = theta.shape[0]
    n = beta.shape[0]

    theta = np.expand_dims(theta, axis=1)
    beta = np.expand_dims(beta, axis=1)

    theta_mat = np.repeat(theta, n, axis=1)
    beta_mat = np.repeat(beta, m, axis=1).T

    p = sigmoid(theta_mat - beta_mat)

    mask = ~np.isnan(data)

    c_masked = np.where(mask, data, 0)
    p_masked = np.where(mask, p, 0)

    dtheta = np.sum(c_masked - p_masked, axis=1)
    dbeta = np.sum(p_masked - c_masked, axis=0)

    theta = theta.flatten() + dtheta * lr
    beta = beta.flatten() + dbeta * lr

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

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

    # df = pd.DataFrame(data)
    # df = df.pivot(index='user_id', columns='question_id')
    # data = df.to_numpy()

    # print(data.shape)

    m = data.shape[0]
    n = data.shape[1]

    theta = np.random.random(size=(m, ))
    beta = np.random.random(size=(n, ))

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
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
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    df = pd.DataFrame(train_data)
    df = df.pivot(index='user_id', columns='question_id')
    train_data_matrix = df.to_numpy()

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    # print(train_data)
    irt(data=train_data_matrix, val_data=val_data, lr=0.01, iterations=22)
    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
