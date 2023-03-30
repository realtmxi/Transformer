from starter_code.utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    # TODO: Implement the function as described in the docstring.
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
    train_neg_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        train_neg_lld_lst.append(neg_lld)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_neg_lld_lst


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


def visualize(lst1, lst2):

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title(f'Training Loss vs. Epoch with k={10}, epochs={len(lst1)}, lr=0.01, lamb=0')
    plt.plot(lst1)

    plt.subplot(1, 2, 2)
    plt.title(f'Validation Accuracy vs. Epoch with k={10}, epochs={len(lst1)}, lr=0.01, lamb=0')
    plt.plot(lst2)
    plt.savefig('../out/irt.jpg', dpi=400)
    plt.show()


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
    theta, beta, val_acc_lst, train_neg_lld_lst = irt(data=train_data_matrix,
                                                      val_data=val_data,
                                                      lr=0.01,
                                                      iterations=20)
    # visualize(train_neg_lld_lst, val_acc_lst)

    print(f'final value accuracy = {val_acc_lst[-1]}')
    test_accuracy = evaluate(data=test_data, theta=theta, beta=beta)
    print(f"final test accuracy: {test_accuracy}")
    #####################################################################
    # TODO:
    # Implement part (d)                                                #
    #####################################################################
    # Step 1: Select three questions
    j1, j2, j3 = 1, 100, 1000  # You can choose any other indices

    # Step 2: Generate a range of theta values
    theta_range = np.linspace(-4, 4, 100)

    # Step 3: Compute the probabilities for each question and theta value
    p_j1 = sigmoid(theta_range - beta[j1])
    p_j2 = sigmoid(theta_range - beta[j2])
    p_j3 = sigmoid(theta_range - beta[j3])

    # Step 4: Plot the three curves on the same plot
    plt.plot(theta_range, p_j1, label=f"Question {j1}")
    plt.plot(theta_range, p_j2, label=f"Question {j2}")
    plt.plot(theta_range, p_j3, label=f"Question {j3}")

    plt.xlabel("Theta (Ability)")
    plt.ylabel("P(c_ij = 1)")
    plt.legend()
    plt.title("Probability of Correct Response as a Function of Theta")
    plt.savefig('../out/irt_d.jpg', dpi=400)
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
