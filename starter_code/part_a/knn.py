import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *
import numpy as np
import os, sys


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
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
    """ Fill in the missing values using k-Nearest Neighbors based on
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
    mat = nbrs.fit_transform(matrix.T)

    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

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
    impute = [knn_impute_by_user, knn_impute_by_item]
    approach = ['user', 'item']
    acc_by_approach = dict()
    max_test_acc = []
    ibests = []

    for i in range(len(impute)):
        fn = impute[i]

        print('----------------------------------------------------')
        print(f'KNN with {approach[i]}')
        print('----------------------------------------------------')
        k_arr = np.linspace(1, 26, 6).astype(int)
        acc_by_k = []
        for k in k_arr:
            acc = fn(matrix=sparse_matrix, valid_data=val_data, k=k)
            acc_by_k.append(acc)
        acc_by_k = np.array(acc_by_k)
        ibest = np.argmax(acc_by_k)

        final_test_acc = fn(matrix=sparse_matrix, valid_data=test_data, k=k_arr[ibest])

        print(f'Test acc with maximum val acc with k* = {k_arr[ibest]} is {final_test_acc}')

        plt.figure(figsize=(8, 5))
        plt.plot(k_arr, acc_by_k)
        plt.title(f'KNN accuracy on validation set with similarity by {approach[i]}')
        plt.savefig(f'../out/KNN_{approach[i]}.jpg', dpi=450)

        acc_by_approach[approach[i]] = acc_by_k

        plt.show()

        max_test_acc.append(final_test_acc)
        ibests.append(ibest)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i in range(len(approach)):
        plt.plot(k_arr, acc_by_approach[approach[i]], label=approach[i])
        plt.title(f'Comparision of accuracy on validation set')

    plt.subplot(1, 2, 2)
    plt.bar(approach, max_test_acc)
    plt.title(f'Comparision of best accuracy on test set with k*={k_arr[ibests]}')
    plt.ylim((0.65, 0.7))

    plt.savefig(f'../out/KNN_compare.jpg', dpi=450)
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
