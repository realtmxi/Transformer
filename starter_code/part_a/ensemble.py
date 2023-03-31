from matplotlib import pyplot as plt

from neural_network import AutoEncoder, train, load_data
from starter_code.utils import *
from sklearn.impute import KNNImputer
import torch

from knn import knn_impute_by_user
from item_response import irt
from torch.autograd import Variable

EPOCH = 10
LEARNING_RATE_IRT = 0.01
LEARNING_RATE_NN = 0.1
LAMBDA = 0
NUM_MODELS = 3
EMBED_DIMENSION = 256

model_type = ['knn', 'irt', 'neurl']


def get_train_data():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    train_data = load_train_csv('../data/')

    return zero_train_matrix, train_data, valid_data, test_data


def bootstrap(dataset):
    bootstrapped = dict()

    users = np.array(dataset['user_id'])
    questions = np.array(dataset['question_id'])
    is_correct = np.array(dataset['is_correct'])

    n_sample = len(users)
    indices = np.random.choice(range(n_sample), n_sample, True)

    bootstrapped['user_id'] = users[indices]
    bootstrapped['question_id'] = questions[indices]
    bootstrapped['is_correct'] = is_correct[indices]

    return bootstrapped


def dict2matrix(data):
    user_ids = data['user_id']
    question_ids = data['question_id']
    is_correct = data['is_correct']

    matrix = np.zeros((np.max(user_ids) + 1, np.max(question_ids) + 1))
    matrix[:] = np.nan
    matrix[user_ids, question_ids] = is_correct
    return matrix


def generate_bootstrap(train_data):
    train_data = train_data.numpy()
    n_rows, n_cols = train_data.shape

    n_samples = 542

    # Bootstrap the sparse matrix
    num_sets = 3

    training_data = []

    for j in range(num_sets):
        bootstrapped_mats = []
        for i in range(n_samples):
            # Generate a bootstrap sample by randomly selecting rows with replacement
            row_indices = np.random.choice(n_rows, size=n_rows, replace=True)
            bootstrapped_mat = train_data[row_indices, :]
            bootstrapped_mats.append(bootstrapped_mat)

        training_data.append(torch.tensor(np.array(bootstrapped_mat)))

    return training_data[0], training_data[1], training_data[2]


def generate_masked(train_data):
    m, n = train_data.size()[0], train_data.size()[1]
    part_size = m * n // 3

    indices = torch.randperm(m * n)

    # Generate three masks that form a partition of the tensor
    mask1 = torch.ones((m, n))
    mask1.view(-1)[indices[:part_size]] = 0
    mask2 = torch.ones((m, n))
    mask2.view(-1)[indices[part_size:2 * part_size]] = 0
    mask3 = torch.ones((m, n))
    mask3.view(-1)[indices[2 * part_size:]] = 0

    partition_1 = torch.masked_fill(train_data, mask=mask1.bool(), value=torch.tensor(float('nan')))
    partition_2 = torch.masked_fill(train_data, mask=mask2.bool(), value=torch.tensor(float('nan')))
    partition_3 = torch.masked_fill(train_data, mask=mask3.bool(), value=torch.tensor(float('nan')))

    return partition_1, partition_2, partition_3


def get_models(train_data, zero_mat, val_data):
    num_question = train_data['neurl'].size()[1]

    models = dict()
    models['knn'] = knn_impute_by_user
    models['irt'] = irt
    models['neurl'] = AutoEncoder(num_question, k=EMBED_DIMENSION)

    theta, beta, _ = models['irt'](data=train_data['irt'],
                                   val_data=val_data,
                                   lr=LEARNING_RATE_IRT,
                                   iterations=EPOCH)

    train(models['neurl'], lr=LEARNING_RATE_NN, lamb=LAMBDA,
          train_data=train_data['neurl'],
          zero_train_data=zero_mat,
          valid_data=val_data,
          num_epoch=EPOCH)

    return models, (theta, beta)


def visualize(mask1, mask2):
    masks = [mask1, mask2]

    rslt = mask1 + mask2
    rslt /= 2

    plt.figure(figsize=(15, 4))
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(masks[i])
    plt.show()

    plt.imshow(rslt)
    plt.show()
    return


def pred(models, irt_param, train_data, valid_data):

    # Get KNN pred_mat
    theta, beta = irt_param[0], irt_param[1]

    nbrs = KNNImputer(n_neighbors=11)
    knn_rslt = nbrs.fit_transform(train_data)

    # Get irt pred_mat
    m = theta.shape[0]
    n = beta.shape[0]

    theta = np.expand_dims(theta, axis=1)
    beta = np.expand_dims(beta, axis=1)

    theta_mat = np.repeat(theta, n, axis=1)
    beta_mat = np.repeat(beta, m, axis=1).T

    x = theta_mat - beta_mat
    irt_rslt = np.exp(x) / (1 + np.exp(x))

    nn_rslt = np.zeros_like(knn_rslt)
    # Get nn pred
    with torch.no_grad():
        nn = models['neurl']
        nn.eval()
        counter = 0
        correct = 0
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(train_data[u]).unsqueeze(0)
            output = nn(inputs)
            nn_guess = output[0][valid_data["question_id"][i]].item() >= 0.5

            knn_guess = knn_rslt[u][valid_data["question_id"][i]] >= 0.5
            irt_guess = irt_rslt[u][valid_data["question_id"][i]] >= 0.5

            guesses = np.array( [nn_guess, knn_guess, irt_guess])

            num_true = np.count_nonzero(guesses)
            num_false = guesses.size - num_true

            guess = True if num_true > num_false else False

            if guess == valid_data["is_correct"][i]:
                correct += 1

            counter += 1

        acc = correct / counter

        return knn_rslt, irt_rslt, acc

def main():
    zero_mat, train_data, valid_data, test_data = get_train_data()
    # train_data = torch.ones(size=(542, 1774))
    # fold1, fold2, fold3= generate_masked(train_data)
    fold1 = dict2matrix(bootstrap(train_data)).astype(np.float32)
    fold2 = dict2matrix(bootstrap(train_data)).astype(np.float32)
    fold3 = dict2matrix(bootstrap(train_data)).astype(np.float32)

    all_train_data = {'knn': fold1, 'irt': fold2, 'neurl': torch.tensor(fold3)}

    # visualize(train_data.numpy(), fold1.numpy(), fold2.numpy(), fold3.numpy())

    zero_train_matrix = torch.tensor(fold3.copy())

    zero_train_matrix[np.isnan(fold3)] = 0

    models, (theta, beta) = get_models(all_train_data, zero_train_matrix, valid_data)
    knn_rslt, irt_rslt, val_acc = pred(models, (theta, beta), torch.tensor(dict2matrix(train_data).astype(np.float32)), valid_data)
    knn_rslt, irt_rslt, test_acc = pred(models, (theta, beta), torch.tensor(dict2matrix(train_data).astype(np.float32)), test_data)

    print(f'Validation Accuracy:{val_acc}')
    print(f'Test Accuracy:{test_acc}')

    visualize(knn_rslt, irt_rslt)


if __name__ == '__main__':
    main()