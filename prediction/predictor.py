from tqdm import tqdm

import numpy as np


def predict_vector(rating_vector, train_vector, remove_train=True):
    dim = len(rating_vector)
    candidate_index = np.argpartition(-rating_vector, dim-1)[:dim]
    prediction_items = candidate_index[rating_vector[candidate_index].argsort()[::-1]]

    if remove_train:
        return np.delete(prediction_items, np.isin(prediction_items, train_vector.nonzero()[1]).nonzero()[0])
    else:
        return prediction_items


def predict_scores(matrix_U, matrix_V, bias=None, gpu=False):
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    if bias is None:
        prediction = matrix_U.dot(matrix_V.T)
        return prediction

    if gpu:
        import cupy as cp
        return prediction + cp.array(bias)
    else:
        return prediction + bias


def predict_items(prediction_scores, topK, matrix_Train, gpu=False):
    prediction = []

    for user_index in tqdm(range(prediction_scores.shape[0])):
        vector_u = prediction_scores[user_index]
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            train_index = vector_train.nonzero()[1]

            if gpu:
                import cupy as cp
                candidate_index = cp.argpartition(-vector_u, topK+len(train_index))[:topK+len(train_index)]
                vector_u = candidate_index[vector_u[candidate_index].argsort()[::-1]]
                vector_u = cp.asnumpy(vector_u).astype(np.float32)
            else:
                candidate_index = np.argpartition(-vector_u, topK+len(train_index))[:topK+len(train_index)]
                vector_u = candidate_index[vector_u[candidate_index].argsort()[::-1]]
            vector_u = np.delete(vector_u, np.isin(vector_u, train_index).nonzero()[0])

            vector_predict = vector_u[:topK]
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def predict(matrix_U, matrix_V, topK, matrix_Train, bias=None, gpu=False):
    if gpu:
        import cupy as cp
        matrix_U = cp.array(matrix_U)
        matrix_V = cp.array(matrix_V)

    prediction = []

    for user_index in tqdm(range(matrix_U.shape[0])):
        vector_u = matrix_U[user_index]
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(vector_u, matrix_V, vector_train, bias, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def sub_routine(vector_u, matrix_V, vector_train, bias, topK=500, gpu=False):

    train_index = vector_train.nonzero()[1]

    vector_predict = matrix_V.dot(vector_u)

    if bias is not None:
        if gpu:
            import cupy as cp
            vector_predict = vector_predict + cp.array(bias)
        else:
            vector_predict = vector_predict + bias

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]

