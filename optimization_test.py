# Import the Required Libraries
<<<<<<< HEAD
import numpy as np
=======
import autograd.numpy as np
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
# from autograd import grad
# import random
import time
from numba import jit, cuda, prange, float32
<<<<<<< HEAD
from numpy import linalg as LA
=======
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6

lmd_BPR = 100
lmd_u = 1
lmd_v = 1
NUM_USER = 7920
NUM_ITEM = 13428
EXP = 2.718281828459045
blockdim = (32, 32)


def lossfunction_all(rating_matrix, movie_vectors, user_vectors, flag):
    if flag == 1:  # used in user tree construction, user_vectors is a 1*K vector
        user_vectors = np.array([user_vectors for i in range(len(rating_matrix))])
    if flag == 0:  # used in item tree construction, movie_vectors is a 1*K vector
        movie_vectors = np.array([movie_vectors for i in range(rating_matrix.shape[1])])
    value = 0
    # Add regularization term
    user_l = user_vectors[np.nonzero(user_vectors)]
    value = value + lmd_u * np.dot(user_l, user_l)
    mov_l = movie_vectors[np.nonzero(movie_vectors)]
    value = value + lmd_v * np.dot(mov_l, mov_l)

    if len(rating_matrix) == 0:
        return value

    predict = np.dot(user_vectors, movie_vectors.T)
    P = predict[np.nonzero(rating_matrix)]
    R = rating_matrix[np.nonzero(rating_matrix)]
    Err = P - R
    value = value + np.dot(Err, Err)

    np.random.seed(0)
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    for i in range(num_pair):
        c1, c2 = np.random.randint(0, num_item * num_user, 2)
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item
        if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
            diff = np.dot(user_vectors[u1, :].T, movie_vectors[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                movie_vectors[i2, :])
            diff = -diff
            value = value + lmd_BPR * np.log(1 + np.exp(diff))

    return value


@jit
def get_user_gradient(selected_points, selected_pairs, rating_matrix, user_vector, movie_vectors, lmd_u, lmd_BPR):
    num_user, num_item = rating_matrix.shape
    delta_u = 0
    for sp in selected_points:
        u1, i1 = sp // num_item, sp % num_item
        if rating_matrix[u1, i1] != 0:
            pred = 0
            for i in range(len(user_vector)):
                pred += user_vector[i] * movie_vectors[i1, i]
            delta_u += -2 * (rating_matrix[u1, i1] - pred) * movie_vectors[i1] + 2 * lmd_u * user_vector

    for j in range(int(len(selected_pairs) / 2)):
        c1 = selected_pairs[j * 2]
        c2 = selected_pairs[j * 2 + 1]
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item

        if rating_matrix[u1, i1] > rating_matrix[u2, i2]:
            diff = 0
            for i in range(len(user_vector)):
                diff += - user_vector[i] * (movie_vectors[i1, i] - movie_vectors[i2, i])
<<<<<<< HEAD
            vec_diff = 0
            vec_diff = movie_vectors[i2] - movie_vectors[i1]
            delta_u += lmd_BPR * vec_diff * pow(EXP, diff) / (1 + pow(EXP, diff))
=======
            delta_u += lmd_BPR * (movie_vectors[i2, :] - movie_vectors[i1, :]) * pow(EXP, diff) / (1 + pow(EXP, diff))
            # delta_u += lmd_BPR * (movie_vectors[i2, :] - movie_vectors[i1, :]) * np.exp(diff) / (1 + np.exp(diff))
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6

    return delta_u


def selfgradu(rating_matrix, movie_vectors, current_vector, user_vector):
    delta_u = np.zeros_like(user_vector)
    num_point = 100
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    np.random.seed(0)
    user_vector = user_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_u

    selected_points = np.random.randint(0, num_item * num_user, num_point)
    selected_pairs = np.random.randint(0, num_item * num_user, num_pair * 2)
    delta_u = get_user_gradient(selected_points, selected_pairs, rating_matrix, user_vector, movie_vectors, lmd_u, lmd_BPR)
<<<<<<< HEAD
=======
    # for i in range(num_point):
    #     c1 = np.random.randint(0, num_item * num_user)
    #     u1, i1 = c1 // num_item, c1 % num_item
    #     if rating_matrix[u1][i1] != 0:
    #         delta_u += -2 * (rating_matrix[u1][i1] - np.dot(user_vector, movie_vectors[i1])) * movie_vectors[i1] + 2 * lmd_u * user_vector

    # for i in range(num_pair):
    #     c1, c2 = np.random.randint(0, num_item * num_user, 2)
    #     u1, i1 = c1 // num_item, c1 % num_item
    #     u2, i2 = c2 // num_item, c2 % num_item
    #     if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
    #         diff = np.dot(user_vector.T, movie_vectors[i1, :]) - np.dot(user_vector.T, movie_vectors[i2, :])
    #         diff = -diff
    #         delta_u += lmd_BPR * (movie_vectors[i2, :] - movie_vectors[i1, :]) * np.exp(diff) / (1 + np.exp(diff))
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    return delta_u


@jit
def get_item_gradient(selected_points, selected_pairs, rating_matrix, user_vectors, movie_vector, lmd_v, lmd_BPR):
    num_user, num_item = rating_matrix.shape
    delta_v = 0
    for sp in selected_points:
        u1, i1 = sp // num_item, sp % num_item
        if rating_matrix[u1, i1] != 0:
            pred = 0
            for i in range(len(movie_vector)):
                pred += user_vectors[u1, i] * movie_vector[i]
            delta_v += -2 * (rating_matrix[u1, i1] - pred) * user_vectors[u1] + 2 * lmd_v * movie_vector

    for j in range(int(len(selected_pairs) / 2)):
        c1 = selected_pairs[j * 2]
        c2 = selected_pairs[j * 2 + 1]
        u1, i1 = c1 // num_item, c1 % num_item
        u2, i2 = c2 // num_item, c2 % num_item

        if rating_matrix[u1, i1] > rating_matrix[u2, i2]:
            diff = 0
            for i in range(len(movie_vector)):
<<<<<<< HEAD
                diff += -movie_vector[i] * (user_vectors[u1, i] - user_vectors[u2, i])
            vec_diff = 0
            vec_diff = user_vectors[u2] - user_vectors[u1]
            delta_v += lmd_BPR * vec_diff * pow(EXP, diff) / (1 + pow(EXP, diff))
=======
                diff += - movie_vector[i] * (user_vectors[u1, i] - user_vectors[u2, i])
            delta_v += lmd_BPR * (user_vectors[u2, :] - user_vectors[u1, :]) * pow(EXP, diff) / (1 + pow(EXP, diff))
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6

    return delta_v


def selfgradv(rating_matrix, movie_vector, current_vector, user_vectors):
    delta_v = np.zeros_like(movie_vector)
    num_point = 100
    num_pair = 20
    num_user, num_item = rating_matrix.shape
    np.random.seed(0)
    movie_vector = movie_vector + current_vector
    if len(rating_matrix) == 0:
        return delta_v

    selected_points = np.random.randint(0, num_item * num_user, num_point)
    selected_pairs = np.random.randint(0, num_item * num_user, num_pair * 2)
    delta_v = get_item_gradient(selected_points, selected_pairs, rating_matrix, user_vectors, movie_vector, lmd_v, lmd_BPR)

<<<<<<< HEAD
=======
    # for i in range(num_point):
    #     c1 = np.random.randint(0, num_item * num_user)
    #     u1, i1 = c1 // num_item, c1 % num_item
    #     if rating_matrix[u1][i1] != 0:
    #         delta_v += -2 * (rating_matrix[u1][i1] - np.dot(user_vectors[u1], movie_vector)) * user_vectors[u1] + 2 * lmd_v * movie_vector
    # for i in range(num_pair):
    #     c1, c2 = np.random.randint(0, num_item * num_user, 2)
    #     u1, i1 = c1 // num_item, c1 % num_item
    #     u2, i2 = c2 // num_item, c2 % num_item
    #     if rating_matrix[u1][i1] > rating_matrix[u2][i2]:
    #         diff = np.dot(user_vectors[u1].T, movie_vector) - np.dot(user_vectors[u2].T, movie_vector)
    #         diff = -diff
    #         delta_v += lmd_BPR * (user_vectors[u2, :] - user_vectors[u1, :]) * np.exp(diff) / (1 + np.exp(diff))
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    return delta_v


def cf_user(rating_matrix, item_vectors, current_vector, indices, K):
    # user_vector is len(indices)*K matrix
    # Stores the user profile vectors
    user_vector = np.random.rand(K)
    index_matrix = rating_matrix[indices]
    num_iter = 20
    eps = 1e-8
    lr = 0.1
    # set the variable user_vector to be gradient
    # mg = grad(lossfunction, argnum=2)
    sum_square_u = eps + np.zeros_like(user_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_u = selfgradu(index_matrix, item_vectors, current_vector, user_vector)
        sum_square_u += np.square(delta_u)
        lr_u = np.divide(lr, np.sqrt(sum_square_u))
        user_vector -= lr_u * delta_u
    user_vector = user_vector + current_vector

    return user_vector


def cf_item(rating_matrix, user_vectors, current_vector, indices, K):
    movie_vector = np.random.rand(K)
    rating_matrix = rating_matrix[:, indices]
<<<<<<< HEAD
    num_iter = 20
=======
    num_iter = 1000
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    eps = 1e-8
    lr = 0.1
    sum_square_v = eps + np.zeros_like(movie_vector)

    # SGD procedure:
    for i in range(num_iter):
        delta_v = selfgradv(rating_matrix, movie_vector, current_vector, user_vectors)
        sum_square_v += np.square(delta_v)
        lr_v = np.divide(lr, np.sqrt(sum_square_v))
        movie_vector -= lr_v * delta_v
    movie_vector = movie_vector + current_vector

    return movie_vector


# @jit('void(float64[:,:],float64[:,:],float64[:,:])')
@jit
def matmul(matrix1, matrix2, rmatrix):
    for i in range(len(matrix1)):
        for j in range(len(matrix2)):
            for k in range(len(matrix2[0])):
                rmatrix[i][j] += matrix1[i, k] * matrix2[j, k]


@cuda.jit
def matmul_cuda(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
    # cuda.syncthreads()


# @cuda.jit
# def vec_inner_cuda(vec_1, vec_2, vec_3):
#     i = cuda.grid(1)
#     if i < len(vec_1):
#         vec_3[i] = vec_1[i] * vec_2[i]
#     cuda.syncthreads()

<<<<<<< HEAD
# TPB = 16
=======
TPB = 16
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6


@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


@jit
def vec_inner(vec_1, vec_2):
    r = 0
    for i in range(len(vec_1)):
        r += vec_1[i] * vec_2[i]
    return r


@cuda.reduce
def sum_reduce(a, b):
    return a + b


@jit
def calculate_error(user_vector, movie_vectors, pred, rating):
    for i in range(len(user_vector)):
        for j in range(len(movie_vectors)):
            for k in range(len(movie_vectors[0])):
                pred[i, j] += user_vector[i, k] * movie_vectors[j, k]
    mask = rating != 0
    err = (pred - rating)[mask]
    r = 0
    for i in range(len(err)):
        r += pow(err[i], 2)

    return r


def cal_splitvalue(rating_matrix, movie_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    like = rating_matrix[indices_like]
    dislike = rating_matrix[indices_dislike]
    unknown = rating_matrix[indices_unknown]
    like_vector = np.zeros(K)
    dislike_vector = np.zeros(K)
    unknown_vector = np.zeros(K)
    value = 0.0

    if len(indices_like) > 0:
        like_vector = cf_user(rating_matrix, movie_vectors, current_vector, indices_like, K)
        like_vector = np.repeat(like_vector.reshape(1, -1), len(indices_like), axis=0)
        pre_like = np.dot(like_vector, movie_vectors.T)
        Err_like = (pre_like - like)[np.nonzero(like)]
        value += np.dot(Err_like, Err_like)
<<<<<<< HEAD

=======
        # value += sum_reduce(np.square(Err_like))
    # t2 = time.time()
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    if len(indices_dislike) > 0:
        # print(indices_dislike)
        dislike_vector = cf_user(rating_matrix, movie_vectors, current_vector, indices_dislike, K)
        dislike_vector = np.repeat(dislike_vector.reshape(1, -1), len(indices_dislike), axis=0)
        pre_dislike = np.dot(dislike_vector, movie_vectors.T)
        Err_dislike = (pre_dislike - dislike)[np.nonzero(dislike)]
        value += np.dot(Err_dislike, Err_dislike)

    if len(indices_unknown) > 0:
        # print(indices_unknown)
        unknown_vector = cf_user(rating_matrix, movie_vectors, current_vector, indices_unknown, K)
        unknown_vector = np.repeat(unknown_vector.reshape(1, -1), len(indices_unknown), axis=0)
        pre_unknown = np.dot(unknown_vector, movie_vectors.T)
        Err_unknown = (pre_unknown - unknown)[np.nonzero(unknown)]
        value += np.dot(Err_unknown, Err_unknown)
    # t2 = time.time()
<<<<<<< HEAD
    lkv_l = like_vector.flatten()
    dlkv_l = dislike_vector.flatten()
    unkv_l = unknown_vector.flatten()
    mov_l = movie_vectors.flatten()

    value += lmd_u * (np.dot(lkv_l, lkv_l) + np.dot(dlkv_l, dlkv_l) + np.dot(unkv_l, unkv_l))
    value += lmd_v * np.dot(mov_l, mov_l)
=======
    lkv_l = like_vector[np.nonzero(like_vector)]
    dlkv_l = dislike_vector[np.nonzero(dislike_vector)]
    unkv_l = unknown_vector[np.nonzero(unknown_vector)]
    mov_l = movie_vectors[np.nonzero(movie_vectors)]

    value += lmd_u * (vec_inner(lkv_l, lkv_l) + vec_inner(dlkv_l, dlkv_l) + vec_inner(unkv_l, unkv_l))
    value += lmd_v * vec_inner(mov_l, mov_l)
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6

    np.random.seed(0)
    num_pair = 20

    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(like_vector[u1], movie_vectors[i1] - movie_vectors[i2])
=======
                diff = vec_inner(like_vector[u1], movie_vectors[i1] - movie_vectors[i2])
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(dislike_vector[u1], movie_vectors[i1] - movie_vectors[i2])
=======
                diff = vec_inner(dislike_vector[u1], movie_vectors[i1] - movie_vectors[i2])
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(unknown_vector[u1], movie_vectors[i1] - movie_vectors[i2])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    # print(value)
=======
                diff = vec_inner(unknown_vector[u1], movie_vectors[i1] - movie_vectors[i2])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    return value


def cal_splitvalueI(rating_matrix, user_vectors, current_vector, indices_like, indices_dislike, indices_unknown, K):
    like = rating_matrix[:, indices_like]
    dislike = rating_matrix[:, indices_dislike]
    unknown = rating_matrix[:, indices_unknown]
    like_vector = np.zeros(K)
    dislike_vector = np.zeros(K)
    unknown_vector = np.zeros(K)
<<<<<<< HEAD
    value = 0.0

    if len(indices_like) > 0:
        like_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_like, K)
        like_vector = np.repeat(like_vector.reshape(1, -1), len(indices_like), axis=0)
        pre_like = np.dot(user_vectors, like_vector.T)
        Err_like = (pre_like - like)[np.nonzero(like)]
        value += np.dot(Err_like, Err_like)

    if len(indices_dislike) > 0:
        dislike_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_dislike, K)
        dislike_vector = np.repeat(dislike_vector.reshape(1, -1), len(indices_dislike), axis=0)
        pre_dislike = np.dot(user_vectors, dislike_vector.T)
        Err_like = (pre_dislike - dislike)[np.nonzero(dislike)]
        value += np.dot(Err_like, Err_like)

    if len(indices_unknown) > 0:
        unknown_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_unknown, K)
        unknown_vector = np.repeat(unknown_vector.reshape(1, -1), len(indices_unknown), axis=0)
        pre_unknown = np.dot(user_vectors, unknown_vector.T)
        Err_like = (pre_unknown - unknown)[np.nonzero(unknown)]
        value += np.dot(Err_like, Err_like)

    lkv_l = like_vector.flatten()
    dlkv_l = dislike_vector.flatten()
    unkv_l = unknown_vector.flatten()
    user_l = user_vectors.flatten()

    value += lmd_v * (np.dot(lkv_l, lkv_l) + np.dot(dlkv_l, dlkv_l) + np.dot(unkv_l, unkv_l))
    value += lmd_u * np.dot(user_l, user_l)
=======
    value = 0

    if len(indices_like) > 0:
        # print(indices_like)
        like_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_like, K)
        like_vector = np.array([like_vector for i in range(len(indices_like))])
        pre_like = np.dot(user_vectors, like_vector.T)
        Err_like = pre_like[np.nonzero(like)] - like[np.nonzero(like)]
        value = value + np.dot(Err_like, Err_like)
    # print(value)
    if len(indices_dislike) > 0:
        # print(indices_dislike)
        dislike_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_dislike, K)
        dislike_vector = np.array([dislike_vector for i in range(len(indices_dislike))])
        pre_dislike = np.dot(user_vectors, dislike_vector.T)
        Err_dislike = pre_dislike[np.nonzero(dislike)] - dislike[np.nonzero(dislike)]
        value = value + np.dot(Err_dislike, Err_dislike)
    # print(value)

    if len(indices_unknown) > 0:
        # print(indices_unknown)
        unknown_vector = cf_item(rating_matrix, user_vectors, current_vector, indices_unknown, K)
        unknown_vector = np.array([unknown_vector for i in range(len(indices_unknown))])
        pre_unknown = np.dot(user_vectors, unknown_vector.T)
        Err_unknown = pre_unknown[np.nonzero(unknown)] - unknown[np.nonzero(unknown)]
        value = value + np.dot(Err_unknown, Err_unknown)
    # print(value)

    lkv_l = like_vector[np.nonzero(like_vector)]
    dlkv_l = dislike_vector[np.nonzero(dislike_vector)]
    unkv_l = unknown_vector[np.nonzero(unknown_vector)]
    value = value + lmd_v * (np.dot(lkv_l, lkv_l) + np.dot(dlkv_l, dlkv_l) + np.dot(unkv_l, unkv_l))

    user_l = user_vectors[np.nonzero(user_vectors)]
    value = value + lmd_u * np.dot(user_l, user_l)
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6

    np.random.seed(0)
    num_pair = 20
    num_user, num_item = like.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if like[u1][i1] > like[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(user_vectors[u1] - user_vectors[u2], like_vector[i1])
=======
                diff = np.dot(user_vectors[u1, :].T, like_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                  like_vector[i2, :])
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = dislike.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if dislike[u1][i1] > dislike[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(user_vectors[u1] - user_vectors[u2], dislike_vector[i1])
=======
                diff = np.dot(user_vectors[u1, :].T, dislike_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                     dislike_vector[i2, :])
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))

    num_user, num_item = unknown.shape
    if num_user * num_item != 0:
        for i in range(num_pair):
            c1, c2 = np.random.randint(0, num_item * num_user, 2)
            u1, i1 = c1 // num_item, c1 % num_item
            u2, i2 = c2 // num_item, c2 % num_item
            if unknown[u1][i1] > unknown[u2][i2]:
<<<<<<< HEAD
                diff = np.dot(user_vectors[u1] - user_vectors[u2], unknown_vector[i1])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    # print(value)
=======
                diff = np.dot(user_vectors[u1, :].T, unknown_vector[i1, :]) - np.dot(user_vectors[u2, :].T,
                                                                                     unknown_vector[i2, :])
                diff = -diff
                value = value + lmd_BPR * np.log(1 + np.exp(diff))
    print(value)
>>>>>>> 6d0c0965c69a1779e4c4a31459d922c3b6978bf6
    return value
