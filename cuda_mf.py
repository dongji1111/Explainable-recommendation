import numpy as np
# import matplotlib.pyplot as plt
from numba import cuda
np.random.seed(1234)
cuda.select_device(0)

global stream, dR, dP, dU, dI


@cuda.jit
def predict(U, I, P):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    x = tx + bx * bw
    y = ty + by * bh

    P[x, y] = 0
    for k in range(K):
        P[x, y] += U[x, k] * I[k, y]
    cuda.syncthreads()


@cuda.jit
def error(R, P, U, I, Err, UReg, IReg):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    Err[i, j] = 0
    if R[i, j] != 0:
        for k in range(K):
            UReg[i, k] = U[i, k] ** 2
            IReg[k, j] = I[k, j] ** 2
        Err[i, j] = (R[i, j] - P[i, j]) ** 2


# @cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
@cuda.jit
def factorize(R, P, U, I):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    if R[i, j] != 0:
        for k in range(K):
            err_U = 0
            for c in range(numberOfItems):
                if R[i, c] != 0:
                    err_U += (R[i, c] - P[i, c]) * I[k, c]
            err_I = 0
            for c in range(numberOfUsers):
                if R[c, j] != 0:
                    err_I += (R[c, j] - P[c, j]) * U[c, k]

            U[i, k] += alpha * 2 * (err_U - beta * U[i, k])
            I[k, j] += alpha * 2 * (err_I - beta * I[k, j])


# data
# K = 20
# file = open("yelp_train.txt", 'r')
# lines = file.readlines()
# numberOfUsers = 0
# numberOfItems = 0
# userID = np.zeros((len(lines)), dtype=int)
# itemID = np.zeros((len(lines)), dtype=int)
# rating = np.zeros((len(lines)))
# count = 0

# print("Preparing data.........")
# for line in lines:
#     listOfLine = line.split("\n")[0].split(",")
#     userID[count] = int(listOfLine[0])
#     # print(userID[count])

#     if userID[count] + 1 > numberOfUsers:
#         numberOfUsers = userID[count] + 1

#     itemID[count] = int(listOfLine[1])
#     # print(itemID[count])
#     if itemID[count] + 1 > numberOfItems:
#         numberOfItems = itemID[count] + 1
#     rating[count] = float(listOfLine[2])
#     count = count + 1
# rating_matrix = np.zeros((numberOfUsers, numberOfItems))
# for line in lines:
#     listOfLine = line.split("\n")[0].split(",")
#     uID = int(listOfLine[0])
#     iID = int(listOfLine[1])
#     r = float(listOfLine[2])
#     rating_matrix[uID, iID] = r
# print("Finish preparing data")

# random
K = 2
numberOfUsers = 50
numberOfItems = 50
rating_matrix = np.random.rand(numberOfUsers, numberOfItems)

# begin
hR = rating_matrix
hU = np.asarray(np.random.rand(numberOfUsers, K), dtype=np.float32)
hI = np.asarray(np.random.rand(K, numberOfItems), dtype=np.float32)
hP = np.zeros(rating_matrix.shape)

hErr = np.zeros((numberOfUsers, numberOfItems), dtype=np.float32)
hUReg = np.zeros((numberOfUsers, K), dtype=np.float32)
hIReg = np.zeros((K, numberOfItems), dtype=np.float32)

stream = cuda.stream()
with stream.auto_synchronize():
    dR = cuda.to_device(hR)
    dU = cuda.to_device(hU)
    dI = cuda.to_device(hI)
    dP = cuda.to_device(hP)
    dErr = cuda.to_device(hErr)
    dUReg = cuda.to_device(hUReg)
    dIReg = cuda.to_device(hIReg)


alpha = 0.05
beta = 0.02

blockdim = (32, 32)
griddim = (numberOfUsers // blockdim[0] + 1, numberOfItems // blockdim[1] + 1)


errors = []
for i in range(100):
    predict[griddim, blockdim, stream](dU, dI, dP)
    stream.synchronize()

    factorize[griddim, blockdim, stream](dR, dP, dU, dI)
    stream.synchronize()

    error[griddim, blockdim, stream](dR, dP, dU, dI, dErr, dUReg, dIReg)
    stream.synchronize()

    Err = np.sum(dErr.copy_to_host())
    UReg = np.sum(dUReg.copy_to_host())
    IReg = np.sum(dIReg.copy_to_host())
    errors.append(Err + beta * (UReg + IReg))
    print(Err)
