# Import the Required Libraries
import autograd.numpy as np
import math
# import form user defined libraries
import decision_tree_f as dtree
import optimization as opt


def getRatingMatrix(filename):
    # Open the file for reading data
    file = open(filename, "r")

    while 1:
        # Read all the lines from the file and store it in lines
        lines = file.readlines(1000000000)

        # if Lines is empty, simply break
        if not lines:
            break

        # Create a list to hold all the data
        data = []
        data_fo = []

        print("Number of Lines: ", len(lines))

        # For each Data Entry, get the rating and the f-o pairs in their respective list
        for line in lines:
            #print("dealing with new line")
            # Get all the attributes by splitting on ','
            list1 = line.split("\n")[0].split(",")
            list2 = list1.pop()
            # list1 store userid itemid and rating
            # list2 store all f-o pair for each userid
            list2 = list2.split(" ")
            list2 = [int(j) for j in list2]
            list1 = [int(j) for j in list1]
            # Add to the data
            data.append(list1)
            data_fo.append(list2)

        index_f = []
        for i in data_fo:
            index_f.extend(i)

        index_f = np.array(index_f)
        index_f = index_f[np.argmax(index_f)]+1
        print("Number of feature : ", index_f)
        # convert data into numpy form
        data = np.array(data)

        # Get the indices of the maximum Values in each column
        a = np.argmax(data, axis=0)
        # print(a)
        num_users = data[a[0]][0]+1
        num_items = data[a[1]][1]+1

        # print "Max values Indices: ", a
        print("Number of Users: ", num_users)
        print("Number of Items: ", num_items)

        # print(data_fo)
        # print(data)

        ratingMatrix = np.zeros((num_users, num_items), dtype=float)
        opinionMatrix = np.zeros((num_users, index_f), dtype=float)
        opinionMatrix_I = np.zeros((num_items,index_f),dtype=float)
        # print(len(data))
        # print(len(data_fo))

        for i in range(len(data)):
            list1 = data[i] # userid itemid rating in line i
            list2 = data_fo[i] # all f-o pair in line i
            ratingMatrix[list1[0]][list1[1]] = list1[2]
            for j in range(0, len(list2), 2):
                # list2[j] is feature_id list2[j+1] is value of opinion
                opinionMatrix[list1[0]][list2[j]] = opinionMatrix[list1[0]][list2[j]] + list2[j+1]
                opinionMatrix_I[list1[1]][list2[j]] = opinionMatrix_I[list1[1]][list2[j]] + list2[j+1]
        #print(ratingMatrix)
        #print(opinionMatrix)
        for i in range(len(opinionMatrix)):
            for j in range(len(opinionMatrix[0])):
                if opinionMatrix[i][j] > 0:
                    opinionMatrix[i][j] = 1
                if opinionMatrix[i][j] < 0:
                    opinionMatrix[i][j] = -1
                if opinionMatrix[i][j] == 0:
                    opinionMatrix[i][j] = 0

        for i in range(len(opinionMatrix_I)):
            for j in range(len(opinionMatrix_I[0])):
                if opinionMatrix_I[i][j] > 0:
                    opinionMatrix_I[i][j] = 1
                if opinionMatrix_I[i][j] < 0:
                    opinionMatrix_I[i][j] = -1
                if opinionMatrix_I[i][j] == 0:
                    opinionMatrix_I[i][j] = 0
        return ratingMatrix, opinionMatrix,opinionMatrix_I


# Function to calculate the RMSE Error between the predicted and actual rating
'''def getRMSE(Actual_Rating, Predicted_Rating):
    # Calculate the Root Mean Squared Error(RMSE)
    rmse = 0.0
    for i in range(len(Actual_Rating)):
        for j in range(len(Actual_Rating[0])):
            if Actual_Rating[i][j] > 0:
                rmse = rmse + pow((Actual_Rating[i][j] - Predicted_Rating[i][j]), 2)

    rmse = rmse * 1.0 / (len(Actual_Rating) * len(Actual_Rating[0]))  # recorrect
    rmse = math.sqrt(rmse)

    # Print and return the RMSE
    print('Root Mean Squared Error(RMSE) = ', rmse)
    return rmse'''


'''def getNDCG(predict, real, N):
    NDCG = []
    predict = np.array(predict)
    real = np.array(real)
    for i in range(len(predict)):
        arg_pre = np.argsort(-predict[i])
        rec_pre = real[i][arg_pre]
        rec_pre = [rec_pre[k] for k in range(N)] # value of real rating with Top N predict recommendation
        #rec_pre = np.array(rec_pre)
        arg_real = np.argsort(-real[i]) # ideal ranking of real rating with Top N
        rec_real = real[i][arg_real]
        rec_real = [rec_real[k] for k in range(N)]
        #print("rec_pre",rec_pre)
        #print("rec_real",rec_real)
        dcg = 0
        idcg = 0
        for j in range(N):
            dcg = dcg + rec_pre[j]/math.log2(j+2)
            idcg = idcg + rec_real[j]/math.log2(j+2)
        NDCG.append(dcg/idcg)
    print(NDCG)
    sum = 0
    for i in range(len(NDCG)):
        sum = sum +NDCG[i]
    ndcg = sum/len(NDCG)
    return ndcg'''

# Used to randomly split the data by row
def random_split(rating_matrix, opinion_matrix):
    # Split the data set into 75% and 25%
    SPLIT_PERCENT = 0.75

    # Get Random Indices to shuffle the rows around
    indices = np.random.permutation(rating_matrix.shape[0])
    # Random lists of row indices

    # Get the number of rows
    num_rows = len(rating_matrix[:, 0])

    # Get the indices for training and testing sets
    training_indices, test_indices = indices[: int(SPLIT_PERCENT * num_rows)], indices[int(SPLIT_PERCENT * num_rows):]

    # return the training and the test set
    return rating_matrix[training_indices, :], rating_matrix[test_indices, :], opinion_matrix[training_indices, :], opinion_matrix[test_indices,:]


# Returns the rating Matrix with approximated ratings for all users for all items using fMf
def alternateOptimization(opinion_matrix,opinion_matrix_I, rating_matrix, NUM_OF_FACTORS, MAX_DEPTH):
    # Save and print the Number of Users and Movies
    NUM_USERS = rating_matrix.shape[0]
    NUM_MOVIES = rating_matrix.shape[1]
    NUM_FEATURE = opinion_matrix.shape[1]
    print("Number of Users", NUM_USERS)
    print("Number of Item", NUM_MOVIES)
    print("Number of Feature",NUM_FEATURE)
    print("Number of Latent Factors: ", NUM_OF_FACTORS)

    # Create the user and item profile vector of appropriate size.
    # Initialize the item vectors randomly, check the random generation
    user_vectors = np.random.rand(NUM_USERS, NUM_OF_FACTORS)
    item_vectors = np.random.rand(NUM_MOVIES, NUM_OF_FACTORS)

    i = 0

    print("Entering Main Loop of alternateOptimization")

    decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)

    # Do converge Check
    while i < 20:
        # Create the decision Tree based on item_vectors
        print("Creating Tree.. for i = ", i, "for user")
        decTree = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        decTree.fitTree_U(decTree.root, opinion_matrix, rating_matrix, item_vectors, NUM_OF_FACTORS)

        print("Getting the user vectors from tree")
        # Calculate the User vectors using dtree
        user_vectors_before = user_vectors
        user_vectors = decTree.getVectors_f(opinion_matrix, NUM_OF_FACTORS)

        print("Creating Tree.. for i = ", i, "for item")
        decTreeI = dtree.Tree(dtree.Node(None, 1), NUM_OF_FACTORS, MAX_DEPTH)
        decTreeI.fitTree_I(decTreeI.root, opinion_matrix_I, rating_matrix, user_vectors, NUM_OF_FACTORS)

        print("Getting the item vectors from tree")
        item_vectors_before = item_vectors
        item_vectors = decTreeI.getVectors_f(opinion_matrix_I, NUM_OF_FACTORS)

        # Calculate Error for Convergence check
        error_u = 0
        for iteri in range(NUM_USERS):
            for iterj in range(NUM_OF_FACTORS):
                du = user_vectors[iteri][iterj] - user_vectors_before[iteri][iterj]
                error_u = error_u + du * du
        error_u = math.sqrt(error_u)
        error_v = 0
        for iteri in range(NUM_MOVIES):
            for iterj in range(NUM_OF_FACTORS):
                dv = item_vectors[iteri][iterj] - item_vectors_before[iteri][iterj]
                error_v = error_v + dv * dv
        error_v = math.sqrt(error_v)
        print("error_u ", error_u)
        print("error_v ", error_v)
        if error_v < 0.1:
            if error_u < 0.1:
                break
        i = i + 1

    return decTree, decTreeI, item_vectors.T


def printTopKMovies(test, predicted, K):
    # Gives top K  recommendations
    print("Top Movies Not rated by the user")

    for i in range(len(test)): # for each user
        zero_list = []
        item_list = []
        for j in range(len(test[0])): # for each item
            if test[i][j] == 0:
                zero_list.append(predicted[i][j])  # rating value
                item_list.append(j)  # item index

            zero_array = np.array(zero_list)
            item_array = np.array(item_list)

            args = np.argsort(zero_array)
            item_array = item_array[args]
        if K < len(item_array):
            print("user ", i, " : ", item_array[0:K])
        else:
            print("user", i, " : ", item_array)


if __name__ == "__main__":
    # Get the Data
    (rating_matrix, opinion_matrix, opinion_matrixI) = getRatingMatrix("test.txt")

    print("Dimensions of the Dataset: ", rating_matrix.shape)
    train_rating = rating_matrix

    # Split the data 80-20 into training and testing dataset
    #(train_r, test_r, train_o, test_o,) = random_split(rating_matrix, opinion_matrix)
    #print("Dimensions of the Training Set: ", train_r.shape)
    #print("Dimensions of the Testing Set: ", test_r.shape)
    num_user,num_item = rating_matrix.shape
    num_test = int(0.2*num_user*num_item)
    print("Number of test set: ",num_test)
    index_row = []
    index_col = []
    np.random.seed(0)
    for i in range(num_test):
        c1 = np.random.randint(0, num_item * num_user)
        u1,i1 = c1 // num_item, c1 % num_item
        index_row.append(u1)
        index_col.append(c1)
        train_rating[u1][i1] = 0

    test_indices = (np.array(index_row), np.array(index_col))

    # Set the number of Factors
    NUM_OF_FACTORS = 20
    MAX_DEPTH = 6

    # Build decision tree on training set
    (decisionTree, decisionTreeI, item_vector) = alternateOptimization(opinion_matrix, opinion_matrixI, train_rating, NUM_OF_FACTORS, MAX_DEPTH)
    np.savetxt('item_vector.txt', item_vector, fmt='%0.8f')
    # Traverse the tree with the opinion to get user_profile
    user_vectors = decisionTree.getVectors_f(opinion_matrix, NUM_OF_FACTORS)
    np.savetxt('user_vectors.txt', user_vectors, fmt='%0.8f')
    Predicted_Rating = np.dot(user_vectors, item_vector)
    np.savetxt("rating_predict.txt", Predicted_Rating, fmt='%0.8f')
    test_r = rating_matrix[test_indices]
    Predicted_Rating = Predicted_Rating[test_indices]
    # print("Predicted_Rating for Test: ", Predicted_Rating)
    # print("Test Rating: ", test)
    '''RMSE = getRMSE(test_r, Predicted_Rating)
    RMSE = np.array([RMSE])
    NDCG = getNDCG(Predicted_Rating, test_r, 10)
    print("NDCG@10: ", NDCG)
    NDCG = getNDCG(Predicted_Rating, test_r, 20)
    print("NDCG@20: ", NDCG)
    NDCG = getNDCG(Predicted_Rating, test_r, 50)
    print("NDCG@50: ", NDCG)
    np.savetxt("RMSE.txt", RMSE,fmt="%0.8f")'''
    print("print user tree")
    decisionTree.printtree(decisionTree.root)
    print("print item tree")
    decisionTree.printtree(decisionTreeI.root)

    # Top K new recommendations:
    # printTopKMovies(test_r, Predicted_Rating, 5)

