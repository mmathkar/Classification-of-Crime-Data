from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import random as np
import pandas as pd


def load_data_set(filename):
    try:
        with open(filename, newline='') as iris:
            return list(reader(iris, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data)-1]] + [data[len(data)-1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


def get_classes(training_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def knn2(input_data, training_set, labels, k=1):
    distance_diff = training_set - input_data
    distance_squared = distance_diff ** 2
    distance = distance_squared.sum(axis=1) ** 0.5
    distance_df = pd.concat([distance, labels], axis=1)
    colname = list(distance_df)[0]
    distance_df.sort_values(by = [colname], inplace=True)
    top_knn = distance_df[:k]
    ser = top_knn.ix[:,1]
    # maxfreq= top_knn[1].value_counts()
    return ser.value_counts().index.values[0]



def knn(training_set, test_set, k):
    distances = []
    dist = 0
    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in test_set:
            for row in training_set:
                for x, y in zip(row[:limit], test_instance):
                    dist += (x-y) * (x-y)
                distances.append(row + [sqrt(dist)])
                dist = 0

            distances.sort(key=itemgetter(len(distances[0])-1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)

            # Display prediction
            print('The predicted class for sample ' + str(test_instance) + ' is : ' + classes[index])
            print('Number of votes : ' + str(value) + ' out of ' + str(k))

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)



def main():
    try:
        # get value of k
        k = int(input('Enter the value of k : '))

        # load the training and test data set
        # training_file = input('Enter name of training data file : ')
        # test_file = input('Enter name of test data file : ')

        ca=pd.read_csv("C:\\Users\\mathk\\fall2017\\IDM\\k-nearest-neighbors-master\\k-nearest-neighbors-master\\iris-dataset.csv")

        # training_set = ca.ilock[:10,:]
        trainX = ca.iloc[:int(len(ca) * 0.8), :4]
        testX = ca.iloc[int(len(ca) * 0.8):, :4]
        trainY = ca.ix[:int(len(ca) * 0.8), 4]
        testY = ca.ix[int(len(ca) * 0.8):, 4]
        result_df = pd.Series()
        # for i in range(len(trainX)):
        #     result_df.iloc[i] = knn2(trainX.iloc[i, :], trainX, trainY, k=3)
        result_df = testX.apply(lambda row: knn2(row, trainX, trainY, k=3), axis =1)
        error_df = result_df == testY
        print(error_df.value_counts())
        # test_set = ca.ilock[10:,:]

        # training_labels = pd.Series(raw_training_labels)
        # training_data = pd.DataFrame.from_records(np.array(raw_training_data, int))
        #
        # test_labels = pd.Series(raw_test_labels)
        # test_data = pd.DataFrame.from_records(np.array(raw_test_data, int))
        #
        # # Apply kNN algorithm to all test data
        # result_df = test_data.apply(lambda row: classify(row, training_data, training_labels, k=3), axis=1)


        # training_set = convert_to_float(load_data_set(training_file), 'training')
        # test_set = convert_to_float(load_data_set(test_file), 'test')


        # file = open("datafile.txt", "r")
        # data = list()
        # for line in file:
        #     data.append(line.split(','))
        #     file.close()
        #     random.shuffle(data)
        #     train_data = data[:int((len(data) + 1) * .80)]  # Remaining 80% to training set
        #     test_data = data[int(len(data) * .80 + 1):]  # Splits 20% data to test set

        # knn(training_set, test_set, k)

        # Apply kNN algorithm to all test data
        # result_df = testX.apply(lambda row: knn2(row, trainX, trainY, k=3))

        # if not training_set:
        #     print('Empty tr3aining set')
        #
        # elif not test_set:
        #     print('Empty3
        #  test set')
        #
        # elif k > len(training_set):
        #     print('Expected number of neighbors is higher than number of training data instances')
        #```````````````````````````````````````````````
        # else:
        #     knn(training_set, test_set, k)

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print('File not found')


if __name__ == '__main__':
    main()
