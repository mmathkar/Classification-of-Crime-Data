from __future__ import with_statement
import sys
import os.path
import dtree
import id3
import math
import collections
import pandas as pd


# os.system("dtree.py")
# os.system("id3.py")

def entropy(data, target_attr):
    """
    Calculates the entropy of the given data set for the target attribute.
    """
    val_freq = {}
    data_entropy = 0.0

    # Calculate the frequency of each of the values in the target attr
    for record in data:
        # if (val_freq.has_key(record[target_attr])):
        if (record[target_attr] in val_freq):
            val_freq[record[target_attr]] += 1.0
        else:
            val_freq[record[target_attr]] = 1.0

    # Calculate the entropy of the data for the target attribute
    for freq in val_freq.values():
        data_entropy += (-freq / len(data)) * math.log(freq / len(data), 2)

    return data_entropy


def gain(data, attr, target_attr):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    """
    val_freq = {}
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        # if (val_freq.has_key(record[attr])):
        if (record[attr] in val_freq):
            val_freq[record[attr]] += 1.0
        else:
            val_freq[record[attr]] = 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob = val_freq[val] / sum(val_freq.values())
        data_subset = [record for record in data if record[attr] == val]
        subset_entropy += val_prob * entropy(data_subset, target_attr)

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    return (entropy(data, target_attr) - subset_entropy)


def majority_value(data, target_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    data = data[:]
    return most_frequent([record[target_attr] for record in data])


def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)

    return most_freq


def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)

    # Return the list with all redundant values removed.
    return unique_lst


def get_values(data, attr):
    """
    Creates a list of values in the chosen attribut for each record in data,
    prunes out all of the redundant values, and return the list.
    """
    data = data[:]
    return unique([record[attr] for record in data])


def choose_attribute(data, attributes, target_attr, fitness):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    data = data[:]
    best_gain = 0.0
    best_attr = None

    for attr in attributes:
        gain = fitness(data, attr, target_attr)
        if (gain >= best_gain and attr != target_attr):
            best_gain = gain
            best_attr = attr

    return best_attr


def get_examples(data, attr, value):
    """
    Returns a list of all the records in <data> with the value of <attr>
    matching the given value.
    """
    data = data[:]
    rtn_lst = []

    if not data:
        return rtn_lst
    else:
        record = data.pop()
        if record[attr] == value:
            rtn_lst.append(record)
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst
        else:
            rtn_lst.extend(get_examples(data, attr, value))
            return rtn_lst


def get_classification(record, tree):
    """
    This function recursively traverses the decision tree and returns a
    classification for the given record.
    """
    # If the current node is a string, then we've reached a leaf node and
    # we can return it as our answer
    if type(tree) == type("string"):
        return tree

    # Traverse the tree further until a leaf node is found.
    else:
        # attr = tree.keys()[0]
        attr = list(tree)[0]
        t = tree[attr][record[attr]]
        return get_classification(record, t)


def classify(tree, data):
    """
    Returns a list of classifications for each of the records in the data
    list as determined by the given decision tree.
    """
    data = data[:]
    classification = []

    for record in data:
        classification.append(get_classification(record, tree))

    return classification


def create_decision_tree(data, attributes, target_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = majority_value(data, target_attr)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
        return default
    # If all the records in the dataset have the same classification,
    # return that classification.
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        # Choose the next best attribute to best classify our data
        best = choose_attribute(data, attributes, target_attr,
                                fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        # We use the collections.defaultdict function to add a function to the
        # new tree that will be called whenever we query the tree with an
        # attribute that does not exist.  This way we return the default value
        # for the target attribute whenever, we have an attribute combination
        # that wasn't seen during training.
        tree = {best:collections.defaultdict(lambda: default)}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr,
                fitness_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree


def get_filenames():
    """
    Tries to extract the training and test data filenames from the command
    line.  If none are present, it prompts the user for both filenames and
    and makes sure that both exist in the system before returning the names
    back to the calling function.
    """
    # If no filenames were given at the command line, ask for them
    if len(sys.argv) < 3:
        training_filename = input("Training Filename: ")
        test_filename = input("Test Filename: ")
    # otherwise, read the filenames from the command line
    else:
        training_filename = sys.argv[1]
        test_filename = sys.argv[2]

    # This is a local function that takes a filename and returns true or false
    # depending on whether or not the file exists in the system.
    def file_exists(filename):
        if os.path.isfile(filename):
            return True
        else:
            print("Error: The file '%s' does not exist." % filename)
            return False

    # Make sure both files exist, otherwise print an error and exit execution
    if ((not file_exists(training_filename)) or
        (not file_exists(test_filename))):
        sys.exit(0)

    # Return the filenames of the training and test data files
    return training_filename, test_filename

def get_attributes(filename):
    """
    Parses the attribute names from the header line of the given file.
    """
    # Create a list of all the lines in the training file
    with open(filename, 'r') as fin:
        header = fin.readline().strip()

    # Parse the attributes from the header
    attributes = [attr.strip() for attr in header.split(",")]

    return attributes

def get_data(filename, attributes):
    """
    This function takes a file and list of attributes and returns a list of
    dict objects that represent each record in the file.
    """
    # Create a list of all the lines in the training file
    with open(filename) as fin:
        lines = [line.strip() for line in fin.readlines()]

    # Remove the attributes line from the list of lines
    del lines[0]

    # Parse all of the individual data records from the given file
    data = []
    for line in lines:
        data.append(dict(zip(attributes,
                             [datum.strip() for datum in line.split(",")])))
    
    return data
    
def print_tree(tree, str):
    """
    This function recursively crawls through the d-tree and prints it out in a
    more readable format than a straight print of the Python dict object.  
    """
    if type(tree) == dict:
        print ("%s%s" % (str, list(tree)[0]))
        # for item in tree.values()[0].keys():
        for item in list(tree.values())[0]:
            print("%s\t%s" % (str, item))
            print_tree(list(tree.values())[0][item], str + "\t")
    else:
        print("%s\t->\t%s" % (str, tree))


# if __name__ == "__main__":
#     # Get the training and test data filenames from the user
#     training_filename, test_filename = get_filenames()
#
#
#
#     # Extract the attribute names and the target attribute from the training
#     # data file.
#     attributes = get_attributes(training_filename)
#     target_attr = attributes[-1]
#
#     # Get the training and test data from the given files
#     training_data = get_data(training_filename, attributes)
#     test_data = get_data(test_filename, attributes)
#
#     # Create the decision tree
#     dtree = create_decision_tree(training_data, attributes, target_attr, gain)
#
#     # Classify the records in the test data
#     classification = classify(dtree, test_data)
#
#     # Print the results of the test
#     print("------------------------\n")
#     print("--   Classification   --\n")
#     print("------------------------\n")
#     print("\n")
#     for item in classification: print(item)
#
#     # Print the contents of the decision tree
#     print("\n")
#     print("------------------------\n")
#     print("--   Decision Tree    --\n")
#     print("------------------------\n")
#     print("\n")
#     print_tree(dtree, "")

if __name__ == "__main__":
    # Get the training and test data filenames from the user
    training_filename, test_filename = get_filenames()

    ca = pd.read_csv("C:\\Users\\mathk\\fall2017\\IDM\\dtree-master\\dtree-master\\iris-dataset.csv")

    # training_set = ca.ilock[:10,:]
    trainX = ca.iloc[:int(len(ca) * 0.8), :4]
    testX = ca.iloc[int(len(ca) * 0.8):, :4]
    trainY = ca.ix[:int(len(ca) * 0.8), 4]
    testY = ca.ix[int(len(ca) * 0.8):, 4]
    result_df = pd.Series()

    # Extract the attribute names and the target attribute from the training
    # data file.
    attributes = get_attributes(trainX)
    target_attr = attributes[-1]

    # Get the training and test data from the given files
    training_data = get_data(trainX, attributes)
    test_data = get_data(testY, attributes)

    # Create the decision tree
    dtree = create_decision_tree(training_data, attributes, target_attr, gain)

    # Classify the records in the test data
    classification = classify(dtree, test_data)

    # Print the results of the test
    print("------------------------\n")
    print("--   Classification   --\n")
    print("------------------------\n")
    print("\n")
    for item in classification: print(item)

    # Print the contents of the decision tree
    print("\n")
    print("------------------------\n")
    print("--   Decision Tree    --\n")
    print("------------------------\n")
    print("\n")
    print_tree(dtree, "")
