import copy
import pickle

import nltk
from nltk import WordNetLemmatizer

from string_processing import (
    process_tokens,
    tokenize_text,
)


def intersect_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) intersection algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    # Set two indexes for two different list, when the elements of two index position equals put the element in res list
    # and then add two index by 1. If the element in doc_list1 < element in doc_list2, then index for the first add 1
    # else the index for the second index add 1
    res = []
    i, j = 0, 0
    while i < len(doc_list1) and j < len(doc_list2):
        # put relative value in the result list and continue moving the two pointers
        if doc_list1[i] == doc_list2[j]:
            res.append(doc_list1[i])
            i += 1
            j += 1
        else:
            if doc_list1[i] < doc_list2[j]:
                i += 1
            else:
                j += 1
    return res


def union_query(doc_list1, doc_list2):
    # TODO: you might like to use a function like this in your run_boolean_query implementation
    # for full marks this should be the O(n + m) union algorithm for sorted lists
    # using data structures such as sets or dictionaries in this function will not score full marks

    # Set two indexes for two different list, when the elements of two index position equals put the element in res list
    # and then add two index by 1. If the element in doc_list1 < element in doc_list2, then put the element in doc_list1
    # in res list and index for the first add 1, else put the element in doc_list2 in res list and the index for the second
    # index add 1. In the end, add the rest element in res list if there is one list longer

    res = []
    i, j = 0, 0
    while i < len(doc_list1) or j < len(doc_list2):
        # put the rest of longer one in the result list
        if i > len(doc_list1) - 1:
            res.append(doc_list2[j])
            j += 1
            continue
        if j > len(doc_list2) - 1:
            res.append(doc_list1[i])
            i += 1
            continue
        # put relative value in the result list and continue moving the two pointers
        if doc_list1[i] == doc_list2[j]:
            res.append(doc_list1[i])
            i += 1
            j += 1
        else:
            if doc_list1[i] < doc_list2[j]:
                res.append(doc_list1[i])
                i += 1
            else:
                res.append(doc_list2[j])
                j += 1

    return res


def run_boolean_query(query_string, index):
    # Create final list which contains doc ids
    relevant_docs = []
    # Define a list to contain doc ids for each token except OR and AND
    token_list = []
    # Define a list to filter OR and AND
    boolean_List = ['and', 'or']

    # Tokenize the query string
    query_tokens = tokenize_text(query_string)

    # Normalisation to make all words lowercase
    query_tokens = [element.lower() for element in query_tokens]

    # Iteration each token in query, create a list to contain relative doc ids for doc which contain token
    # Then, intersect or union token list depend on 'OR' or 'AND' between them from left to right
    for i in range(len(query_tokens)):
        # Clear up the token_list for different token
        token_list = []
        # Put aimed doc ids in token_list and these doc contain token
        if query_tokens[i] in index and query_tokens[i] not in boolean_List:
            for tuple in index.get(query_tokens[i]):
                token_list.append(tuple[0])

            # If this is the first token, the relevant doc id at present is token_list
            if i == 0:
                relevant_docs = copy.deepcopy(token_list)
                continue

            # If the token has relationship with the front token, which means it's not the first one
            if query_tokens[i - 1] in boolean_List:
                if query_tokens[i - 1] == 'or':
                    relevant_docs = copy.deepcopy(union_query(relevant_docs, token_list))
                if query_tokens[i - 1] == 'and':
                    relevant_docs = copy.deepcopy(intersect_query(relevant_docs, token_list))
    return relevant_docs


if __name__ == '__main__':
    # load the stored index
    (index, doc_freq, doc_ids, num_docs) = pickle.load(open("stored_index.pkl", "rb"))

    print("Index length:", len(index))
    if len(index) != 808777:
        print("Warning: the length of the index looks wrong.")
        print("Make sure you are using `process_tokens_original` when you build the index.")
        raise Exception()

    # the list of queries asked for in the assignment text
    queries = [
        "Workbooks",
        "Australasia OR Airbase",
        "Warm AND WELCOMING",
        "Global AND SPACE AND economies",
        "SCIENCE OR technology AND advancement AND PLATFORM",
        "Wireless OR Communication AND channels OR SENSORY AND INTELLIGENCE",
    ]
    # run each of the queries and print the result
    ids_to_doc = {docid: path for (path, docid) in doc_ids.items()}
    for query_string in queries:
        print(query_string)
        doc_list = run_boolean_query(query_string, index)
        res = sorted([ids_to_doc[docid] for docid in doc_list])
        for path in res:
            print(path)
