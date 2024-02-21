import math
from collections import defaultdict
from string_processing import (
    process_tokens,
    tokenize_text,
)
from query import (
    get_query_tokens,
    count_query_tokens,
    query_main,
)


def get_doc_to_norm(index, doc_freq, num_docs):
    """Pre-compute the norms for each document vector in the corpus using tfidf.

    Args:
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        num_docs (int): number of documents in the corpus

    Returns:
        dict(int: float): a dictionary mapping doc_ids to document norms
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the get_doc_to_norm function in query.py
    #       but should use tfidf instead of term frequency

    doc_norm = defaultdict(float)

    # calculate square of norm for all docs
    for term in index.keys():
        for (docid, tf) in index[term]:
            doc_freq_term = doc_freq[term]
            right = math.log(num_docs / (1 + doc_freq_term))
            doc_norm[docid] += (tf * right) ** 2

    # take square root
    for docid in doc_norm.keys():
        doc_norm[docid] = math.sqrt(doc_norm[docid])

    return doc_norm


def run_query(query_string, index, doc_freq, doc_norm, num_docs):
    """ Run a query on the index and return a sorted list of documents. 
    Sorted by most similar to least similar.
    Documents not returned in the sorted list are assumed to have 0 similarity.

    Args:
        query_string (str): the query string
        index (dict(str : list(tuple(int, int)))): The index aka dictionary of posting lists
        doc_freq (dict(str : int)): document frequency for each term
        doc_norm (dict(int : float)): a map from doc_ids to pre-computed document norms
        num_docs (int): number of documents in the corpus

    Returns:
        list(tuple(int, float)): a list of document ids and the similarity scores with the query
        sorted so that the most similar documents to the query are at the top.
    """

    # TODO: Implement this function using tfidf
    # Hint: This function is similar to the run_query function in query.py
    #       but should use tfidf instead of term frequency

    # pre-process the query string
    qt = get_query_tokens(query_string)
    query_token_counts = count_query_tokens(qt)

    # calculate the norm of the query vector
    query_norm = 0
    for (term, tf) in query_token_counts:
        # ignore term if not in index (to be comparable to doc_norm)
        # note that skipping this will not change the rank of retrieved docs
        if term not in index:
            continue
        # calculate query norm
        doc_freq_term = doc_freq[term]
        right = math.log(num_docs / (1 + doc_freq_term))
        query_norm += (tf * right) ** 2
    query_norm = math.sqrt(query_norm)

    # calculate cosine similarity for all relevant documents
    doc_to_score = defaultdict(float)
    for (term, tf_query) in query_token_counts:
        # ignore query terms not in the index
        if term not in index:
            continue
        # add to similarity for documents that contain current query word
        for (docid, tf_doc) in index[term]:
            # calculate cosine similarity of TF
            doc = tf_query * tf_doc / (doc_norm[docid] * query_norm)
            # calculate idf
            doc_freq_term = doc_freq[term]
            idf = math.log(num_docs / (1 + doc_freq_term))
            # add idf to cosine similarity of TF
            doc_to_score[docid] += doc * idf**2

    sorted_docs = sorted(doc_to_score.items(), key=lambda x: -x[1])
    return sorted_docs


if __name__ == '__main__':
    queries = [
        'Is nuclear power plant eco-friendly?',
        'How to stay safe during severe weather?',
    ]
    query_main(queries=queries, query_func=run_query, doc_norm_func=get_doc_to_norm)
