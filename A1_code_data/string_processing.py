import nltk
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer


def process_tokens(toks):
    # TODO: fill in the functions: process_tokens_1, 
    # process_tokens_2, process_tokens_3 functions and
    # uncomment the one you want to test below 
    # make sure to rebuild the index

    # return process_tokens_1(toks)
    # return process_tokens_2(toks)
    #return process_tokens_3(toks)
    #return process_tokens_4(toks)
    return process_tokens_original(toks)


# get the nltk stopwords list
stopwords = set(nltk.corpus.stopwords.words("english"))

def process_tokens_original(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        new_toks.append(t)
    return new_toks


def process_tokens_1(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        # do stemming for token
        stemmer = PorterStemmer()
        t = stemmer.stem(t)
        #TODO: your code should modify t and/or do some sort of filtering

        new_toks.append(t)
    return new_toks


def process_tokens_2(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        # do lemmatisation for token
        lemmatizer = WordNetLemmatizer()
        t = lemmatizer.lemmatize(t)

        #TODO: your code should modify t and/or do some sort of filtering

        new_toks.append(t)
    return new_toks


def process_tokens_3(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        # remove symbols except letters and numbers in token
        t = ''.join(c for c in t if c.isalnum())
        #TODO: your code should modify t and/or do some sort of filtering

        new_toks.append(t)
    return new_toks

def process_tokens_4(toks):
    """ Perform processing on tokens. This is the Linguistics Modules
    phase of index construction

    Args:
        toks (list(str)): all the tokens in a single document

    Returns:
        list(str): tokens after processing
    """
    new_toks = []
    for t in toks:
        # ignore stopwords
        if t in stopwords or t.lower() in stopwords:
            continue
        # lowercase token
        t = t.lower()
        # do lemmatisation and remove symbols except letters and numbers in token
        t = ''.join(c for c in t if c.isalnum())
        lemmatizer = WordNetLemmatizer()
        t = lemmatizer.lemmatize(t)
        #TODO: your code should modify t and/or do some sort of filtering

        new_toks.append(t)
    return new_toks


def tokenize_text(data):
    """Convert a document as a string into a document as a list of
    tokens. The tokens are strings.

    Args:
        data (str): The input document

    Returns:
        list(str): The list of tokens.
    """
    # split text on spaces
    tokens = data.split()
    return tokens

