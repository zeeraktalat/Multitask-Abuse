from typing import Tuple, Callable, List, Dict, Union
from collections import Counter
from nltk import ngrams
import re


def unigrams(doc: List[str]) -> Dict[str, int]:
    """Compute unigrams.
    :param doc: Document to be parsed.
    :return: counted ngrams.
    """
    return Counter(doc)


def n_grams(doc: Union[str, List[str]], n: int = 2) -> Dict[str, int]:
    """
    :param doc: Document to create ngrams from.
    :param n: Size of ngrams.
    :return: Counter object of counted token bigrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, n)]) + unigrams(doc)


def char_ngrams(doc: str, n: int = 2) -> Dict[str, int]:
    """
    :param doc: Document to character create ngrams from.
    :return: Counter object of counted char ngrams.
    """
    return Counter(["_".join(gram) for gram in ngrams(doc, 2)])


def char_count(doc: str) -> Dict[str, int]:
    """
    :param doc: Document to create character counts from.
    :return: Counter dict of counted characters in document.
    """
    return Counter(doc)


def find_mentions(doc: str) -> List[str]:
    """
    :param doc: Document to find mentions in.
    :return: List of mentions.
    """
    return re.findall(r'@[a-zA-Z0-9]', doc)


def find_hashtags(doc: str) -> List[str]:
    """
    :param doc: Document to find hashtags in.
    :return: List of hashtags.
    """
    return re.findall(r'#[a-zA-Z0-9]', doc)


def find_urls(doc: str) -> List[str]:
    """
    :param doc: Document to find URLs in.
    :return: List of urls.
    """
    return re.findall(r'http:\/\/[a-zA-Z0-9\.\/a-zA-Z0-9]+', doc)


def find_retweets(doc: str) -> List[str]:
    """
    :param doc: Document to find retweets in.
    :return: List of retweets.
    """
    return re.findall(r'\wRT\w @', doc)


def count_syllables(doc: List[str]) -> int:
    """Simplistic syllable count.
    :param doc: Document to count syllables in.
    :return: Count of syllables.
    """
    count = 0
    vowels = 'aeiouy'
    exceptions = ['le', 'es', 'e']
    for word in doc:
        prev_char = None
        for i in word:
            if i == len(word) and (prev_char + word[i] in exceptions or word[i] in exceptions):
                prev_char = word[i]
                continue
            if (word[i] in vowels) and (prev_char not in vowels and not prev_char):
                prev_char = word[i]
                count += 1
    return {'NO_SYLLABLES': count}


def word_list(doc: List[str], word_list: List[str], salt: str) -> Dict[str, int]:
    """Identify if words are in word_list.
    :param doc: Tokenised document.
    :param word_list: List containing words occurring in the wordlist.
    :param salt: To add in front of word name.
    :return: Counts of each word appearing in dict.
    """
    salt += '_WORDLIST_'
    res = []
    for w in doc:
        if w in word_list:
            res.append(salt + w)
    return Counter(res) if len(res) != 0 else {salt: 0}


def _pos_helper(docs: List[str]) -> Tuple[List[str], List[str], List[str]]:
    # for doc in tqdm(docs, desc = "POS helper"):
    for doc in docs:
        tokens = []
        pos = []
        confidence = []
        for tup in doc:
            tokens.append(tup[0])
            pos.append(tup[1])
            confidence.append(tup[2])
        yield tokens, pos, confidence


def sentiment_polarity(doc: str, sentiment: Callable) -> Dict[str, float]:
    """Compute sentiment polarity scores and return features.
    :param doc: Document to be computed for.
    :param sentiment: Callable sentiment analysis method.
    :return features: Features dict to return.
    """
    features = {}
    polarity = sentiment.polarity_scores(doc)
    features.update({'SENTIMENT_POS': polarity['pos']})
    features.update({'SENTIMENT_NEG': polarity['neg']})
    features.update({'SENTIMENT_COMPOUND': polarity['compound']})

    return features


def head_of_token(parsed):
    """Retrieve the head of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"HEAD_OF_{0}".format(token): token.head.text for token in parsed}


def children_of_token(parsed):
    """Retrieve the children of the current token.
    :param parsed: The parsed document by spacy.
    """
    return {"children_of_{0}".format(token): "_".join([str(child) for child in token.children])
            for token in parsed}


def number_of_arcs(parsed):
    """Retrieve the number of right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"NO_RIGHT_ARCS_{0}".format(token): token.n_rights,
                "NO_LEFT_ARCS_{0}".format(token): token.n_lefts,
                "NO_TOTAL_ARCS_{0}".format(token): int(token.n_rights) + int(token.n_lefts)}
        features.update(arcs)
    return features


def arcs(parsed):
    """Retrieve the right and left arcs.
    :param parsed: The parsed document by spacy.
    """
    features = {}
    for token in parsed:
        arcs = {"RIGHT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.rights]),
                "LEFT_ARCS_{0}".format(token): "_".join([arc.text for arc in token.lefts])}
        features.update(arcs)
    return features


def get_brown_clusters(doc: List[str], cluster: Dict[str, str], salt: str = '') -> List[str]:
    """Generate cluster for each word.
    :param doc: Document ebing procesed as a list.
    :param cluster: Cluster computed using clustering algorithm.
    :param salt: To add in front of the features.
    :return: Dictionary of clustered values."""
    if salt != '':
        salt = salt.upper() + '_'
    return Counter([salt + cluster.get(w, 'CLUSTER_UNK') for w in doc])


def liwc(doc: List[str], liwc_dict: Dict[str, str]) -> List[Dict[str, float]]:
    """Computes LIWC Categories.
    :param doc: Document to be considered.
    :return: dictionary of computed values.
    """
    liwc_list = []
    liwc_vals = []
    kleene_star = [k[:-1] for k in liwc_dict if k[-1] == '*']

    for w in doc:
        if w in liwc_dict:
            liwc_list.append(liwc_dict[w])
            liwc_vals.extend(liwc_dict[w])
        else:
            # This because re.findall is slow.
            candidates = [r for r in kleene_star if r in w]
            cand_len = len(candidates)
            if cand_len == 0:
                continue
            elif cand_len == 1:
                term = candidates[0]
            elif cand_len > 1:
                sorted_cands = sorted(candidates, key=len, reverse = True)
                if sorted_cands == candidates:
                    term = candidates[0]
                else:
                    term = sorted_cands[0]

                term = liwc_dict[term + '*']
                liwc_list.append(term)
                liwc_vals.extend(term)

    liwc_vals = Counter(['LIWC_' + item for item in liwc_vals])

    found = {k: liwc_vals[k] / len(doc) for k in liwc_vals}
    return found
