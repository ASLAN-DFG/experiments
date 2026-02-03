from nltk.tokenize import word_tokenize
from nltk import pos_tag


def keep_content_words_nltk(input_list):
    """
    Filters a list of strings using NLTK to keep nouns, adjectives, and verbs.
    """
    filtered_list = []

    # NLTK Tag prefixes:
    # N = Noun, V = Verb, J = Adjective
    allowed_prefixes = ('N', 'V', 'J')

    for text in input_list:
        # 1. Tokenize the string into individual words
        tokens = word_tokenize(text)

        # 2. Assign Part-of-Speech tags
        tagged_words = pos_tag(tokens)

        # 3. Filter based on the first letter of the tag
        content_words = [
            word for word, tag in tagged_words
            if tag.startswith(allowed_prefixes)
        ]

        filtered_list.append(" ".join(content_words))

    return filtered_list