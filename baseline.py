import csv
import random

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


stopwords = set(stopwords.words('english'))

def progress(iterable):
    import os, sys
    if os.isatty(sys.stderr.fileno()):
        try:
            import tqdm
            return tqdm.tqdm(iterable)
        except ImportError:
            return iterable
    else:
        return iterable

def evaluate_random(fin: str) -> float:
    '''
    @fin: path for csv of processed triplets
    
    @output: MRR score for the dev dataset using a random metric
    '''
    with open(fin, "r") as f:
        data = list(csv.reader(f))

    context_mrr = 0
    avg_sentence = 0

    for _, _, context, full_context in progress(data):
        context_l = len(nltk.sent_tokenize(context))
        context_rand = random.randint(1, context_l+1)
        context_mrr += 1 / context_rand

    return (
        context_mrr / len(data),
    )


def evaluate_intersection(fin: str) -> float:
    '''
    @fin: path for csv of processed triplets
    
    @output: MRR score for the dev dataset using intersection metric
    '''
    with open(fin, "r") as f:
        data = list(csv.reader(f))
    
    context_mrr = 0

    for query, answer, context, full_context in progress(data):
        context_mrr += intersection_rank(query, answer, context)

    return (
        context_mrr / len(data),
    )


def intersection_rank(query: str, answer: str, paragraph: str) -> float:
    query_set = {
        word for word in nltk.word_tokenize(query)
        if word not in stopwords
    }

    paragraph = [ s for s in nltk.sent_tokenize(paragraph) ]

    scores = []
    for sentence in paragraph:
        sentence_set = {
            word for word in nltk.word_tokenize(sentence)
            if word not in stopwords
        }

        intersection = len(query_set.intersection(sentence_set))
        scores.append((intersection, sentence))

    sorted_scores = sorted(scores, key=lambda x: x[0], reverse=True)
    for i, (score, sent) in enumerate(sorted_scores):
        if answer.strip() in sent.strip():
            return 1 / (i + 1)


if __name__ == "__main__":
    context_random = evaluate_random("data/processed/dev.csv")
    print(f"RANDOM MRR score [ context ]: {context_random}")

    context_keyword = evaluate_intersection("data/processed/dev.csv")
    print(f"KEYWORD MRR score [ context ]: {context_keyword}")
