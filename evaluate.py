import csv
import nltk

from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine

#model = SentenceTransformer("./GEORGETHISONE")
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#embedder = SentenceTransformer("./GEORGETHISONE")

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

def evaluate(fin: str) -> float:
    '''
    @fin: str, path for csv of processed triplets
    
    @output: float, MRR score for the dev dataset using intersection metric
    '''
    with open(fin, "r") as f:
        data = list(csv.reader(f))

    context_mrr = 0
    for query, answer, context, full_context in progress(data):
        context_mrr += rank(query, answer, context)

    return (
        context_mrr / len(data)
    )


def rank(query: str, answer: str, paragraph: str) -> float:
    '''
    @query: str, question to be answered
    @answer: str, sentence containing the answer
    @paragraph: str, group of sentences, one of which is the correct answer

    @output: reciprocal rank score for this query
    '''
    context = [query] + [ s.strip() for s in nltk.sent_tokenize(paragraph) ]
    scores = []
    embeddings = model.encode(context)
    query_embedding = embeddings[0]

    for embedding, sent in list(zip(embeddings, context))[1:]:
        scores.append(
            (cosine(query_embedding, embedding), sent)
        )

    sorted_scores = sorted(scores, key=lambda x: x[0])
    for i, (score, sent) in enumerate(sorted_scores):
        if answer.strip() in sent.strip():
            return 1 / (i + 1)



if __name__ == "__main__":
    # NOTE: running this is a bit lengthy. Expect ~1 hour on a Macbook Pro

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model to use')
    parser.add_argument('--data', type=str, help='data')
    args = parser.parse_args()

    if args.model == "distilbert-mean":
        model = SentenceTransformer("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
    elif args.model == "distilbert-max":
        model = SentenceTransformer("sentence-transformers/nli-distilbert-base-max-pooling")
    elif args.model == "bert":
        model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
    elif args.model == "roberta":
        model = SentenceTransformer("sentence-transformers/roberta-base-nli-mean-tokens")
    else:
        model = SentenceTransformer(args.model)

    print(f"MRR [{args.model}]: {evaluate(args.data)}")
