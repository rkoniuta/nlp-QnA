import re
import nltk
import sys

from urllib.request import urlopen
from bs4 import BeautifulSoup

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


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


def load_models():
    print("Loading model...")
    model = SentenceTransformer("./st_retrained")
    print("Model loaded.")

    return model


def read_page(url):
    '''
    @url: str representing which url to fetch text from

    @out: list of sentences parsed from the HTML
    '''
    print("Reading url for text content...")
    if not url.startswith("https://"):
        print("try a url that starts with https:, i.e. 'https://en.wikipedia.org/wiki/University_of_Notre_Dame'")
        sys.exit()

    soup = BeautifulSoup(urlopen(url.strip()).read(), 'lxml')

    text = ''
    for paragraph in soup.find_all('p'):
        text += paragraph.text

    text = re.sub(r'\[.*?\]+', '', text)
    text = text.replace('\n', '')
    sents = [ s.strip() for s in nltk.sent_tokenize(text) ]

    print("Finished parsing text from url.")

    return sents


def encode_text(sents):
    '''
    @sents: list of strings of text in the corpus

    @output: zipped list of sentences and their embeddings
    '''
    print("Encoding text...")
    embeddings = []

    for sent in progress(sents):
        embeddings.append(model.encode(sent))

    data = list(zip(embeddings, sents))

    print("Text encoded.")

    return data


def query(q):
    print("Fetching answers...")
    query_embedding = model.encode(q.strip())

    results = sorted(embeddings, key=lambda x: cosine(x[0], query_embedding))[:5]

    print("TOP ANSWERS:")
    for i, (emb, sent) in enumerate(results):
        print(f"{i + 1}) [score: {cosine(emb, query_embedding)}]: {sent}")
    print()



if __name__ == "__main__":
    model = load_models()
    url = input("Enter a page URL: ")
    text = read_page(url)

    while len(text) <= 5:
        url = input("There's barely any text scraped from this page. Try another site: ")
        text = read_page(url)
    
    embeddings = encode_text(text)

    while True:
        q = input("Enter a query: ")
        query(q)
