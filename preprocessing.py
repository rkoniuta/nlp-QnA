import csv

import json
import nltk
import sys


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


def preprocess(fin: str, fout: str) -> None:
    '''
    @fin: string for file path of input SQuAD data
    @fout: string for csv of processed triplets
    
    @output: csv of Question | Positive Answer | Context | Big Context pairs
    '''
    with open(fin, 'r') as fi, open(fout, 'w') as fo:
        data = json.load(fi)['data']
        out =  csv.writer(fo)
        for topic in progress(data):
            full_context = ' '.join([p["context"] for p in topic["paragraphs"]]).replace("\n", '')
            for paragraph in topic["paragraphs"]:
                context = paragraph["context"].replace("\n", '')
                sentences = nltk.sent_tokenize(context)
                for qa in paragraph["qas"]:
                    if not qa["is_impossible"]:
                        question = qa["question"].replace("\n", '')
                        for answer in [qa["answers"][0]]: ###### remove [0] to revert
                            idx = answer["answer_start"]
                            for sentence in sentences:
                                idx = idx - len(sentence)
                                if idx < 0:
                                    answer_sentence = sentence.replace("\n", '')
                                    out.writerow([question, answer_sentence, context, full_context])
                                    break

if __name__ == "__main__":
    preprocess("data/raw/dev.json", "data/processed/dev.csv")
    preprocess("data/raw/train.json", "data/processed/train.csv")
