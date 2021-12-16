from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import evaluation
import nltk

import csv

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, help='data')
    parser.add_argument('--dev_data', type=str, help='data')
    parser.add_argument('--save', type=str, help='path to save model')
    args = parser.parse_args()

    model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')

    with open(args.train_data, "r") as f:
        data = list(csv.reader(f))
        train_examples = []
        for query, answer, context, full_context in progress(data):
            for neg in nltk.sent_tokenize(context):
                if answer.strip() not in neg.strip():
                    train_examples.append(InputExample(texts=[query, answer, neg]))
        print(f"Finished parsing train triplets: total train examples, {len(train_examples)}")
    
    with open(args.dev_data, "r") as f:
        data = list(csv.reader(f))
        anchor_dev_examples = []
        pos_dev_examples = []
        neg_dev_examples = []

        for query, answer, context, full_context in progress(data):
            for neg in nltk.sent_tokenize(context):
                if answer.strip() not in neg.strip():
                    anchor_dev_examples.append(query)
                    pos_dev_examples.append(answer)
                    neg_dev_examples.append(neg)

        print(f"Finished parsing dev triplets: total dev examples, {len(anchor_dev_examples)}")
        
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=model)

    evaluator = evaluation.TripletEvaluator(anchor_dev_examples, pos_dev_examples, neg_dev_examples)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=100,
        output_path=args.save,
        evaluator=evaluator,
        evaluation_steps=1000,
    )
