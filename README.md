Enviroment:
sentence-transformers==2.1.0
torch==1.9.0
nltk==3.4.5
scipy==1.5.2


Data processing:

approximate time: 1-2 minutes

`python preprocessing.py`

output files given in `data/processed/train.csv` and `data/processed/dev.csv`


Baseline:

approximate time: 1 minute

`python baseline.py`


Training:

approximate time: 3 days on G4 AWS spot instance (4 tesla t4 gpus)

`python train.py --dev_data data/processed/dev.csv --train_data data/processed/train.csv --save model`


Evaluation:

approximate time: 1.5-2.5 hours

Best performing model (DistilBERT trained):
`python evaluate.py --data data/processed/dev.csv --model ./st_retrained`


DistilBERT mean pooling:
`python evaluate.py --data data/processed/dev.csv --model distilbert-mean`


DistilBERT max pooling:
`python evaluate.py --data data/processed/dev.csv --model distilbert-max`


BERT:
`python evaluate.py --data data/processed/dev.csv --model bert`


roBERTa:
`python evaluate.py --data data/processed/dev.csv --model roberta`


Interactive system:

approximate time: 3 minutes

`python interactive.py`

example:

url: https://en.wikipedia.org/wiki/University_of_Notre_Dame

query: What is the most recognizable landmark on campus?

answers: see `sample_interactive.png`, `sample_interactive2.png`
