import torch
import argparse
import flair
from typing import List
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
from flair.embeddings import *
from torch.optim.lr_scheduler import OneCycleLR

parser = argparse.ArgumentParser(description='Train flair model')
parser.add_argument('--input', '-i',
                        help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output', '-o',
                        help='Name of the output folder')
parser.add_argument('--gpu', '-g',
                        help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file', default=None)
parser.add_argument('--transformer')
parser.add_argument('--use_crf', action='store_true', help='if use crf, default False')
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--mini_batch_size', type=int, default=8)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--patience', type=int, default=2)


args = parser.parse_args()
input_folder=args.input
output_folder=args.output
gpu_type=args.gpu
flair.device = torch.device(gpu_type)


# Change this line if you have POS tags in your data, eg.- {0: 'text', 1:'pos', 2:'ner'}
columns = {0: 'text', 1:'ner'}

data_folder = input_folder

tag_type = 'ner'

corpus: Corpus = ColumnCorpus(data_folder, 
                                columns, 
                                train_file=args.train_file,
                                dev_file=args.dev_file,
                                test_file=args.test_file)

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

embedding_types: List[TokenEmbeddings] = [
     TransformerWordEmbeddings(args.transformer, fine_tune=True),
    #  CharacterEmbeddings()
 ]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


# initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=args.use_crf,
                                        use_rnn=False,
                                        reproject_embeddings=False,)

# initialize trainer with AdamW optimizer
trainer: ModelTrainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

# train the model
trainer.train(output_folder, 
              learning_rate=args.learning_rate,
              mini_batch_size=args.mini_batch_size,
              max_epochs=args.max_epochs,
              patience=args.patience,
              scheduler=OneCycleLR,
              embeddings_storage_mode='gpu',
              weight_decay=0.01,)
