import torch
import os
import argparse
import flair
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger

from DataProcess import flair_pred2sample_pred
from eval_tsd import evaluate


parser = argparse.ArgumentParser(description='make predictions')
parser.add_argument('--input', '-i',
                        help='Name of the input folder containing train, dev and test files')
parser.add_argument('--output', '-o',
                        help='Name of the output folder')
parser.add_argument('--gpu', '-g',
                        help='Use gpu/cpu, put "cuda" if gpu and "cpu" if cpu')
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--checkpoint')
parser.add_argument('--predict_file')
parser.add_argument("--sample_pred_file",
                        required=True,
                        help="save path of the txt file that is standard prediction.")

args = parser.parse_args()
flair.device = torch.device(args.gpu)

# Initialize Data
columns = {0:'text'}

data_folder = args.input

corpus: Corpus = ColumnCorpus(data_folder, 
                              columns,
                              train_file=args.train_file,
                              dev_file=args.dev_file,
                              test_file=args.test_file)
    
# what tag do we want to predict?
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# Load the model
model = SequenceTagger.load(os.path.join(args.output, args.checkpoint))

# Write predictions in file
with open(os.path.join(args.output, args.predict_file), 'w',encoding="utf-8") as f:
    for sentence in corpus.dev:
        model.predict(sentence)
        for token in sentence:
            f.write(f"{token.text}\t{token.get_tag('ner').value}" + "\n")   
        f.write("\n")
        
# compute the F1 score
flair_pred2sample_pred(os.path.join(args.input, args.test_file.replace('.txt', '_modify.csv')), os.path.join(args.output, args.predict_file), os.path.join(args.output, args.sample_pred_file))
evaluate(os.path.join(args.output, args.sample_pred_file), os.path.join(args.input, args.test_file.replace('.txt', '_modify.csv')))
