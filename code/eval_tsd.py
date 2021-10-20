import pandas as pd
import numpy as np 
import sys
import argparse

def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    if len(predictions) == 0:
        return 0.
    predictions_set = set(predictions)
    gold_set = set(gold)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)



def evaluate(pred_file, test_file):
    '''
    pred_file: path to the prediction file
    test_file: path to the test csv file
    '''
    test_df = pd.read_csv(test_file)

    gold_spans = test_df.spans.apply(eval).to_list()

    pred_spans = [eval(line.strip()) for line in open(pred_file).readlines()]

    if(len(gold_spans) != len(pred_spans)):
        print('Error: the number of predictions does not match the number of test examples!')
        sys.exit(1)


    scores = []

    for pred, gold in zip(pred_spans, gold_spans):
        scores.append(f1(pred, gold))

    print('F1 score: ',  np.mean(scores))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prediction_file",
                        required=True,
                        help="path to the line-by-line file containing system predictions.")
    
    parser.add_argument("--test_file",
                        required=True,
                        help="path to the csv file with gold spans.")
    
    args = parser.parse_args()

    evaluate(args.prediction_file, args.test_file)


if __name__ == "__main__":
    main()