import pandas as pd
import os
from eval_tsd import evaluate
from DataProcess import flair_pred2sample_pred


class BagOfWords:
    '''
    Bag of words method.
    '''
    def __init__(self, threshold=0.63, data_path='../data/'):
        self.overall_count = {}
        self.toxic_count = {}
        self.threshold = threshold
        self.data_path = data_path

    def _input_date(self, token, tag):
        if token in self.overall_count.keys():
            self.overall_count[token] += 1
        else:
            self.overall_count[token] = 1
        if tag == 'B' or tag == 'I':
            if token in self.toxic_count.keys():
                self.toxic_count[token] += 1
            else:
                self.toxic_count[token] = 1

    def train(self, filepath):
        # file should be in BIO format.
        filepath = self.data_path+filepath
        with open(filepath,'r',encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                token, tag = line.split()
                self._input_date(token, tag)

    def _predict_token(self, token):
        if token not in self.toxic_count.keys():
            return False
        score = self.toxic_count[token] / self.overall_count[token]
        if score > self.threshold:
            return True
        else:
            return False

    def predict(self, test_filepath, output_filepath):
        test_filepath = self.data_path+test_filepath
        output_filepath = self.data_path+output_filepath
        with open(test_filepath, 'r', encoding='utf-8') as test_f:
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                for line in test_f:
                    line = line.strip()
                    if len(line) == 0:
                        out_f.write('\n')
                        pre_tag = ''
                    else:
                        token, tag = line.split('\t')
                        if self._predict_token(token):
                            tag = 'B' if pre_tag not in ('B', 'I') else 'I'
                        else:
                            tag = 'O'
                        out_f.write(f"{token}\t{tag}" + "\n")
                        pre_tag = tag


if __name__ == '__main__':
    BOW = BagOfWords(threshold=0.36)
    BOW.train('fold_1234.txt')
    BOW.predict('fold_5.txt', 'bag_of_words_result/fold_5_predicted.txt')
    flair_pred2sample_pred(BOW.data_path+'bag_of_words_result/fold_5_predicted.txt', BOW.data_path+'bag_of_words_result/fold_5_predicted_recovered.txt')
    evaluate(BOW.data_path+'bag_of_words_result/fold_5_predicted_recovered.txt', BOW.data_path+'fold_5_modify.csv', BOW.data_path+'bag_of_words_result')









