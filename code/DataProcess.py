import pandas as pd
from ast import literal_eval
import re
import os


def string_split(string):
    # split string by punctuation
    if string == '':
        return ['']
    else:
        string = re.split("([!“\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n])", string)
        while "" in string:
            string.remove('')
        new_string = []
        for i in range(len(string)):
            new_string.append(string[i])
            if i != len(string)-1:
                new_string.append('@')
        return new_string


def text_split(text):
    # split a text
    text = re.sub(r'\n', '.', text)
    return sum([string_split(i) for i in text.split(" ")], [])


def create_new_span(spans):
    # convert spans into intervals
    new_span = []
    for i in range(len(spans)):
        if i == 0 or i == len(spans)-1:
            new_span.append(spans[i])
        else:
            if spans[i] != spans[i-1] + 1 or spans[i] != spans[i+1] - 1:
                new_span.append(spans[i])
    return new_span


def BIO_toxic(text):
    # convert toxic into bio format
    text = text_split(text)
    bio = [[text[0], 'B']]
    for i in range(1, len(text)):
         bio.append([text[i], 'I'])
    return bio


def BIO_nontoxic(text):
    # convert non-toxic into bio format
        if len(text) == 0:
            return []
        else:
            return [[i, 'O'] for i in text_split(text)]


def text2BIO(new_span, text):
    # convert a text to bio format according to spans
    bio = []
    for i in range(0, len(new_span), 2):
        toxic = text[new_span[i]: new_span[i+1]+1]
        toxic_bio = BIO_toxic(toxic)
        if i == 0:
            new_text = [text[: new_span[i]], text[new_span[i+1]+1:]]
        else:
            new_text = [text[new_span[i-1]+1: new_span[i]], text[new_span[i+1]+1:]]
            # print(new_text)
        
        new_text_0 = new_text[0]
        if len(new_text_0) > 0:
            if new_text_0[0] == ' ':
                new_text_0 = new_text_0[1:]
                begin_delete = True
            else:
                begin_delete = False
            if len(new_text_0) > 0 and new_text_0[-1] == ' ':
                new_text_0 = new_text_0[:-1]
                end_delete = True
            else:
                end_delete = False
            if i > 0 and len(new_text_0) > 0 and new_text_0[0] in "!“\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n" and not begin_delete:
                bio.extend([['@', 'O']])
            bio.extend(BIO_nontoxic(new_text_0))
            if len(new_text_0) > 0 and new_text_0[-1] in "!“\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n" and not end_delete:
                bio.extend([['@', 'O']])
                
        bio.extend(toxic_bio)
        
    if len(new_span) != 0:
        new_text_1 = new_text[1]
        if len(new_text_1) > 0 and new_text_1[0] == ' ':
            new_text_1 = new_text_1[1:]
            beg_delete = True
        else:
            beg_delete = False
        if len(new_text_1) > 0 and new_text_1[0] in "!“\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n" and not beg_delete:
            bio.extend([['@', 'O']])
        bio.extend(BIO_nontoxic(new_text_1))
    else:
        bio = BIO_nontoxic(text)
        
    convert_space_bio = []
    for i in bio:
        if i[0] == '':
            if i[1] == 'B':
                convert_space_bio.extend([['@', 'B'], ['.', 'B']])
            elif i[1] == 'I':
                convert_space_bio.extend([['@', 'I'], ['.', 'I']])
            else:
                convert_space_bio.extend([['@', 'O'], ['.', 'O']])
        else:
            convert_space_bio.append(i)
                
    return convert_space_bio
    
    
def BIO2idx(bios):
    # convert bio format to spans
    idx = []
    count = 0
    for i in range(len(bios)):
        if bios[i][0] != '@':
            
            if bios[i][1] == "B":
                idx.extend(list(range(count, count + len(bios[i][0]))))
            if bios[i][1] == "I":
                if bios[i-1][0]!='@':
                    idx.extend(list(range(count-1, count + len(bios[i][0]))))
                else:
                    idx.extend(list(range(count, count + len(bios[i][0]))))
        
            if i != len(bios)-1 and bios[i+1][0]!='@':
                count += (len(bios[i][0])+1)
            else:
                count += len(bios[i][0])
    return idx
            

def delete_error_data(csv_file_name, modified_csv_name):
    # some examples are wrongly labeled, so we want to delete these examples
    train = pd.read_csv(csv_file_name)
    
    # convert spans to intervals and delete incorrect spans
    train.spans = train.spans.apply(lambda x: literal_eval(x))
    train['new_span'] = train.spans.apply(lambda x: create_new_span(x))
    train['if_correct_span'] = train.new_span.apply(lambda x: 1 if len(x) % 2 == 0 else 0)
    train = train[train.if_correct_span == 1]
    
    # delete examples whose bio length is larger than 500 (because the max sequence length of transformers is 512)
    train['seq_length'] = train.apply(lambda x: len(text2BIO(x.new_span, x.text)), axis=1)
    train = train[train.seq_length <= 500].reset_index(drop=True)

    # detect error examples
    error_idx = []
    for i in range(len(train)):
        text = train.text[i]
        new_span = train.new_span[i]
        x = text2BIO(new_span, text)
        if BIO2idx(x) != train.spans[i]:
            error_idx.append(i)
            
    print("Number of examples in {:}: {:}".format(csv_file_name, len(train)))
    print("Detect number of error examples: ", len(error_idx))
    print("Nnumber of correct examples: ", len(train) - len(error_idx))
    
    train.drop(index = error_idx)[['spans', 'text']].to_csv(modified_csv_name, index=False)
    print("save modified file to ", modified_csv_name)
    
    
def generate_BIO_txt(csv_file_name, txt_file_name):
    # convert csv file to bio txt file, then we can use txt as the input of Flair package
    train = pd.read_csv(csv_file_name)
    train.spans = train.spans.apply(lambda x: literal_eval(x))
    train['new_span'] = train.spans.apply(lambda x: create_new_span(x))

    f = open(txt_file_name, 'w', encoding='utf-8')
    for i in range(len(train)):
        text = train.text[i]
        new_span = train.new_span[i]
        BIO = text2BIO(new_span, text)
        # print(BIO)
        for j in BIO:
            f.write(j[0] + '\t' + j[1] + '\n')
        f.write('\n')
    f.close()
    print("save txt file to ", txt_file_name)
    
    if len(txt2BIO(txt_file_name)) != len(train):
        raise Exception("The number of examples in txt file is not equal to the number of examples in csv file", txt_file_name)
        
    
def txt2BIO(txt_file_name):
    # read txt file and return BIO list
    f = open(txt_file_name, 'r', encoding='utf-8')
    bios = []
    bio = []
    while True:
        line = f.readline()
        if line and line != '\n':
            bio.append(line.strip('\n').split('\t'))
        elif line and line == '\n':
            bios.append(bio)
            bio = []
        else:
            break
    f.close()
    print("number of examples in txt: {:}\n".format(len(bios)))
    return bios
        
            
def merge_files(txts, output_file):
    writer = open(output_file, 'w', encoding='utf-8')
    for txt in txts:
        lines = open(txt, 'r', encoding='utf-8').readlines()
        for i, line in enumerate(lines):
            writer.write(line)
    writer.close()
    print("create merged file: ", output_file)


def flair_pred2sample_pred(flair_pred_file, sample_pred_file):
    bios = txt2BIO(flair_pred_file)
    f = open(sample_pred_file, 'w', encoding='utf-8')
    for bio in bios:
        f.write(str(BIO2idx(bio)) + '\n')
    f.close()
    print("create sample prediction file: ", sample_pred_file)




if __name__ == "__main__":
    """
    only need to modify the data path
    
    """
    
    data_path = "C:/Users/31906/Desktop/5222_project/data/"
    
    for i in range(1, 6):
        delete_error_data(os.path.join(data_path, "fold_"+str(i)+".csv"), os.path.join(data_path, "fold_"+str(i)+"_modify.csv"))
        generate_BIO_txt(os.path.join(data_path, "fold_"+str(i)+"_modify.csv"), os.path.join(data_path, "fold_"+str(i)+".txt"))
    
    for i in range(1, 6):
        # 5-fold cross validation
        lists = list(range(1, 6))
        lists.remove(i)
        txts = [os.path.join(data_path, "fold_"+str(j)+".txt") for j in lists]
        merge_files(txts, os.path.join(data_path, "fold_"+"".join(str(i) for i in lists)+".txt"))    
    

##############################################################################
# after get the flair prediction txt file
# sample_pred_file = "C:/Users/31906/Desktop/5222_project/data/predict_fold_5.txt"
# flair_pred2sample_pred(flair_pred_file, sample_pred_file)

"""
%run eval_tsd.py --prediction_file C:/Users/31906/Desktop/5222_project/data/predict_fold_5.txt \
                 --test_file C:/Users/31906/Desktop/5222_project/data/fold_5_modify.csv
F1 score:  0.6704417291328144
"""









    
    
    
    