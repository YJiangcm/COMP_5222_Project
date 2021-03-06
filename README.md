# COMP_5222_Project
This is the repository of COMP 5222 group project, our group number is 13.

## **How to run:**

### 0. Download all files in this repository.

### 1. Prepare the train and dev data:  
Open your terminal, navigate to the "code" folder and run _DataProcess.py_. 
```
! python DataProcess.py --data_path ../data
```

### 2. Train the model (need a >=10 GB GPU server):  
Open your terminal and run _flair_train.py_  in the "code" folder.   
**Here you can change the model structure flexibly.** (refer to the tutorial of [flair](https://github.com/flairNLP/flair)).
```
! python flair_train.py --input ../data \
              --output ../output \
              --gpu 'cuda' \
              --train_file 'fold_1234.txt' \
              --dev_file 'fold_5_dev.txt' \
              --test_file 'fold_5_test.txt' \
              --transformer 'bert-base-uncased' \
              --learning_rate 2e-5 \
              --mini_batch_size 8 \
              --max_epochs 20 \
              --patience 2
```

### 3. Make prediction and compute F1 score:
Open your terminal and run _predict.py_ in the "code" folder. (**you need to modify data_path according to your situation**)  
---------predict_file: the path of file that your model predicts.  
---------sample_pred_file: the path where you want to save the standard prediction.
```
! python predict.py --input ../data \
              --output ../output \
              --gpu 'cuda' \
              --train_file 'fold_1234.txt' \
              --dev_file 'fold_5_test.txt' \
              --test_file 'fold_5_test.txt' \
              --checkpoint 'best-model.pt' \
              --predict_file 'bert_fold_5_test.txt' \
              --sample_pred_file 'fold_5_test_recover.txt'
```
Then you can get a result like this:  
_F1 score:  0.6704417291328144_
