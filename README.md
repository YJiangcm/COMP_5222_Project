# COMP_5222_Project
This is the repository of COMP 5222 group project, our group number is 16.

## **How to run:**

### 0. Download all files in this repository.

### 1. Prepare the train and dev data:  
Open your terminal and run _DataProcess.py_ in the "code" file. (**you need to modify data_path according to your situation**)
```
! python DataProcess.py --data_path C:/Users/31906/Desktop/5222_project/data
```

### 2. Train the model and make prediction:  
Open _Run.ipynb_ in the "code" file. You can directly run the code in **Google Colab**.
Here you can change the model structure flexibly.   
(refer to the tutorial of [flair](https://github.com/flairNLP/flair)).

### 3. Compute F1 score:
Open your terminal and run _eval_tsd.py_ in the "code" file. (**you need to modify data_path according to your situation**)
```
! python eval_tsd.py --flair_pred_file C:/Users/31906/Desktop/5222_project/data/bert-bio-fold_5.txt \
                     --sample_pred_file C:/Users/31906/Desktop/5222_project/data/predict_fold_5.txt \
                     --test_file C:/Users/31906/Desktop/5222_project/data/fold_5_modify.csv
```
Then you can get a result like this:  
_F1 score:  0.6704417291328144_
