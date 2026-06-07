# DepressionEmo
"DepressionEmo: A novel dataset for multilabel classification of depression emotions"
https://www.sciencedirect.com/science/article/abs/pii/S0165032724012278 

https://arxiv.org/pdf/2401.04655.pdf

# Dataset
## Subsets
The dataset is divided into 3 subsets:
* Training set
* Validation set
* Test set

## An example data
Each example contains "id", "title", "post", "text", "upvotes", "date", "emotions", and "label_id". We use "text" (concatenate from "title" and "post") for the depression detection.

There are 8 depression emotions:
```
emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
```

# Training
This is a multilabel classification problem, and we use only 1 model to detect all emotions at once.

## SVM, Light GBM, and XGBoost
All these methods use TfidfVectorizer and have no preprocessing steps. To train the models by these methods use:

```
python svm.py
python xgb.py
python light_gbm.py
```

## BERT
To train the model:
```
python bert.py  --mode "train" --model_name "bert-base-cased" --epochs 25 --batch_size 8 --max_length 256
```
