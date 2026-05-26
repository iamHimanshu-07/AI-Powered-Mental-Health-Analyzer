# DepressionEmo
"DepressionEmo: A novel dataset for multilabel classification of depression emotions"
https://www.sciencedirect.com/science/article/abs/pii/S0165032724012278 

https://arxiv.org/pdf/2401.04655.pdf

If you use our dataset in your research paper, please make sure to cite the following paper:

Rahman, A. B. S., Ta, H. T., Najjar, L., Azadmanesh, A., & Gönül, A. S. (2024). DepressionEmo: A novel dataset for multilabel classification of depression emotions. Journal of Affective Disorders

Use of the DepressionEmo dataset or a part of your whole dataset for developing any web application, mobile app, or commercial tool is strictly prohibited without prior written permission.
Please contact us at abubakarsiddiqurra@unomaha.edu before initiating any such use.

© 2024 Abu Bakar Siddiqur Rahman. All Rights Reserved.


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
