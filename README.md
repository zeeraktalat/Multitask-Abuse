# Multitask Learning for Hate Speech Detection

In this project, we seek to improve classification performance for abusive language detection by leveraging multitask learning on a set of tasks that appear related to abuse detection

## TODO

- Modeling

  - Code MTL model

    - Multilabel prediction

  - Create model which trains on a randomly assigned task but evaluates on hate speech

- Datasets

  - Sarcasm (https://arxiv.org/pdf/1704.05579.pdf)

  - Offensive Language: Davidson et al.

  - Hate speech (Waseem & Hovy, Waseem)

  - Toxicity (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

  - Sentiment (Semeval 2017 Task 4: http://alt.qcri.org/semeval2017/task4/index.php?id=results)

  - Moral Foundations Prediction (MFTC: https://psyarxiv.com/w4f72)

  - Twitter datasets?

  - Demographic dataset: Preotiuc and Ungar.
    
    Race key:
    
    - 1: African-American

    - 2: Latinx/Hispanic

    - 3: Asian

    - 4: White

    - 5: Multiracial

    - NULL: Didn't answer/Other race (usually Native American)

- Models

  - LSTM
