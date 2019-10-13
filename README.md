# Multitask Learning for Hate Speech Detection

In this project, we seek to improve classification performance for abusive language detection by leveraging multitask learning on a set of tasks that appear related to abuse detection

## TODO

- Modeling

  - Code MTL model

    - Randomly choose task to work on.

    - Develop joint loss for training

  - Create model which trains on a randomly assigned task but evaluates on hate speech

  - Fix data reading

  - Create dev sets for everything

- Datasets

  - Sarcasm (https://arxiv.org/pdf/1704.05579.pdf)

  - Offensive Language: Davidson et al.

  - Toxicity (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

  - Sentiment (Semeval 2017 Task 4: http://alt.qcri.org/semeval2017/task4/index.php?id=results)

  - Rumour Detection (PHEME dataset: https://github.com/kochkinaelena/Multitask4Veracity OR RumourEval2017)

  - Moral Foundations Prediction (MFTC: https://psyarxiv.com/w4f72)

  - Twitter datasets?

- Models

  - LSTM
