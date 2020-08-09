# Multitask Learning for Hate Speech Detection

In this project, we seek to improve classification performance for abusive language detection by leveraging multitask learning on a set of tasks that appear related to abuse detection

## TODO

- Modeling

  - Code MTL model

    - Multilabel prediction

  - Try setting aux task weights based on how hard tasks are on the fly

  - Try with BPE (as en encoder)

  - Try with LIWC for every task

- Datasets

  - Sarcasm (https://nlds.soe.ucsc.edu/sarcasm2)

  - Factual or Feeling based argument (https://nlds.soe.ucsc.edu/factfeel)

  - Offensive Language: Davidson et al.

  - Hate speech (Waseem & Hovy, Waseem)

  - Toxicity (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

  - Sentiment (Hovy Sentiment gender)

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
