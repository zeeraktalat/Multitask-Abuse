import os
from collections import Counter


def majority_voted(annotations):
    cleaned = []
    candidates = []
    for a in annotations:
        candidates.extend([cand for cand in a.split(',') if cand != ''])
    cleaned.extend(candidates)
    c = Counter(cleaned)
    return c.most_common(1)[0][0]


def clean_mftc():
    fin = open('~/Desktop/Datasets/MFTC_V4_text.json', 'r', encoding = 'utf-8')
    fout = open('MFTC_V4_text_parsed.tsv', 'w', encoding = 'utf-8')
    tweet_data = []
    annotations = []

    fout.write("\t".join(['tweet_id', 'tweet_text', 'annotator_1', 'annotator_2', 'annotator_3',
                          'annotator_4', 'annotator_5', 'annotator_6', 'annotator_7', 'annotator_8',
                          'annotation_1', 'annotation_2', 'annotation_3', 'annotation_4', 'annotation_5',
                          'annotation_6', 'annotation_7', 'annotation_8', 'majority_label', 'corpus']) + '\n')

    for line in fin:
        if ':' in line:
            line = line.replace('\n', '').replace('"', '').replace('\\n', '')
            if 'Corpus' in line:
                corpus = line.split(':')[1].strip().strip(',')
                continue

            if 'tweet_id' in line:
                value = line.split(':')[1].strip().strip(',')
                if tweet_data == [] and annotations == []:
                    tweet_data.append(value)
                elif tweet_data != [] and annotations != []:  # A new data object has been encountered.

                    if len(annotations) < 8:  # Ensure that there are the same number of annotators for each line
                        while len(annotations) <= 7:
                            annotations.append('')
                            annotators.append('')

                    # Identify Majority voted label
                    majority = majority_voted(annotations)
                    tweet_data.extend(annotators)
                    tweet_data.extend(annotations)  # Add annotations to output line
                    tweet_data.append(majority)
                    tweet_data.append(corpus)  # Add corpus information
                    fout.write("\t".join(tweet_data) + '\n')  # Write the data line

                    # Prepare for a new tweet
                    tweet_data = [value]
                    annotations = []
                    continue

            if 'tweet_text' in line:
                value = ":".join(line.split(':')[1:]).strip().strip('\n')[:-1]
                tweet_data.append(value)
                continue

            if 'annotations' in line:
                annotations = []
                annotators = []
                while ']' not in line:
                    line = next(fin)
                    if 'annotator' in line:
                        annotators.append(line.strip().split(':')[1].strip().replace('"', '').replace(',', ''))
                    if 'annotation' in line:
                        annotations.append(line.strip().split(':')[1].strip().replace('\n', '').replace('"', ''))
                continue


def clean_sentiment():
    indir = '/Users/zeerakw/Desktop/Datasets/semeval2017_sentiment/DOWNLOAD/Subtask_A/'
    of_train = '/Users/zeerakw/Documents/PhD/projects/Multitask-abuse/data/semeval_sentiment_train.tsv'
    of_test = '/Users/zeerakw/Documents/PhD/projects/Multitask-abuse/data/semeval_sentiment_test.tsv'

    with open(of_train, 'w', encoding = 'utf-8') as otrain, open(of_test, 'w', encoding = 'utf-8') as otest:
        files = os.listdir(indir)
        otrain.write("\t".join(['tweet_id', 'sentiment', 'tweet_text']) + '\n')
        otest.write("\t".join(['tweet_id', 'sentiment', 'tweet_text']) + '\n')
        for fh in files:
            if 'sms' in fh or 'live' in fh:
                continue
            print("Processing file: {0}".format(fh))
            inf = open(os.path.join(indir, fh), 'r', encoding = 'latin1').read()
            if 'train' in fh or 'dev' in fh:
                otrain.write(inf + '\n')
            if 'test' in fh and 'dev' not in fh:
                otest.write(inf + '\n')


if __name__ == "__main__":
    # clean_mftc()
    clean_sentiment()
