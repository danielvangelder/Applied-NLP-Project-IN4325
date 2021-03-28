import os
import csv
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel

def fnc(path_headlines, path_bodies):

    stance_map = {'agree': 0, 'disagree':1, 'discuss':2, 'unrelated':3}

    with open(path_bodies, encoding='utf_8') as fb:  # Body ID,articleBody
        body_dict = {}
        lines_b = csv.reader(fb)
        for i, line in enumerate(tqdm(list(lines_b), ncols=80, leave=False)):
            if i > 0:
                body_id = int(line[0].strip())
                body_dict[body_id] = line[1]

    with open(path_headlines, encoding='utf_8') as fh: # Headline,Body ID,Stance
        lines_h = csv.reader(fh)
        h = []
        b = []
        l = []
        for i, line in enumerate(tqdm(list(lines_h), ncols=80, leave=False)):
            if i > 0:
                body_id = int(line[1].strip())
                label = line[2].strip()
                if label in stance_map and body_id in body_dict:
                    h.append(line[0])
                    l.append(stance_map[line[2]])
                    b.append(body_dict[body_id])
    return h, b, l
if __name__ == '__main__':
    data_dir = 'data/fnc-1/'
    headlines, bodies, labels = fnc(
        os.path.join(data_dir, 'train_stances.csv'),
        os.path.join(data_dir, 'train_bodies.csv')
    )

    list_of_tuples = list(zip(headlines, bodies, labels))
    df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels'])
    train_df, val_df = train_test_split(df)
    labels_val = pd.Series(val_df['labels']).to_numpy()

    headlines, bodies, labels = fnc(
        os.path.join(data_dir, 'competition_test_stances.csv'),
        os.path.join(data_dir, 'competition_test_bodies.csv')
    )

    list_of_tuples = list(zip(headlines, bodies, labels))
    test_df = pd.DataFrame(list_of_tuples, columns=['text_a', 'text_b', 'labels'])
    labels_test = pd.Series(test_df['labels']).to_numpy()

    # display(train_df.sample(n=5))
    # display(test_df.sample(n=5))
    # print(set(train_df['labels'].values))

    # Original: 'bert', 'bert-base-uncased'
    # New: 'albert', 'albert-base-v2'
    model = ClassificationModel('albert', 'albert-base-v2', num_labels=4, use_cuda=False, args={
        'learning_rate':1e-5,
        'num_train_epochs': 5,
        'reprocess_input_data': True,
        'overwrite_output_dir': True,
        'process_count': 10,
        'train_batch_size': 4,
        'eval_batch_size': 4,
        'max_seq_length': 512,
        'fp16': True
    })

    model.train_model(train_df)

