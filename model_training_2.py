import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AutoConfig
import os
import os
import csv
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split

class FncDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='albert-base-v2'):
        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'text_a'])
        sent2 = str(self.data.loc[index, 'text_b'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',       # Pad to max_length
                                      truncation=True,            # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')        # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = int(self.data.loc[index, 'labels'])
            return {"input_ids": token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids, "label": label  }
        else:
            return {"input_ids": token_ids, 'attention_mask': attn_masks, 'token_type_ids': token_type_ids, "label": label  }
            # return token_ids, attn_masks, token_type_ids

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

    # Reset index
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    # Define bert model
    bert_model = 'albert-base-v2'
    # Create datasets
    train_dataset = FncDataset(train_df, 512, bert_model=bert_model)
    val_dataset = FncDataset(val_df, 512, bert_model=bert_model)
    test_dataset = FncDataset(test_df, 512, bert_model=bert_model)
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
    )
    label_to_id = {'agree': 0, 'disagree':1, 'discuss':2, 'unrelated':3}
    id_to_label = {v:k for (k,v) in label_to_id.items()}

    config = AutoConfig.from_pretrained(bert_model, num_labels=4)
    model = BertForSequenceClassification.from_pretrained(bert_model, config=config)
    model.config.num_labels = 4
    model.config.id2label = id_to_label
    model.config.label2id = label_to_id

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )
    trainer.train()

