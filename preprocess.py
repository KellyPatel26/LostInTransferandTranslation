import pandas as pd
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
import re
import string 
import tensorflow as tf
import shutil
from transformers import InputExample, InputFeatures
from transformers import BertTokenizer, TFBertModel

DATA_COLUMN = 'Tweet'
LABEL_COLUMN = 'Label'

def balance(data):
    '''
    This function balances the data so that around 50% are labeled 1 and 50% are labeled 0.
    We will use undersampling.

    inputs:
        data - a dataframe of a csv with an text column and a target column
    output:
        a balanced dataset
    '''
    data_0_class = data[data['Label'] == 0]
    data_1_class = data[data['Label'] == 1]
    # print("************", len(data_0_class), len(data_1_class))
    data_0_class_undersampled = data_0_class.sample(data_1_class.shape[0], replace = True)
    data = pd.concat([data_0_class_undersampled, data_1_class], axis = 0)
    # print("************", len(data[data['Label'] == 0]), len(data[data['Label'] == 1]))
    # print("YOYOYOYOYOYOYOYO",data[data['Label'] == 0].head(10))
    return data

def remove_URLS(text):
    '''
    Removes URLS from text
    '''
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ', text)


# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    '''
    Removes emojis from text. Used reference above.
    Example:
        text = "Sad days 😔😔"
        remove_emoji("Sad days 😔😔") = Sad days
    '''
    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags = re.UNICODE)
    return emojis.sub(r' ', text)

def remove_punct(text):
    '''
    Removes punctuation from text including hashtags
    Example:
        text = "#Newswatch 2 vehicles"
        output = "Newswatch 2 vehicles"
    '''
    table = str.maketrans(' ',' ', string.punctuation)
    return text.translate(table)


def make_csv(tweets_file, labels_file):
    with open(tweets_file, encoding="UTF-8") as f:
        contents = f.read()

    with open(labels_file) as f:
        labels = f.readlines()

    contents = contents.split("STOPSTOPSTOP\n")
    contents = [c for c in contents if c != "" and c!="\n"]
    labels = [label.strip() for label in labels]
    # Construct a dataframe
    data = []
    positive = 0
    negative = 0
    for i in range(len(contents)):
        if labels[i] != "Neutral":
            if labels[i] == "Positive":
                # print("HERE!")
                positive += 1
                label = 1
            else:
                negative += 1
                label = 0
            tmp = [contents[i], label]
            if tmp not in data:
                data.append(tmp)
                
    # print(len(data), positive, negative)

    df = pd.DataFrame(data, columns=["Tweet", "Label"])

    # print(df.head(15))
    df.to_csv("data/processed_slovenian_tweets.csv", index=False)


def get_data():

    df = pd.read_csv("data/processed_slovenian_tweets.csv", encoding="ISO-8859-1")
    # we need to even this out
    df = balance(df)
    # remove URLS
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_URLS(x))
    # remove emojis
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_emoji(x))
    # remove punctuation
    df['Tweet'] = df['Tweet'].apply(lambda x: remove_punct(x))
    # make everything lowercase
    df['Tweet'] = df['Tweet'].apply(lambda x: str.lower(x))

    # Remove multiple spaces
    df['Tweet']= df['Tweet'].str.replace('   ', ' ')
    df['Tweet']= df['Tweet'].str.replace('     ', ' ')
    df['Tweet']= df['Tweet'].str.replace('\xa0 \xa0 \xa0', ' ')
    df['Tweet']= df['Tweet'].str.replace('  ', ' ')
    df['Tweet']= df['Tweet'].str.replace('—', ' ')
    df['Tweet']= df['Tweet'].str.replace('-', ' ')

    # print(df.head(100))
    # print(len(df))
    # split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df['Tweet'],df['Label'], stratify=df['Label'])
    # return df
    return X_train[0:15000], y_train[0:15000], X_test[0:5000], y_test[0:5000]

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
    InputExample(guid=None,
            text_a = "Hello, world",
            text_b = None,
            label = 1)
    train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                            text_a = x[DATA_COLUMN], 
                                                            text_b = None,
                                                            label = x[LABEL_COLUMN]), axis = 1)

    validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                            text_a = x[DATA_COLUMN], 
                                                            text_b = None,
                                                            label = x[LABEL_COLUMN]), axis = 1)
    
    return train_InputExamples, validation_InputExamples

  
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=300):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )





if __name__ == "__main__":
    # make_csv("polish_tweets.txt", "polish_labels.txt")
    # make_csv("croatian_tweets.txt", "croatian_labels.txt")
    # make_csv("slovenian_tweets.txt", "slovenian_labels.txt")
    # make_csv("english_tweets.txt", "english_labels.txt")



    # x_train, y_train, x_test, y_test = get_data()
    # print(len(x_train), len(y_train), len(x_test), len(y_test))
    # print(x_train.values.tolist())


    # MODEL_NAME = 'bert-base-multilingual-cased'
    # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # model = TFBertModel.from_pretrained(MODEL_NAME)

    
    # data = get_data()
    # train = data[0:30000]
    # test = data[30000:]
    # train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
    #                                                                        test, 
    #                                                                        'Tweet', 
    #                                                                        'Label')

    # print(train_InputExamples)
    # train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    # validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    
    # print(train_data)
    print("Preprocessing Complete")
