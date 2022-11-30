from lib2to3.pgen2 import token
from preprocess import *
import tensorflow as tf
import matplotlib.pyplot as plt
import transformers
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MAX_SEQUENCE_LENGTH = 300
# MODEL_NAME = 'bert-base-multilingual-cased'
MODEL_NAME = "bert-base-uncased"
EMBEDDING_SIZE = 768
LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 2
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
#     combine step for tokenization, 
#     WordPiece vector mapping, 
#     adding special tokens as well as 
#     truncating reviews longer than the max length 

def convert_example_to_feature(review):
  return tokenizer.encode_plus(review, 
                add_special_tokens = True,     # add [CLS], [SEP]
                max_length = 512,              # max length of the text that can go to BERT
                padding='max_length',
                truncation=True,
                return_attention_mask = True,  # add attention mask to not focus on pad tokens
              )

# map to the expected input to TFBertForSequenceClassification
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []
  for review, label in ds:
    bert_input = convert_example_to_feature(review)
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])

  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

def freeze(bert, n):
  for layer in bert.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer in enumerate(layer.encoder.layer):
              if idx in range(0, n):
                layer.trainable = False
    
    # CHECKING FROZEN LAYERS
  for layer in bert.layers[:]:
      if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
          for idx, layer in enumerate(layer.encoder.layer):
              print(layer, layer.trainable)
def graph(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.show()

if __name__ == '__main__':
    # Preprocessing
    X_train_text, y_train, X_test_text, y_test = get_data()
    start = time.time()

    ds_train = zip(X_train_text, y_train)
    ds_test = zip(X_test_text, y_test)
    ds_train_encoded = encode_examples(ds_train).shuffle(len(X_train_text)).batch(BATCH_SIZE)
    ds_test_encoded = encode_examples(ds_test).batch(BATCH_SIZE)

    # Freeze layers in BERT
    n = 8
    freeze(bert, n)

    # Initializing input, attention mask for BERT
    print(bert.summary())
    # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
    learning_rate = 2e-5
    # multiple epochs might be better as long as we will not overfit the model
    number_of_epochs = 2

    # choosing Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    bert.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=metric)

    history = bert.fit(ds_train_encoded,
                         batch_size=BATCH_SIZE,
                         epochs=number_of_epochs,
                         validation_data=ds_test_encoded)

    end = time.time()
    print("Time: ", end - start)

    graph(history)
