from lib2to3.pgen2 import token
from preprocess import *
import tensorflow as tf
import matplotlib.pyplot as plt
import transformers
from transformers import BertTokenizer, TFBertModel
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MAX_SEQUENCE_LENGTH = 300
MODEL_NAME = 'bert-base-multilingual-cased'
EMBEDDING_SIZE = 768
LEARNING_RATE = 0.01
BATCH_SIZE = 4
NUM_EPOCHS = 2

if __name__ == '__main__':
    # Preprocessing
    # X_train, y_train, X_test, y_test = get_data()
    # fix_gpu()
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = TFBertModel.from_pretrained(MODEL_NAME)

    data = get_data()
    train = data[0:30000]
    test = data[30000:]

    # print(train)
    # print(test)

    def untuple(ds):
        for x in ds:
            x = x[0]
        return ds



    train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           test, 
                                                                           'Tweet', 
                                                                           'Label')
    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)

    # train_data = train_data.map(lambda x:x[0])
    # validation_data = validation_data.map(lambda x:x[0])

    # print(train_InputExamples)
    train_data = train_data.batch(BATCH_SIZE)
    validation_data = validation_data.batch(BATCH_SIZE)
    print("*****")
    for elem in train_data.take(1).as_numpy_iterator():
        print(elem)
    # x = list(train_data.take(1).as_numpy_iterator())[0][0]
    # for idx in x:
    #     print(len(x[idx]))

    print(train_data)

    # # FREEZING LAYERS
    for layer in model.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer in enumerate(layer.encoder.layer):
              if idx in [0, 1]:
                layer.trainable = False
    
    # CHECKING FROZEN LAYERS
    for layer in model.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer in enumerate(layer.encoder.layer):
                print(layer, layer.trainable)

    classification_layers = tf.keras.Sequential(
    [
    tf.keras.layers.Dense(
        units=EMBEDDING_SIZE,
        activation='relu',
        name="dense_01",),
    tf.keras.layers.Dense(
        units=EMBEDDING_SIZE,
        activation='relu',
        name="dense_02",),
    tf.keras.layers.Dense(
        units=2,
        activation='softmax',
        name="softmax")
    ])

    model.add(classification_layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), 
              loss="categorical_crossentropy", 
               metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    model.fit(train_data, epochs=NUM_EPOCHS, validation_data=validation_data)

