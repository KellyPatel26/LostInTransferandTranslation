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
NUM_EPOCHS = 1

if __name__ == '__main__':
    # Preprocessing
    X_train, y_train, X_test, y_test = get_data()
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = TFBertModel.from_pretrained(MODEL_NAME)
    tester_data = X_test[0:10]
    tester_labels = y_test[0:10]
    tokenized_train_input = tokenizer(
        X_train.values.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf"
    )

    tokenized_test_input = tokenizer(
        X_test.values.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf"
    )

    tokenized_tester_input = tokenizer(
        tester_data.values.tolist(),
        truncation=True,
        padding='max_length',
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="tf"
    )

    for layer in bert.layers[:]:
          if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
              for idx, layer in enumerate(layer.encoder.layer):
                # if idx in [0, 1]:
                layer.trainable = False
      
      # CHECKING FROZEN LAYERS
    for layer in bert.layers[:]:
        if isinstance(layer, transformers.models.bert.modeling_tf_bert.TFBertMainLayer):
            for idx, layer in enumerate(layer.encoder.layer):
                print(layer, layer.trainable)

    train_input = dict(tokenized_train_input)
    train_label = tf.convert_to_tensor(y_train)
    test_input = dict(tokenized_test_input)
    test_label = tf.convert_to_tensor(y_test)

    tester_input = dict(tokenized_tester_input)
    tester_labels = tf.convert_to_tensor(tester_labels)
    
    # Initializing input, attention mask for BERT
    input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')
    
    # print(tf.shape(input_ids))
    # print(tf.shape(attention_mask))
    # print(train_input)
    # Getting the output from bert
    output = bert([input_ids, attention_mask]).last_hidden_state[:, 0, :]
    
    # Classification Layers
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

    # Feeding the output of BERT into classification layers 
    output = classification_layers(output)

    # Compiling the model and specifying loss, optimizer
    model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
      optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
      metrics=['accuracy'])

    # Training the model
    print("BEFORE")
    print(model(tester_input))
    print(tester_labels)
    tester_bert1 = bert(tester_input)
    print(tester_bert1)

    history = model.fit(train_input,train_label,batch_size=BATCH_SIZE, epochs = NUM_EPOCHS, verbose = 1, validation_split = 0.2)
    print("AFTER")

    print(model(tester_input))
    print(tester_labels)
    tester_bert2 = bert(tester_input)
    print("TESTER BERT")
    print(tester_bert2)
    print(tf.sum((tester_bert2[0] - tester_bert2[1])))
    print("TESTER+Class")
    print(classification_layers(tester_bert2))

    train_loss, train_acc = model.evaluate(train_input, train_label)
    # Testing the model
    test_loss, test_acc = model.evaluate(test_input,test_label)
    print('\Train accuracy: {}'.format(train_acc))
    print('\nTest accuracy: {}'.format(test_acc))
    print("Acc:", history.history['accuracy'])

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

