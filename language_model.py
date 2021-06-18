import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
import string

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
df = pd.read_table(path_to_file, names=["text"])
tokenizer = Tokenizer(filters = string.punctuation,
           lower=True,
           char_level=True,
           )

tokenizer.fit_on_texts(df["text"])

df["text_idx"] = df["text"].apply(lambda x: tokenizer.texts_to_sequences([x])[0])

df["input"] = df["text_idx"].apply(lambda x: x[:-1])
df["output"] = df["text_idx"].apply(lambda x: x[1:])

df_train, df_test = train_test_split(df, test_size=0.2)
vocab_size = max(tokenizer.index_word) + 1

df_train["input_ohe"] = df_train["input"].apply(lambda x:  tf.keras.utils.to_categorical(y=x,  num_classes=vocab_size))
df_train["input_ohe"] = df_train["input_ohe"].apply(np.array)
df_train["input_ohe"] = df_train["input_ohe"].apply(lambda x: pad_sequences([x], maxlen=47, padding="pre")[0])
df_test["input_ohe"] = df_test["input"].apply(lambda x:  tf.keras.utils.to_categorical(y=x, num_classes=vocab_size))
df_test["input_ohe"] = df_test["input_ohe"].apply(np.array)
df_test["input_ohe"] = df_test["input_ohe"].apply(lambda x: pad_sequences([x], maxlen=47,  padding="pre")[0])



df_train["output_ohe"] = df_train["output"].apply(lambda x:  tf.keras.utils.to_categorical(y=x, 
                                                                                           num_classes=vocab_size))
df_train["output_ohe"] = df_train["output_ohe"].apply(lambda x: pad_sequences([x], maxlen=47,  padding="pre")[0])

df_test["output_ohe"] = df_test["output"].apply(lambda x:  tf.keras.utils.to_categorical(y=x, num_classes=vocab_size))
df_test["output_ohe"] = df_test["output_ohe"].apply(lambda x: pad_sequences([x], maxlen=47,  padding="pre")[0])



train_x_ohe =  np.array(df_train["input_ohe"].values.tolist())
test_x_ohe =  np.array(df_test["input_ohe"].values.tolist())

train_y_ohe = np.array(df_train["output_ohe"].values.tolist())
test_y_ohe = np.array(df_test["output_ohe"].values.tolist())


inp = L.Input(shape = (47, 39))
lstm = L.LSTM(units=128, return_sequences=True, return_state=True)
hidden_states, h_t, c_t = lstm(inp)
lstm_2 = L.LSTM(units=128, return_sequences=True, return_state=True)
hidden_states, h_t, c_t = lstm_2(hidden_states)
out = L.TimeDistributed(L.Dense(units=vocab_size, activation="softmax"))(hidden_states)



model = Model(inputs = [inp], outputs=[out])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
hist = model.fit(train_x_ohe, train_y_ohe, 
                 validation_data = (test_x_ohe, test_y_ohe), 
                 batch_size=32, epochs=20)
                 
temperature = 0.5
start = "who ar"


for i in range(50):
    new_input_idx = np.array(tokenizer.texts_to_sequences([start]))
    new_input_ohe = tf.keras.utils.to_categorical(y=new_input_idx,  num_classes=vocab_size)
    new_input_padded = pad_sequences(new_input_ohe, maxlen=47, padding="pre")
    output = model.predict(new_input_padded)
    output_t = tf.math.log(output[0,-1:, :])/temperature
    sample_word_idx = tf.random.categorical(output_t, num_samples=1).numpy()[0][0]
    predicted_token = tokenizer.index_word[sample_word_idx]
    start = start + predicted_token
    print(i, start)

