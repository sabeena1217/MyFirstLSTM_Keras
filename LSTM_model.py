# https://www.liip.ch/en/blog/sentiment-detection-with-keras-word-embeddings-and-lstm-deep-learning-networks
import keras
from keras.datasets import imdb
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing import sequence
from keras.models import Sequential
from numpy import array

top_words = 5000

# Step 1: Get the data
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# All words have been mapped to integers and the integers represent the words sorted by their frequency
# The integer 1 is reserved reserved for the start marker, the integer 2 for an unknown word and 0 for padding

# first_review_train = X_train[0]

# Step 2: Preprocess the data
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Step 3: Build the model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Step 4: Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=64)
# tell it which data it can use for validation.



# Step 5: Test the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(model.metrics_names)

# Step 6: Predict something

word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+1) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
for review in [good,bad]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length)
    print("%s. Sentiment: %s" % (review, model.predict(array([tmp_padded][0]))[0][0]))


# word_to_id = {k:v for k,v in word_to_id.items()}
# Accuracy: 85.10%
# i really liked the movie and had fun. Sentiment: 0.45147327
# this movie was terrible and bad. Sentiment: 0.8456366


# word_to_id = {k:(v+1) for k,v in word_to_id.items()}
# Accuracy: 85.26%
# ['loss', 'acc']
# i really liked the movie and had fun. Sentiment: 0.7944866
# this movie was terrible and bad. Sentiment: 0.7857737