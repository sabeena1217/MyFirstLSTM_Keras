#one hot encode - to encode your documents with integers

from keras.preprocessing.text import one_hot



docs = ['Sabeena Gayesha','Kumarawadu','Super idee','Perfekt erledigt','exzellent','naja','Schwache arbeit.','Nicht gut','Miese arbeit.','Hatte es besser machen konnen.']

vocab_size = 50

encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)