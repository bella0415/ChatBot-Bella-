import sys
sys.path.append('.')

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from utils.Preprocess import Preprocess
from config.GlobalParams import MAX_SEQ_LEN
from keras.utils.data_utils import pad_sequences

# 데이터 읽어오기
data = pd.read_csv("models/intent/total_train_data.csv")

queries = data["query"].tolist()
intents = data["intent"].tolist()

p = Preprocess(word2index_dic="train_tools/dict/chatbot_dict.bin")

# 단어 시퀀스 생성
sequences = []
for sentence in queries:
    pos = p.pos(sentence)
    keywords = p.get_keywords(pos, without_tag=True)
    seq = p.get_wordidx_sequence(keywords)
    sequences.append(seq)

# 패딩
padded_seqs = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding="post")

print(padded_seqs.shape)
print(len(intents))

# 학습 : 검증 : 테스트 = 7 : 2 : 1

ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)


# 하이퍼 파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(p.word_index) + 1


# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN))
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
dropoub_emb = Dropout(rate=dropout_prob)(embedding_layer)

conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding="valid",
    activation=tf.nn.relu
)(dropoub_emb)
pool1 = GlobalMaxPool1D()(conv1)

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding="valid",
    activation=tf.nn.relu
)(dropoub_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding="valid",
    activation=tf.nn.relu
)(dropoub_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이후 합치기
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(5, name="logits")(dropout_hidden)
predictions = Dense(5, activation=tf.nn.softmax)(logits)

# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer=keras.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics="accuracy")

# 모델 학습
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"Accuracy : {accuracy*100}")
print(f"Loss : {loss}")

# 모델 저장
model.save("models/intent/intent_model.h5")