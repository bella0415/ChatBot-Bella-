# 챗봇에서 사용하는 사전 파일 생성
# import sys
# sys.path.append("/Users/dominick/Library/CloudStorage/Dropbox/stats/Project/chatbot/")
# sys.path.append("/User/dominick/miniconda3/envs/tf/python/lib/site-packages")

from tensorflow import keras
from keras import preprocessing

from utils.Preprocess import Preprocess
import pickle

# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, "r") as f:
        data = [line.split("\t") for line in f.read().splitlines()]
    return data

# 말뭉치 데이터 가져오기
corpus_data = read_corpus_data("corpus.txt")

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess()

dict = []

for c in corpus_data:
    pos = p.pos(c[1])
    for k in pos:
        dict.append(k[0])
        

# 사전에 사용될 word2index 생성
# 사전의 첫 번째 인덱스에는 OOV사용
tokenizer = preprocessing.text.Tokenizer(oov_token="OOV")
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index


# 사전 파일 생성
with open("chatbot_dict.bin", "wb") as f:
    try:
        pickle.dump(word_index, f)
    except Exception as e:
        print(e)