import sys
sys.path.append('.')

from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess

from models.intent.IntentModel import IntentModel
from models.ner.NerModel import NerModel
from utils.FindAnswer import FindAnswer

# 전처리 객체 생성
p = Preprocess(word2index_dic="train_tools/dict/chatbot_dict.bin")

# 질문/답변 학습 DB 연결 객체 생성
db = Database(
    host=DB_HOST, 
    port=DB_PORT, 
    user=DB_USER, 
    password=DB_PASSWORD, 
    db_name=DB_DATABASE
)
db.connect() # DB 연결

# 원문
query = "오전에 짜장면 10개 주문합니다"

# 의도 파악
intent = IntentModel(model_name="models/intent/intent_model.h5", preprocess=p)
predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# 개체명 인식 
ner = NerModel(model_name="models/ner/ner_model.h5", preprocess=p)
predicts = ner.predict(query)
ner_tags = ner.predict_tags(query)

print(f"질문 : {query}")
print("=" * 100)
print(f"의도 파악 : {intent_name}")
print(f"개체명 인식 : {predicts}")
print(f"답변 검색에 필요한 NER 태그 : {ner_tags}")
print("=" * 100)

try:
    f = FindAnswer(db)
    answer_text, answer_image = f.search(intent_name, ner_tags)
    answer = f.tag_to_word(predicts, answer_text)
except:
    answer = "죄송해요. 무슨 말인지 몰?루겠어요"

print(f"답변 : {answer}")

db.close() # DB Disconnect