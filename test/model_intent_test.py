import sys
sys.path.append('.')

from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic="train_tools/dict/chatbot_dict.bin")

intent = IntentModel(model_name="models/intent/intent_model.h5", preprocess=p)

query = "주문해줘"

predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print(f"의도 예측 클래스 : {predict}")
print(f"의도 예측 레이블 : {predict_label}")