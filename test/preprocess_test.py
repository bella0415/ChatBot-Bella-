
from utils.Preprocess import Preprocess

sent = "내일 오전 10시에 짬뽕 주문하고 싶어"

# 전처리 객체 생성
p = Preprocess()

# 형태소 분석기 실행
pos = p.pos(sent)

# 품사 태그와 같이 키워드 출력
ret = p.get_keywords(pos, without_tag=False)
print(ret)

# 품사 태그없이 키워드 출력
ret = p.get_keywords(pos, without_tag=True)
print(ret)