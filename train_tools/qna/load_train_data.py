# 챗봇 학습 데이터 불러오기
import sys
sys.path.append('.')

import pymysql
import openpyxl
from config.DatabaseConfig import *

# 학습 데이터 초기화
def all_clear_train_data(db):
    # 기존 학습 데이터 삭제
    sql = '''
    DELETE FROM chatbot_train_data
    '''
    
    with db.cursor() as cursor:
        cursor.execute(sql)
        
        
    # Auto Increment 초기화
    sql = '''
    ALTER TABLE chatbot_train_data AUTO_INCREMENT = 1
    '''
    
    with db.cursor() as cursor:
        cursor.execute(sql)

# db에 데이터 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_img_url = xls_row
    
    sql = f'''
    INSERT chatbot_train_data(intent, ner, query, answer, answer_image)
        VALUES('{intent.value}', '{ner.value}', '{query.value}', '{answer.value}', '{answer_img_url.value}')
    '''
    
    # 엑셀에서 불러온 cell에 데이터가 없는 경우, null로 치환
    sql = sql.replace("'None'", "null")
    
    with db.cursor() as cursor:
        cursor.execute(sql)
        print(f"{query.value} 저장")
        db.commit()

train_file = "train_tools/qna/train_data.xlsx"
db = None
try:
    db = pymysql.connect(
        host = DB_HOST,
        port = DB_PORT,
        user = DB_USERNAME,
        passwd = DB_PASSWORD,
        database= DB_DATABASE,
        charset = "utf8"
    )
    
    # 기존 학습 데이터 초기화
    all_clear_train_data(db)
    
    # 학습 엑셀 파일 불러오기
    wb = openpyxl.load_workbook(train_file)
    sheet = wb["Sheet1"]
    for row in sheet.iter_rows(min_row=2): # 헤더는 불러오지 않음
        # 데이터 저장
        insert_data(db, row)
        
    wb.close()
except Exception as e:
    print(f"Error : {e}")
    
finally:
    if db is not None:
        db.close()
