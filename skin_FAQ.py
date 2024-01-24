import streamlit as st
import pymysql

# 데이터베이스 연결 설정
def connect_db():
    return pymysql.connect(
        host='127.0.0.1',  # MySQL 서버 호스트를 127.0.0.1로 변경
        port=3306,          # MySQL 서버 포트 번호
        user='root',       # 데이터베이스 사용자 이름
        # password='your_password',  # 데이터베이스 비밀번호 (설정한 경우)
        db='skin',          # 데이터베이스 이름
        charset='utf8'
    )

# 데이터베이스 초기화 (데이터베이스 및 테이블 생성)
def initialize_db():
    conn = None  # 변수를 초기화합니다.
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            # 데이터베이스 생성
            cursor.execute("CREATE DATABASE IF NOT EXISTS skin;")
            cursor.execute("USE skin;")
            # inquiries 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inquiries (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    question TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    finally:
        if conn is not None:
            conn.close()

# 데이터베이스에 데이터 삽입
def insert_data(name, email, question):
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            sql = "INSERT INTO inquiries (name, email, question) VALUES (%s, %s, %s)"
            cursor.execute(sql, (name, email, question))
        conn.commit()
    finally:
        conn.close()

# 데이터베이스 초기화 함수 호출
initialize_db()

# Streamlit 애플리케이션
st.title('1:1 문의')

with st.form(key='contact_form'):
    user_name = st.text_input('이름')
    user_email = st.text_input('이메일')
    user_question = st.text_area('문의 내용')
    submit_button = st.form_submit_button(label='문의하기')

    if submit_button:
        insert_data(user_name, user_email, user_question)
        st.success('문의가 접수되었습니다. 빠른 시일 내에 답변 드리겠습니다.')
