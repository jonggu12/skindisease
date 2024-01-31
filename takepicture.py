import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import requests
import os
import tempfile
from sklearn.preprocessing import LabelEncoder

# 함수 정의
def load_metadata(file_path):
    return pd.read_csv(file_path)

def encode_user_input(metadata_df, age, sex_input, localization_input):
    sex_dict = {"남자": "male", "여자": "female"}
    localization_dict = {
        "복부": "abdomen", "등": "back", "가슴": "chest", "얼굴": "face",
        "발": "foot", "생식기": "genital", "다리": "lower extremity",
        "목": "neck", "두피": "scalp", "몸통": "trunk", "알수없음": "unknown",
        "팔": "upper extremity", "귀": "ear", "손바닥": "acral", "손": "hand",
    }
    sex = sex_dict.get(sex_input, "unknown")
    localization = localization_dict.get(localization_input, "unknown")
    sex_encoded = 0 if sex == "male" else 1
    label_encoder = LabelEncoder().fit(metadata_df['localization'])
    localization_encoded = label_encoder.transform([localization])[0]
    return np.array([[age, sex_encoded, localization_encoded]])

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_skin_disease(model, img_array, meta_input):
    predictions = model.predict([img_array, meta_input])
    return predictions[0]

# 메타데이터 로드
metadata_df = load_metadata('HAM10000_metadata.csv')

# 모델 파일 URL
model_file_url = "https://myjonggu.s3.ap-southeast-2.amazonaws.com/dense201_0125.h5"

# 모델 파일을 임시 파일로 다운로드
response = requests.get(model_file_url)
if response.status_code == 200:
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='wb')
    temp_file.write(response.content)
    temp_file.close()

    # 모델 로드
    model = load_model(temp_file.name)

    # 임시 파일 삭제
    os.remove(temp_file.name)
else:
    st.error("모델 파일 다운로드 실패: HTTP 상태 코드 {}".format(response.status_code))

# 클래스 이름 매핑
class_names = {
    0: '피부 선암 (Actinic keratoses and intraepithelial carcinoma)',
    1: '기저세포암 (Basal cell carcinoma)',
    2: '벤인 케라토시스 라이크 레이즈니즈 (Benign keratosis-like lesions)',
    3: '피부 섬유종 (Dermatofibroma)',
    4: '흑색종 (Melanoma)',
    5: '멜라닌성 낭종 (Melanocytic nevi)',
    6: '혈관 병변 (Vascular lesions)'
}

class_descriptions = {
    0: '이 질환은 피부의 선피로 인한 변화를 나타내며 햇빛에 노출된 피부에서 발생합니다.',
    1: '기저세포암은 피부 기저세포에서 시작되는 흔한 피부암입니다.',
    2: '이 질환은 양성 피부 병변을 나타내며 주로 피부의 형태가 이상한 병변을 포함합니다.',
    3: '피부 섬유종은 피부의 섬유성 종양으로서 일반적으로 작고 단단한 결절로 나타납니다.',
    4: '흑색종은 피부암 중에서 가장 악성이고 위험한 종류로, 피부의 흑색 색소세포인 멜라닌 세포에서 발생합니다.',
    5: '멜라닌 세포에서 나타나는 양성 피부 종양으로 흔한 모양과 색상의 주근깨 또는 점으로 표시됩니다.',
    6: '혈관 이상을 나타내며 흔히 어지러운 혈관 패턴 또는 혈관의 이상이 있는 피부 병변을 포함합니다.'
}

def main():
    st.title("피부 질환 감지 애플리케이션 - 사진찍기 모드")
    
    webrtc_ctx = webrtc_streamer(key="example")

    if webrtc_ctx.video_transformer:
        if st.button("사진찍기"):
            with webrtc_ctx.video_transformer.frame_lock:
                if webrtc_ctx.video_transformer.frame is not None:
                    frame = webrtc_ctx.video_transformer.frame

                    # OpenCV로 이미지 처리
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    st.image(img, caption="캡처된 사진")

                    # 사용자 입력 받기
                    age = st.number_input("나이를 입력하세요", min_value=1, max_value=100, value=30)
                    sex = st.selectbox("성별을 선택하세요", ["남자", "여자"])
                    localization = st.selectbox("질환 부위를 선택하세요", ["복부", "등", "가슴", "얼굴", "발", "생식기", "다리", "목", "두피", "몸통", "알수없음", "팔", "귀", "손바닥", "손"])
                
                    # 캡처된 사진을 피부질환 모델로 예측
                    img = cv2.resize(img, (224, 224))  # 모델 입력 크기에 맞게 리사이즈
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0) / 255.0

                    # 메타데이터 인코딩
                    meta_input = encode_user_input(metadata_df, age, sex_input, localization_input)

                    # 예측 수행
                    predictions = predict_skin_disease(model, img, meta_input)

                    # 예측 결과 표시
                    top_class_index = np.argmax(predictions)
                    top_class_name = class_names[top_class_index]
                    top_probability = predictions[top_class_index] * 100

                    st.write("피부 질환 예측 결과:")
                    st.write(f"1순위: {top_class_name} ({top_probability:.2f}%)")
                    if top_class_index in class_descriptions:
                        st.write("설명:", class_descriptions[top_class_index])

                    # 2순위 예측
                    second_class_index = np.argsort(-predictions)[1]  # 2순위 클래스의 인덱스
                    second_class_name = class_names[second_class_index]
                    second_probability = predictions[second_class_index] * 100

                    st.write(f"2순위: {second_class_name} ({second_probability:.2f}%)")
                    if second_class_index in class_descriptions:
                        st.write("설명:", class_descriptions[second_class_index])

                    # 3순위 예측
                    third_class_index = np.argsort(-predictions)[2]  # 3순위 클래스의 인덱스
                    third_class_name = class_names[third_class_index]
                    third_probability = predictions[third_class_index] * 100

                    st.write(f"3순위: {third_class_name} ({third_probability:.2f}%)")
                    if third_class_index in class_descriptions:
                        st.write("설명:", class_descriptions[third_class_index])

                    # 피부 상태 제품 서비스로 이동하는 하이퍼링크 추가
                    st.markdown("### [여기를 클릭하여 피부 상태 제품 추천 서비스로 이동](https://app-eqmlagtpuerscreh89pscg.streamlit.app/)")

if __name__ == "__main__":
    main()