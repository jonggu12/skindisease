import streamlit as st
# FAQ 데이터
faq_data = [
    {"question": "Q1 : 이 웹사이트에서 피부질환을 어떻게 판별하나요? ", "answer": "사이트는 고급 이미지 인식 알고리즘을 사용하여 사용자가 업로드한 피부 사진을 분석하고, 피부질환의 가능성이 있는 항목을 식별합니다."},
    {"question": "Q2 : 판별 결과의 정확도는 얼마나 되나요? ", "answer": "우리의 알고리즘은 지속적으로 의료 전문가들의 도움으로 훈련되고 있으며, [평균 정확도(퍼센트기재)] 이상을 자랑합니다. 그러나 모든 진단 도구와 마찬가지로 100% 정확도를 보장할 수는 없으므로, 결과는 전문가의 의견을 대체할 수 없습니다."},
    {"question": "Q3 : 판별 결과를 어떻게 해석해야 하나요?", "answer": "웹사이트는 가능한 피부질환에 대한 개요와 함께 판별 결과를 제공합니다. 이 정보는 의료 전문가와 상의할 때 유용한 참고 자료가 될 수 있으나, 최종 진단은 전문가가 해야 합니다."},
    {"question": "Q4 : 판별 후 의료 상담이 필요하다면 어디로 연락해야 하나요?", "answer": "판별 결과에 대한 상세한 상담이 필요하시다면, 웹사이트 내의 [의료 상담 연결] 서비스를 이용하시거나, 가까운 피부과 클리닉에 직접 예약하실 수 있습니다."},
    {"question": "Q5 : 판별 결과가 불확실하거나 예상과 다를 경우 어떻게 해야 하나요?", "answer": "결과에 의문이 있거나 예상과 다를 경우, 가까운 피부과 전문의와 상의하시는 것이 가장 좋습니다. 웹사이트는 초기 판별을 제공하는 도구일 뿐, 전문 의료 서비스를 대체할 수 없습니다."},
]
# 스트림릿 앱 시작
st.title("유비케어 자주찾는 FAQ 페이지")
# 대화형 스타일로 FAQ 표시
for faq_item in faq_data:
    with st.expander(faq_item["question"]):
        st.markdown(f'A:&nbsp; {faq_item["answer"]}', unsafe_allow_html=True)
