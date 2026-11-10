"""2단계 — 텍스트·표·이미지를 검색용 요약문으로 만든다."""
import base64
import os
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

MODEL_NAME = "gpt-4o-mini"

TEXT_PROMPT_KOR = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

# 지시는 영어(모델이 안정적으로 따름), 출력은 한국어(질문과 같은 언어로 색인)
IMAGE_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Provide a concise summary of the image that is well optimized for retrieval.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False):
    """텍스트 및 표 데이터를 요약하여 검색에 활용할 요약본을 생성합니다.

    texts          : 텍스트 리스트
    tables         : 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부
    """
    prompt = ChatPromptTemplate.from_template(TEXT_PROMPT_KOR)
    model = ChatOpenAI(temperature=0, model=MODEL_NAME)

    # 요약 체인 = 입력 포장 + 프롬프트 + LLM + 출력 파서
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = []
    table_summaries = []

    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts  # 요약하지 않으면 원본을 그대로 사용

    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})

    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지 파일을 Base64 문자열로 변환합니다."""
    with open(file=image_path, mode="rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def image_summarize(img_base64, prompt=IMAGE_PROMPT):
    """Base64 이미지 한 장을 멀티모달 LLM 으로 요약합니다."""
    chat = ChatOpenAI(model=MODEL_NAME, max_tokens=1024)
    msg = chat.invoke([
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            },
        ])
    ])
    return msg.content


def generate_img_summaries(path):
    """폴더 안의 모든 이미지를 Base64 로 인코딩하고 요약합니다.

    반환: (img_base64_list, image_summaries)
    """
    img_base64_list = []
    image_summaries = []

    for element in sorted(os.listdir(Path(path))):
        if element.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(path, element)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image))

    return img_base64_list, image_summaries
