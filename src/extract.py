"""1단계 — PDF 에서 텍스트·표·이미지를 추출한다."""
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf


@contextmanager
def _working_dir(path):
    """with 블록 동안만 작업 폴더(cwd)를 바꿉니다."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def extract_pdf_elements(pdf_path, figures_dir):
    """PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    pdf_path   : PDF 파일 경로
    figures_dir: 추출된 이미지 파일(figure-*.jpg)을 저장할 폴더

    주의: 설치된 unstructured 버전은 이미지를 "현재 작업 폴더/figures" 에 저장하고
    image_output_dir_path 인자는 무시합니다. 그래서 임시 폴더에서 추출한 뒤
    figures_dir 로 옮깁니다. (실행 위치와 상관없이 항상 figures_dir 에 저장되도록)
    추출이 성공하면 figures_dir 의 기존 figure-* 파일(이전 실행의 잔재)은 새 이미지로 교체됩니다.
    """
    pdf_path = Path(pdf_path).resolve()
    figures_dir = Path(figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with _working_dir(tmp):
            elements = partition_pdf(
                filename=str(pdf_path),
                languages=["kor", "eng"],           # 한국어+영어 OCR (미지정 시 영어만 → 글자 깨짐)
                strategy="hi_res",                  # 레이아웃 분석 모델로 정밀 추출
                extract_images_in_pdf=True,         # 그림을 별도 파일로 저장
                infer_table_structure=True,         # 표의 행·열 구조 추론
                chunking_strategy="by_title",       # 타이틀 기준으로 텍스트 분할
                max_characters=4000,                # 최대 4000자로 블록 제한
                new_after_n_chars=3800,             # 3800자 이후 새 블록 생성
                combine_text_under_n_chars=2000,    # 2000자 이하는 결합
            )
        # 추출이 성공한 뒤에만 이전 실행의 잔재를 지우고 새 이미지로 교체
        for old in figures_dir.glob("figure-*"):
            old.unlink()
        for img in (Path(tmp) / "figures").glob("*"):
            shutil.move(str(img), str(figures_dir / img.name))

    return elements


def categorize_elements(raw_pdf_elements):
    """PDF에서 추출한 요소들을 텍스트와 테이블로 분류합니다.

    raw_pdf_elements: unstructured.documents.elements 리스트
    반환: (texts, tables)
    """
    tables = []
    texts = []

    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    return texts, tables


def split_texts(texts, chunk_size=2000, chunk_overlap=200):
    """텍스트 블록을 토큰 기준으로 다시 분할합니다.

    chunk_size   : 각 청크의 최대 토큰 수
    chunk_overlap: 청크 간 겹치는 토큰 수 (경계에서 문맥이 끊기는 것을 방지)
    """
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(" ".join(texts))
