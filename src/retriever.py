"""3단계 — 다중 벡터 검색기와 최종 멀티모달 체인."""
import re
import uuid

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ID_KEY = "doc_id"                       # 요약과 원본을 잇는 청구기호
EMBEDDING_MODEL = "text-embedding-3-small"
ANSWER_INSTRUCTION = (
    "아래 자료를 근거로 질문에 한국어로 답하세요.\n"
    "자료에 없는 내용은 모른다고 답하세요.\n\n"
)
# Base64 문자열은 영문/숫자/+/= 만으로 길게 이어진다
_BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")


def create_retriever(collection_name="multimodal_rag"):
    """다중 벡터 검색기를 만듭니다.

    vectorstore(Chroma)  : 요약문 임베딩 — 검색용 (색인 카드)
    docstore(InMemory)   : 원본 텍스트·표·Base64 이미지 — 전달용 (서가)
    반환: (retriever, vectorstore)
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),
        id_key=ID_KEY,
    )
    return retriever, vectorstore


def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def is_image(content):
    """문자열이 Base64 이미지인지 판별합니다."""
    return bool(_BASE64_PATTERN.match(content))


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    b64_images = []
    texts_out = []
    for doc in docs:
        content = doc.page_content if isinstance(doc, Document) else doc
        if is_image(content):
            b64_images.append(content)
        else:
            texts_out.append(content)
    return {"images": b64_images, "texts": texts_out}


def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM이 읽을 형태로 조립합니다."""
    messages = []

    # 1) 이미지를 먼저 넣는다 (모델이 그림을 먼저 보게)
    for image in data_dict["context"]["images"]:
        messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
        })

    # 2) 텍스트 컨텍스트와 질문
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages.append({
        "type": "text",
        "text": (
            ANSWER_INSTRUCTION
            + f"질문: {data_dict['question']}\n\n"
            + f"참고 텍스트:\n{formatted_texts}"
        ),
    })
    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever, model=None):
    """최종 멀티모달 RAG 체인: 검색 → 이미지/텍스트 분리 → 프롬프트 조립 → LLM."""
    model = model or ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def build_text_only_chain(texts, model=None, k=4, collection_name="text_only_rag"):
    """비교용 텍스트 전용 RAG 체인 (Before). 이미지·표 요약 없이 텍스트 청크만 색인합니다."""
    model = model or ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=1024)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )
    store.add_texts(texts)
    text_retriever = store.as_retriever(search_kwargs={"k": k})

    prompt = ChatPromptTemplate.from_template(
        ANSWER_INSTRUCTION + "질문: {question}\n\n참고 텍스트:\n{context}"
    )
    return (
        {
            "context": text_retriever | (lambda docs: "\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
