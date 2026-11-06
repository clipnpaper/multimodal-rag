# 아키텍처 다이어그램

SK네트웍스 *2026년 3분기 경영실적* PDF(`SK_networks.pdf`)를 대상으로 한 **다중 벡터(Multi-Vector) 멀티모달 RAG**의 전체 흐름입니다.
파이프라인은 **① 데이터 추출 → ② 요약 생성 → ③ 색인·검색·답변** 3단계로 나뉩니다.

```mermaid
flowchart TD
    PDF[/"SK_networks.pdf<br/>14쪽 슬라이드형 경영실적"/]

    subgraph S1["① 데이터 추출 — notebook 1장"]
        direction TB
        EXT["unstructured.partition_pdf<br/>strategy=hi_res · OCR kor+eng<br/>표 구조 추론 · 이미지 추출"]
        TXT["텍스트 블록<br/>CompositeElement"]
        TBL["표<br/>Table"]
        IMG["이미지<br/>figures/*.jpg 49장"]
        SPL["토큰 기준 분할<br/>chunk 2000 · overlap 200"]
        EXT --> TXT & TBL & IMG
        TXT --> SPL
    end

    subgraph S2["② 요약 생성 — notebook 2장"]
        direction TB
        SUMT["텍스트 요약<br/>gpt-4o-mini · 한국어 프롬프트"]
        SUMB["표 요약<br/>gpt-4o-mini · 한국어 프롬프트"]
        B64["Base64 인코딩"]
        SUMI["이미지 요약<br/>gpt-4o-mini Vision"]
        B64 --> SUMI
    end

    subgraph S3["③ 색인 · 검색 · 답변 — notebook 3장"]
        direction TB
        VS[("Chroma vectorstore<br/>요약문 임베딩<br/>text-embedding-3-small")]
        DS[("InMemoryStore docstore<br/>원본 텍스트 · 표 · Base64 이미지")]
        MVR{{"MultiVectorRetriever<br/>id_key = doc_id"}}
        SPLIT["split_image_text_types<br/>Base64 이미지 / 텍스트 분리"]
        PROMPT["img_prompt_func<br/>이미지 먼저 + 텍스트·질문"]
        LLM["gpt-4o-mini<br/>멀티모달 답변 생성"]
        ANS(["답변"])
        VS --> MVR
        DS --> MVR
        MVR --> SPLIT --> PROMPT --> LLM --> ANS
    end

    Q(["사용자 질문"])

    PDF --> EXT
    SPL --> SUMT
    TBL --> SUMB
    IMG --> B64

    SUMT -- "요약 + doc_id" --> VS
    SUMB -- "요약 + doc_id" --> VS
    SUMI -- "요약 + doc_id" --> VS

    SPL -. "원본 + 같은 doc_id" .-> DS
    TBL -. "원본 + 같은 doc_id" .-> DS
    B64 -. "원본 + 같은 doc_id" .-> DS

    Q --> MVR
    Q --> PROMPT
```

## 읽는 법

| 선 | 의미 |
|---|---|
| 실선 | 데이터 흐름 |
| 점선 | 원본을 docstore에 보관하는 경로 (요약과 **같은 `doc_id`** 를 붙여 저장) |

## 핵심 아이디어: "요약으로 찾고, 원본으로 답한다"

- **검색은 요약으로** 합니다. 차트·표 이미지는 그대로는 임베딩해서 글로 검색할 수 없으므로, 먼저 한국어 요약문으로 바꿔 벡터화합니다.
- **답변은 원본으로** 합니다. 요약에는 숫자가 빠지거나 틀릴 수 있으므로, 검색에 걸린 요약의 `doc_id`로 docstore에서 **원본(텍스트·표·이미지)** 을 꺼내 LLM에 줍니다.
- 이 때문에 멀티모달 LLM은 요약이 아니라 **실제 차트 이미지**를 보고 수치를 읽습니다.

## 질의 시 흐름 (3단계 중 ③)

1. 질문을 임베딩해 Chroma에서 **요약문**과 가까운 항목을 찾는다.
2. 각 요약의 `doc_id`로 docstore에서 **원본**을 꺼낸다.
3. 원본을 정규식으로 **이미지(Base64)** 와 **텍스트/표**로 나눈다.
4. 이미지를 먼저, 그 뒤에 질문과 참고 텍스트를 넣어 `HumanMessage`를 만든다.
5. `gpt-4o-mini`가 답변을 생성한다.
