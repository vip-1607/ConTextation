import argparse
import os
import re
from collections import Counter

import matplotlib
# 화면용 GUI 라이브러리 없이도 PNG 파일을 만들 수 있는 백엔드
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


# 폰트 깨짐 방지
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# 문서 구조를 설명할 뿐 주제를 대표하지 않는 명사를 제거한다.
KOREAN_STOPWORDS = {
    "것", "수", "등", "때", "곳", "내용", "부분", "경우", "관련", "이것", "그것",
    "저것", "하나", "추이", "때문", "위", "아래", "이후", "이전", "이번", "저번",
    "각각", "모두", "전체", "중", "내", "외", "바", "점", "및",
}
NOUN_TAGS = {"NNG", "NNP"}

# Public market reports are generally English.  Keeping this compact, explicit
# list avoids a heavyweight English NLP model while filtering structural words.
ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "with", "will", "this", "these",
    "those", "we", "our", "you", "your", "they", "their", "has", "have", "had",
    "was", "were", "not", "but", "can", "could", "should", "may", "might", "about",
    "into", "than", "also", "more", "most", "other", "which", "while", "across",
    "source", "date", "url", "report", "research", "insight", "read", "learn",
    "after", "before", "been", "being", "both", "during", "each", "even", "every",
    "first", "get", "going", "here", "how", "just", "like", "much", "new", "now",
    "one", "only", "out", "over", "really", "see", "some", "such", "take", "then",
    "there", "through", "under", "very", "what", "when", "where", "who", "why", "would",
    "year", "goldman", "sachs", "morgan", "blackrock", "global", "all", "any", "because",
    "great", "important", "key", "know", "said", "say", "talk", "thank", "thanks", "think",
    "time", "way", "well", "within",
    "affiliate", "affiliates", "asset", "assets", "client", "clients", "data", "does",
    "financial", "fund", "funds", "including", "information", "investment", "investments",
    "management", "material", "portfolio", "portfolios", "please", "purpose", "purposes",
    "service", "services", "site", "website",
    "advice", "advisor", "advisors", "affiliate", "banking", "company", "companies", "disclosure",
    "guarantee", "help", "index", "insights", "investing", "its", "model", "models", "party",
    "privacy", "provided", "results", "return", "returns", "risk", "risks", "securities", "third",
    "back", "world", "ishares",
    "content", "link", "links", "opportunity", "opportunities", "performance", "policy", "policies",
    "resource", "resources", "security", "solution", "solutions", "tax", "term", "terms", "tool", "tools", "view",
}


def read_file(filepath):
    """텍스트 또는 PDF에서 분석할 본문을 읽는다."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    for encoding in ("utf-8", "cp949"):
        try:
            with open(filepath, "r", encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise ValueError("UTF-8 또는 CP949로 읽을 수 없는 텍스트 파일입니다.")


def preprocess_korean(text, min_term_freq):
    """명사 추출, 불용어 제거, 희소 단어 제거를 적용한 문장 목록을 반환한다."""
    kiwi = Kiwi()
    tokenized_docs = []
    for sentence in kiwi.split_into_sents(text):
        nouns = [
            token.form
            for token in kiwi.tokenize(sentence.text)
            if token.tag in NOUN_TAGS
            and len(token.form) >= 2
            and token.form not in KOREAN_STOPWORDS
        ]
        if nouns:
            tokenized_docs.append(nouns)

    term_counts = Counter(token for doc in tokenized_docs for token in doc)
    valid_terms = {
        term for term, count in term_counts.items() if count >= min_term_freq
    }
    return [" ".join(token for token in doc if token in valid_terms)
            for doc in tokenized_docs
            if sum(token in valid_terms for token in doc) >= 2]


def preprocess_english(text, min_term_freq):
    """Tokenize English reports without requiring a separate language model."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    tokenized_docs = []
    for sentence in sentences:
        tokens = [
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z'-]+", sentence)
            if (len(token) >= 3 or token.lower() == "ai")
            and token.lower() not in ENGLISH_STOPWORDS
        ]
        if tokens:
            tokenized_docs.append(tokens)

    term_counts = Counter(token for doc in tokenized_docs for token in doc)
    valid_terms = {term for term, count in term_counts.items() if count >= min_term_freq}
    return [" ".join(token for token in doc if token in valid_terms)
            for doc in tokenized_docs
            if sum(token in valid_terms for token in doc) >= 2]


def preprocess_text(text, min_term_freq):
    """Choose Korean or English preprocessing from the dominant alphabet."""
    korean_chars = len(re.findall(r"[가-힣]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))
    if english_chars > korean_chars:
        return preprocess_english(text, min_term_freq)
    return preprocess_korean(text, min_term_freq)


def extract_tfidf_keywords(docs, max_keywords):
    """문장별 TF-IDF 합계가 높은 단어만 핵심어로 선택한다."""
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    order = np.argsort(-scores, kind="stable")[:max_keywords]
    return [terms[index] for index in order]


def extract_topic_keywords(docs, max_keywords):
    """BERTopic을 우선 사용하고, 사용할 수 없으면 TF-IDF로 안전하게 전환한다."""
    if len(docs) < 15:
        return extract_tfidf_keywords(docs, max_keywords)

    try:
        # PyTorch 같은 대형 의존성이 없는 환경에서도 짧은 문서 분석은 계속 가능하다.
        from bertopic import BERTopic

        model = BERTopic(nr_topics="auto", language="multilingual")
        model.fit_transform(docs)
        topic_info = model.get_topic_info()
        keywords = []
        for _, topic in topic_info.iterrows():
            if topic["Topic"] == -1:
                continue
            for word in topic["Name"].split("_"):
                if not word.isdigit() and word not in keywords:
                    keywords.append(word)
                    if len(keywords) >= max_keywords:
                        return keywords
        if keywords:
            return keywords
    except (ImportError, OSError, RuntimeError) as error:
        print(f"BERTopic을 사용할 수 없어 TF-IDF로 전환합니다: {error}")

    return extract_tfidf_keywords(docs, max_keywords)


def build_cooccurrence_matrix(docs, edge_threshold):
    """인접 단어쌍의 대칭적 동시 등장 확률(Jaccard)을 계산한다."""
    term_counts = Counter()
    pair_counts = Counter()
    for doc in docs:
        tokens = doc.split()
        term_counts.update(tokens)
        for left, right in zip(tokens, tokens[1:]):
            if left != right:
                pair_counts[tuple(sorted((left, right)))] += 1

    terms = sorted({term for pair in pair_counts for term in pair})
    matrix = pd.DataFrame(0.0, index=terms, columns=terms)
    for (left, right), pair_count in pair_counts.items():
        # P(같이 등장 | 둘 중 하나가 등장): 순서와 무관한 0~1 범위의 관계 강도
        union_count = term_counts[left] + term_counts[right] - pair_count
        probability = pair_count / union_count if union_count else 0.0
        if probability > edge_threshold:
            matrix.loc[left, right] = probability
            matrix.loc[right, left] = probability
    return matrix


def filter_by_degree(graph, min_degree):
    """연결 수가 기준 이하인 노드를 반복 제거해 최종 그래프 품질을 보장한다."""
    filtered = graph.copy()
    while True:
        low_degree_nodes = [
            node for node, degree in filtered.degree() if degree < min_degree
        ]
        if not low_degree_nodes:
            return filtered
        filtered.remove_nodes_from(low_degree_nodes)


def get_node_size(node_values):
    values = np.asarray(list(node_values), dtype=float)
    if len(values) == 0:
        return values
    spread = values.max() - values.min()
    if spread == 0:
        return np.full(len(values), 400.0)
    return 300 + 900 * (values - values.min()) / spread


def layout_with_label_spacing(graph):
    """라벨 길이를 고려해 노드 간 간격을 넓히고, 가까운 라벨을 반복적으로 밀어낸다."""
    node_count = graph.number_of_nodes()
    scale = max(3.5, min(10.0, node_count * 0.26))
    positions = nx.spring_layout(
        graph,
        k=2.1 / np.sqrt(max(node_count, 1)),
        iterations=350,
        seed=17,
        weight="weight",
        scale=scale,
    )

    nodes = list(graph.nodes())
    for _ in range(180):
        moved = False
        for index, left in enumerate(nodes):
            for right in nodes[index + 1:]:
                delta = positions[right] - positions[left]
                distance = np.linalg.norm(delta)
                # 한글·영문 글자 수를 반영한 보수적인 최소 라벨 간격이다.
                minimum = 0.55 + 0.09 * (len(str(left)) + len(str(right)))
                if distance >= minimum:
                    continue
                if distance < 1e-9:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta / distance
                shift = (minimum - distance) / 2
                positions[left] -= direction * shift
                positions[right] += direction * shift
                moved = True
        if not moved:
            break

    center = np.mean(list(positions.values()), axis=0)
    return {node: position - center for node, position in positions.items()}


def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Tool (Korean)")
    parser.add_argument("input", help="텍스트 파일 또는 PDF 경로")
    parser.add_argument("--min-term-freq", type=int, default=6,
                        help="유지할 단어의 최소 전체 등장 횟수 (기본값: 6)")
    parser.add_argument("--max-keywords", type=int, default=40,
                        help="그래프에 사용할 최대 핵심 키워드 수 (기본값: 40)")
    parser.add_argument("--edge-threshold", type=float, default=0.06,
                        help="동시 등장 확률 임계값 (기본값: 0.06)")
    parser.add_argument("--min-degree", type=int, default=3,
                        help="최종 그래프에서 유지할 최소 연결 수 (기본값: 3)")
    parser.add_argument("--output", default="knowledge_graph.png", help="출력 PNG 경로")
    args = parser.parse_args()

    if args.min_term_freq < 1 or args.max_keywords < 1 or args.min_degree < 0:
        parser.error("최소 빈도와 최대 키워드는 1 이상, 최소 연결 수는 0 이상이어야 합니다.")

    print("[1/5] 문서 읽기 및 언어별 전처리 (불용어·희소 단어 제거)...")
    raw_text = read_file(args.input)
    docs = preprocess_text(raw_text, args.min_term_freq)
    if not docs:
        print("유의미한 문장을 추출하지 못했습니다. --min-term-freq 값을 낮춰 보세요.")
        return

    print(f"[2/5] 키워드 추출 (문장 수: {len(docs)})...")
    try:
        topic_list = extract_topic_keywords(docs, args.max_keywords)
    except Exception as error:
        print(f"키워드 추출 실패: {error}")
        return
    if not topic_list:
        print("핵심 키워드를 추출하지 못했습니다.")
        return

    filtered_docs = []
    keyword_set = set(topic_list)
    for doc in docs:
        keywords = [word for word in doc.split() if word in keyword_set]
        if len(keywords) >= 2:
            filtered_docs.append(" ".join(keywords))
    if not filtered_docs:
        print("키워드가 함께 등장한 문장이 없습니다. --min-term-freq 값을 낮춰 보세요.")
        return

    print("[3/5] 대칭 동시 등장 행렬 계산...")
    matrix = build_cooccurrence_matrix(filtered_docs, args.edge_threshold)
    graph = nx.from_pandas_adjacency(matrix)
    graph.remove_nodes_from(list(nx.isolates(graph)))

    print("[4/5] 연결 수가 낮은 노드 제거...")
    graph = filter_by_degree(graph, args.min_degree)
    if graph.number_of_nodes() == 0:
        print("연결된 핵심어가 없습니다. --edge-threshold 또는 --min-degree 값을 낮춰 보세요.")
        return

    print("[5/5] 지식 그래프 렌더링...")
    canvas_size = max(13, min(24, 9 + graph.number_of_nodes() * 0.35))
    plt.figure(figsize=(canvas_size, canvas_size))
    plt.margins(x=0.16, y=0.16)
    plt.title("Knowledge Graph", fontsize=16)
    centrality = nx.betweenness_centrality(graph)
    positions = layout_with_label_spacing(graph)
    nx.draw_networkx(
        graph,
        pos=positions,
        node_color=list(centrality.values()),
        node_size=get_node_size(centrality.values()),
        font_size=10 if graph.number_of_nodes() > 24 else 11,
        alpha=0.8,
        cmap=plt.cm.Blues,
        font_family="Malgun Gothic",
        edge_color="gray",
        width=[1 + 4 * data["weight"] for _, _, data in graph.edges(data=True)],
    )
    plt.savefig(args.output, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"\n[완료] '{args.output}' 파일이 생성되었습니다.")


if __name__ == "__main__":
    main()
