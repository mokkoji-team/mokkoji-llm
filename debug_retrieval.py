from indexing import load_persisted_index
from rag import normalize_query

index = load_persisted_index()

question = "동아리장이 되려는 사용자가 카카오로 최초 가입하면 처음 role은 뭐야?"

normalized = normalize_query(question)

print("original:", question)
print("normalized:", normalized)

retriever = index.as_retriever(
    similarity_top_k=10,
)

nodes = retriever.retrieve(normalized)

for i, node in enumerate(nodes, 1):
    print(f"\n--- rank {i} ---")
    print("score:", round(node.score, 3) if node.score else None)
    print("file:", node.metadata.get("file_name"))
    print(node.text[:1000])
