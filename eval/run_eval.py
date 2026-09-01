import json
from pathlib import Path
from rag import ask

QUESTIONS_PATH = Path(__file__).parent / "questions.jsonl"


def load_questions(path: Path) -> list[dict]:
    questions = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            questions.append(json.loads(line))

    return questions


if __name__ == "__main__":
    questions = load_questions(QUESTIONS_PATH)

    for question in questions:
        print(f"\n[{question['id']}] {question['question']}")

        result = ask(question["question"])

        print("answer:", result["answer"])

        for i, source in enumerate(result["sources"], 1):
            print(f"\n--- source {i} ---")
            print("file:", source["file_name"])
            print("score:", source["score"])
            print("text:")
            print(source["text"])
