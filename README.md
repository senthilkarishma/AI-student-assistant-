"""
AI Learning & Study Assistant
------------------------------
Agentic AI application built using Ollama (local LLM), ChromaDB (RAG),
JSON-based memory, and custom tool-calling for study plans & quizzes.

Requirements (install before running):
    pip install ollama chromadb

Pull a model locally first (run in terminal):
    ollama pull llama3
    ollama serve   # if not already running
"""

import ollama
import chromadb
import json
import os
import re
from datetime import datetime, timedelta

MODEL = "llama3"          # change to any model you have pulled, e.g. "mistral"
MEMORY_FILE = "student_memory.json"

# ----------------------------------------------------------------------
# 1. MEMORY MODULE — stores student progress across sessions
# ----------------------------------------------------------------------
class Memory:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {"topics_covered": [], "quiz_scores": [], "weak_areas": []}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def add_topic(self, topic):
        if topic not in self.data["topics_covered"]:
            self.data["topics_covered"].append(topic)
        self.save()

    def add_quiz_score(self, topic, score, total):
        self.data["quiz_scores"].append(
            {"topic": topic, "score": score, "total": total, "date": str(datetime.now().date())}
        )
        if score / total < 0.6 and topic not in self.data["weak_areas"]:
            self.data["weak_areas"].append(topic)
        self.save()

    def summary(self):
        return json.dumps(self.data, indent=2)


# ----------------------------------------------------------------------
# 2. RAG MODULE — retrieves relevant chunks from course material
# ----------------------------------------------------------------------
class CourseRAG:
    def __init__(self, collection_name="course_materials"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self._counter = 0

    def _embed(self, text):
        resp = ollama.embeddings(model=MODEL, prompt=text)
        return resp["embedding"]

    def add_document(self, text, source="notes.txt", chunk_size=400):
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for chunk in chunks:
            self._counter += 1
            self.collection.add(
                documents=[chunk],
                embeddings=[self._embed(chunk)],
                ids=[f"{source}_{self._counter}"],
            )
        print(f"[RAG] Indexed {len(chunks)} chunks from {source}")

    def retrieve(self, query, top_k=3):
        q_emb = self._embed(query)
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n---\n".join(docs)


# ----------------------------------------------------------------------
# 3. TOOLS — agent-callable actions
# ----------------------------------------------------------------------
def generate_study_plan(topics, days, hours_per_day):
    prompt = f"""
    Create a {days}-day study plan covering these topics: {', '.join(topics)}.
    The student can study {hours_per_day} hours per day.
    Break it down day-by-day with specific sub-topics and short goals.
    Keep it concise and in a clean list format.
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def generate_quiz(context, topic, num_questions=5):
    prompt = f"""
    Based on the following course material, generate {num_questions} multiple-choice
    quiz questions on the topic "{topic}". For each question give 4 options (A-D),
    mark the correct answer, and give a one-line explanation.
    Format strictly as JSON list with keys: question, options, answer, explanation.

    Course material:
    {context}
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"]
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return raw
    return raw


def answer_question(context, question):
    prompt = f"""
    Answer the student's question using ONLY the course material below.
    If the answer isn't in the material, say so honestly.

    Course material:
    {context}

    Question: {question}
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# ----------------------------------------------------------------------
# 4. AGENT — routes student input to the right tool
# ----------------------------------------------------------------------
class StudyAssistantAgent:
    def __init__(self):
        self.rag = CourseRAG()
        self.memory = Memory()

    def load_material(self, text, source="notes.txt"):
        self.rag.add_document(text, source=source)

    def handle(self, user_input):
        lower = user_input.lower()

        if "quiz" in lower:
            topic = user_input.split("on")[-1].strip() if "on" in lower else "general"
            context = self.rag.retrieve(topic)
            quiz = generate_quiz(context, topic)
            self.memory.add_topic(topic)
            return {"type": "quiz", "content": quiz}

        elif "study plan" in lower or "revision plan" in lower:
            topics = self.memory.data["topics_covered"] or ["general revision"]
            plan = generate_study_plan(topics, days=5, hours_per_day=2)
            return {"type": "study_plan", "content": plan}

        else:
            context = self.rag.retrieve(user_input)
            answer = answer_question(context, user_input)
            self.memory.add_topic(user_input[:30])
            return {"type": "answer", "content": answer}


# ----------------------------------------------------------------------
# 5. DEMO RUN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    agent = StudyAssistantAgent()

    sample_notes = """
    Photosynthesis is the process by which green plants convert light energy
    into chemical energy stored in glucose. It occurs in the chloroplasts,
    primarily using chlorophyll to absorb sunlight. The process has two main
    stages: the light-dependent reactions (in the thylakoid membrane) and
    the light-independent reactions, or Calvin cycle (in the stroma).
    Oxygen is released as a byproduct of the light-dependent reactions.
    """
    agent.load_material(sample_notes, source="biology_ch4.txt")

    print("\n--- Q&A ---")
    result = agent.handle("What is photosynthesis and where does it occur?")
    print(result["content"])

    print("\n--- Quiz ---")
    result = agent.handle("quiz me on photosynthesis")
    print(json.dumps(result["content"], indent=2))

    print("\n--- Study Plan ---")
    result = agent.handle("give me a study plan")
    print(result["content"])

    print("\n--- Memory Summary ---")
    print(agent.memory.summary())# study_ai_application# AI-student-assistant-
