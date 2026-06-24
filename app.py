"""QTrade Support Assistant: An interface for customer support using Google Gemini API."""

import os
import sys
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors

class DocumentStore:
    """Parses text documents, generates embeddings, and handles retrieval via cosine similarity."""
    def __init__(self, client: genai.Client):
        self.client = client
        self.titles = []
        self.texts = []
        self.embeddings = []
        self.embedding_model = "gemini-embedding-2"

    def load_documents(self, filepath: str):
        """Loads and embeds documents from a text file into the store."""
        if not os.path.exists(filepath):
            print(f"[Error] Knowledge file missing at: {filepath}")
            sys.exit(1)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for chunk in chunks:
            lines = chunk.split("\n", 1)
            title = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else chunk
            try:
                response = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text
                )
                self.titles.append(title)
                self.texts.append(text)
                self.embeddings.append(response.embeddings[0].values)
            except errors.APIError as e:
                print(f"[API Error] Could not embed context segment '{title}': {e}")
                sys.exit(1)

    def retrieve_closest(self, query: str) -> tuple:
        """Finds the single most mathematically relevant document for the query."""
        try:
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=query
            )
            query_vector = response.embeddings[0].values
        except errors.APIError:
            return self.titles[0], self.texts[0]

        similarities = []
        for doc_vector in self.embeddings:
            dot = np.dot(query_vector, doc_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_d = np.linalg.norm(doc_vector)
            similarity = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
            similarities.append(similarity)
        best_index = int(np.argmax(similarities))
        return self.titles[best_index], self.texts[best_index]

class EscalationGuard:
    """Evaluates upfront signs of critical problems to handle human routing decisions."""
    def __init__(self, client: genai.Client):
        self.client = client

    def requires_human(self, query: str) -> bool:
        """Returns True if the query requires escalation to a human agent."""
        prompt = (
            "You are a triage support classifier for QTrade smart-home retail devices.\n"
            "Analyze the customer query and reply with exactly 'TRUE' if it satisfies any condition, "
            "or 'FALSE' if it does not:\n\n"
            "1. Severe Hardware Defect/Safety: The device popped, smoked, short-circuited, sparked, or exploded.\n"
            "2. Management Escalation: The user explicitly demands a manager/supervisor or yells out of clear ongoing anger.\n"
            "3. Multi-Failure: The user states this is a repeat unresolved attempt (e.g., 'third time calling').\n\n"
            f"Customer Input: \"{query}\"\n"
            "Output strictly 'TRUE' or 'FALSE':"
        )
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": 5
                }
            )
            return "TRUE" in response.text.strip().upper()
        except errors.APIError:
            return False

class SupportAssistant:
    """Coordinates state mapping between the safety gate, vector search, and model completion."""
    def __init__(self, store: DocumentStore, guard: EscalationGuard, client: genai.Client):
        self.store = store
        self.guard = guard
        self.client = client

    def answer_query(self, query: str) -> str:
        """Generates a support response for the given customer query."""
        if self.guard.requires_human(query):
            return "[ESCALATE] Transferring to a Customer Support Agent supervisor. Please contact our live priority queue directly at support@qtrade.com or call +250 7864645756."
        title, context_text = self.store.retrieve_closest(query)

        system_rules = (
            "You are an automated support desk agent for QTrade.\n"
            "Answer the question using ONLY the block provided in the Context section below.\n\n"
            "Strict rules:\n"
            "1. If the info answers the prompt, output a brief answer and append the source exactly as: (Source: [Doc Name])\n"
            "2. If the context explicitly states a timeline breach that requires support (like shipping exceeding 3 days), output: [ESCALATE_TRIGGER]\n"
            "3. If the answer cannot be confidently found in the context block, respond exactly with: I don't know.\n"
        )

        prompt = f"Context:\n{context_text}\n\nQuestion:\n{query}"

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "system_instruction": system_rules,
                    "temperature": 0.0,
                    "max_output_tokens": 200
                }
            )
            output = response.text.strip()

            if "[ESCALATE_TRIGGER]" in output:
                return f"[ESCALATE] This condition meets criteria for priority customer service according to {title}. Please reach out to support@qtrade.com or call +1-800-555-0199 for direct assistance."
            if "I don't know" in output:
                return "I don't know. I am sorry, but that information is not available in our current documentation."

            if title not in output and "I don't know" not in output:
                output += f" (Source: {title})"
            return output
        except errors.APIError as e:
            return f"[Runtime Error] API communication failure: {str(e)}"


def main():
    """Entry point: initializes the client, store, and runs the support loop."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[Error] Missing GEMINI_API_KEY environment variable. Please check your .env file.")
        sys.exit(1)

    print("Building local support indexes from help_docs.txt...")
    client = genai.Client(api_key=api_key)
    store = DocumentStore(client)
    store.load_documents("help_docs.txt")
    guard = EscalationGuard(client)
    assistant = SupportAssistant(store, guard, client)

    print("\n=======================================================")
    print("       QTrade Support Voice Interface: Assistant QT    ")
    print("=======================================================")
    print("[Online] Type 'exit' or 'quit' to close the connection.\n")
    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Closing support session. Goodbye.")
                break
            reply = assistant.answer_query(user_input)
            print(f"Assistant QT: {reply}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting cleanly.")
            break

if __name__ == "__main__":
    main()
