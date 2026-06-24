# QTrade Support Interface: Assistant QT

#### Note: View the write-up [here](https://github.com/buwituze/Assistant-Beni/blob/main/write-up.txt).

A Customer Support Assistant designed for QTrade smart-home devices. Built using Python, a clean object-oriented architecture, and the modern Google GenAI SDK, Assistant QT safely answers customer queries using strict Retrieval-Augmented Generation (RAG) principles.

The system implements strict deterministic guardrails to split safe informational responses ("I don't know") from high-priority human hands-off ("Escalation").

## Key Architectural Decisions

1. **Deterministic Multi-Stage Triage**: Instead of relying on a single complex prompt to handle routing and answering simultaneously, the pipeline splits safety/frustration classification into a high-speed pre-inference step (`EscalationGuard`), isolating logic from the primary generation step.

2. **Strict RAG Grounding**: The system uses `gemini-embedding-2` to convert local support knowledge sections into vector points, retrieving context using a vectorized cosine-similarity matrix via `numpy`. Assistant QT is strictly bound to the retrieved context to prevent structural hallucinations.

3. **Graceful Exceptions**: Any timeline threshold violations or high-liability hardware structural breakdowns (such as devices popping, sparking, or smoking) automatically exit the automation loop and cleanly pass to human supervision.

4. **Contact Information**: The user/customer is iven contact information to reach out to in case of esclation instead of leavin them frustrated.

## Project Structure

```text
Assistant-Beni/
│
├── .env                # Secure configuration for the Gemini API key
├── requirements.txt    # Application package dependencies
├── help_docs.txt       # Local knowledge base text documents
└── app.py              # Central OOP application execution engine
```

## Prerequisites & Installation

Ensure you have Python 3.10+ installed on your system.

Navigate to your project directory:

```bash
cd C:\Users\HP\Programming\ML\Assistant-Beni
```

Initialize and activate a clean virtual environment:

```powershell
# Using PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory to store your credentials securely.

```plaintext
GEMINI_API_KEY=AIzaSyYourActualGeminiAPIKeyHere
```

## Running the Assistant

Ensure your virtual environment is active, then execute the main runtime file:

```bash
python app.py
```

## Verification Sample Queries

You can use the following patterns inside the command-line chat session to verify Assistant QT's routing accuracy:

- **Standard Q&A**: `"How do I reset my SmartHub?"` — Assistant QT answers from the knowledge base and appends a source citation block.

- **Out of Scope (Safe handling)**: `"Do you offer bulk discounts for commercial installs?"` — Assistant QT safely replies with: _"I don't know. I am sorry, but that information is not available..."_

- **Partial Match / Timeline Breach**: `"My order hasn't shipped in 4 days, where is it?"` — Triggers the business rule fallback to connect to priority customer support.

- **Severe Hardware Failure / Defect Triage**: `"I plugged the bulb in, it short-circuited and popped with a bit of smoke."` — Instantly intercepts the request and safely transfers to a human supervisor.
