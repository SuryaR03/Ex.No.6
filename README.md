# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 13-11-2025
# Register no: 212223110056
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

### ⚙️ Primary AI Tools

To execute this project effectively, you’ll need access to the following APIs or libraries:

**1. OpenAI API (GPT, DALL·E, Whisper)**
Purpose: Text generation, summarization, conversation, embeddings, image, and audio tasks.
Integration: via openai Python SDK.
**2. Anthropic Claude API**
Purpose: Natural language reasoning, summarization, and safety-aligned content.
Integration: anthropic package.
**3. Cohere API**
Purpose: Classification, semantic search, and text embeddings.
Integration: cohere SDK.
**4. Hugging Face Transformers**
Purpose: Open-source models for NLP, vision, and multimodal tasks.
Integration: transformers and datasets.
**5. Google Generative AI (Gemini / PaLM)**
Purpose: Advanced text and code generation.
Integration: google-generativeai library.
**6. LangChain**
Purpose: Framework to link multiple AI tools in a single workflow.
Integration: langchain Python package.
**7. Vector Databases (Optional)**
Examples: Pinecone, Weaviate, ChromaDB
Purpose: For storing embeddings and enabling retrieval-augmented generation.

## 🚀 Setup Instructions
**1️⃣ Clone the Repository**

<pre><code>
git clone https://github.com/your-username/multi-ai-tools-automation.git
cd multi-ai-tools-automation
</code></pre>

**2️⃣ Install Required Packages**
<pre><code>
pip install openai cohere anthropic langchain transformers datasets google-generativeai
</code></pre>

**3️⃣ Configure API Keys**
Create a .env file in your project folder:
<pre><code>
OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
</code></pre>

**4️⃣ Execute Sample Script**
<pre><code>
python main.py
</code></pre>

**5️⃣ Optional Enhancements**
Add new tool adapters for other APIs.
Integrate vector databases for semantic retrieval.
Extend logic for visualization and reporting.

## Explanation

This experiment explores the concept of persona-based prompt programming for application-specific AI automation.
By connecting multiple APIs (like OpenAI and Cohere), the system can:

Send a uniform query to different models,
Collect their responses,
Compare semantic similarity and latency, and
Produce an automated analytical report.
This helps understand how different AI engines interpret and respond to the same task. 

# Conclusion:
**🧠 Multi-AI Integration and Analysis**

This project successfully demonstrates a unified framework for interacting with multiple AI tools, analyzing their responses, and extracting valuable insights.
The Python script showcases how to:

Dispatch a single prompt to multiple AI models,
Standardize the results,
Measure semantic similarity and response speed,
Rank providers by quality and efficiency, and
Save outputs for further study.

## Key Features

Unified API interface for multiple AI systems.
Automatic normalization of outputs.
Comparison metrics: response time, length, and semantic overlap.
Insight generation (agreement, diversity, latency).
JSON-based report export.

## Architecture Overview
<pre><code>
+-----------------+       +----------------+       +-----------------+
|   CLI / User    | ---> |  Controller    | --->  |  Provider A API |
+-----------------+       | (Coordinator)  |       +-----------------+
                          |                | --->  |  Provider B API |
                          +----------------+       +-----------------+
                                   |
                                   v
                          +-----------------+
                          | Comparator &    |
                          | Normalizer      |
                          +-----------------+
                                   |
                                   v
                          +-----------------+
                          | Insights / JSON |
                          +-----------------+

</code></pre>

## Environment Setup

1.Use Python 3.10+

2.Install requirements:
<pre><code>
pip install -r requirements.txt
</code></pre>

3.Add your API keys to .env
4.Run:
<pre><code>
python multi_ai_compare.py --prompt "Explain the importance of AI ethics in 3 points"
</code></pre>

## Dependencies

<pre><code>
requests
python-dotenv
numpy
scikit-learn
sentence-transformers
tqdm
</code></pre>

## .env Example
<pre><code>
  OPENAI_API_KEY=your_openai_key
  COHERE_API_KEY=your_cohere_key
</code></pre>

## Python Implementation (multi_ai_compare.py)
<pre><code>
  import os, time, json, argparse, requests
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

@dataclass
class NormalizedResponse:
    provider: str
    text: str
    token_count: int
    latency_ms: float
    metadata: Dict[str, Any]

def call_openai(prompt: str, model="gpt-4o", max_tokens=256) -> NormalizedResponse:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    start = time.time()
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    latency = (time.time() - start) * 1000
    data = res.json()
    text = data["choices"][0]["message"]["content"].strip()
    tokens = data.get("usage", {}).get("total_tokens", -1)
    return NormalizedResponse("openai", text, tokens, latency, {"raw": data})

def call_cohere(prompt: str, model="xlarge", max_tokens=256) -> NormalizedResponse:
    url = "https://api.cohere.ai/generate"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    start = time.time()
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    latency = (time.time() - start) * 1000
    data = res.json()
    text = data["generations"][0]["text"].strip()
    return NormalizedResponse("cohere", text, -1, latency, {"raw": data})

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def compare_outputs(responses: List[NormalizedResponse]) -> Dict[str, Any]:
    texts = [r.text for r in responses]
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True)
    sim = cosine_similarity(embeddings)
    results = {}
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            results[f"{responses[i].provider}_vs_{responses[j].provider}"] = float(sim[i, j])
    best_latency = min(responses, key=lambda r: r.latency_ms).provider
    return {"pairwise_similarity": results, "fastest_provider": best_latency}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()
    responses = []
    try: responses.append(call_openai(args.prompt))
    except Exception as e: print("OpenAI error:", e)
    try: responses.append(call_cohere(args.prompt))
    except Exception as e: print("Cohere error:", e)
    if not responses: return
    analysis = compare_outputs(responses)
    report = {"responses": [asdict(r) for r in responses], "comparison": analysis}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
</code></pre>

## Comparison Metrics

| Metric                | Description                                      |
| --------------------- | ------------------------------------------------ |
| **Cosine Similarity** | Measures how semantically close two outputs are. |
| **Latency**           | Identifies which AI tool responds faster.        |
| **Token Count**       | Indicates response length and approximate cost.  |

## Future Enhancements

Include adapters for Anthropic, Hugging Face, and Google APIs.
Add retry logic and logging for reliability.
Save and visualize results in dashboards or notebooks.
Extend semantic comparison to include sentiment and factual accuracy.


# Result:

The corresponding Prompt is executed successfully.
