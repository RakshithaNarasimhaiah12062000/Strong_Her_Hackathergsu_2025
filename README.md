# HackHers2025_GSU — Strong Her

AI-powered Parenting & Postpartum Wellness hub using a RAG pipeline.

## Tech Stack
- Python, LangChain, OpenAI Embeddings
- FAISS (or Chroma) for vector search
- python-dotenv for secrets

## Quickstart
\`\`\`bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Put your key in .env as: OPENAI_API_KEY=sk-...
python3 phase1.py   # builds vector index
python3 phase2.py   # query / app logic
python3 phase3.py   # UI / API
\`\`\`

## Repo Structure
- \`data/\` PDFs (knowledge base)
- \`static_content/\` CSV/JSON for app
- \`faiss_index/\` (ignored by default; rebuilt by phase1.py)

## Notes
- Do **not** commit \`.env\` (contains secrets).
- Git LFS tracks large assets: PDFs/PNGs/CSVs, etc.
