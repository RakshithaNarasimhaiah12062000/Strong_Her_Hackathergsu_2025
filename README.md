# 💪 StrongHer — AI Wellness Companion for Single Mothers  
**HackHers 2025 | Georgia State University**  
By: Avantika Balaji, Palak Kakani, Rakshitha Narasimhaiah, Shreya Vyas

## 🌸 Overview

**StrongHer** is an AI-powered postpartum and parenting support assistant built with a Retrieval-Augmented Generation (RAG) architecture. It provides single mothers with:

- 💬 Reliable Q&A on parenting, nutrition, mental health, legal & financial guidance  
- 📚 Verified resources from PDFs, CSVs, and expert-written documents  
- 🎨 A calming, modular Streamlit-based UI for journaling, meal planning, and more


## ⚙️ Tech Stack

| Layer         | Tools / Frameworks                                       |
|--------------|----------------------------------------------------------|
| 💻 Frontend   | `Streamlit`, `Custom CSS`, `Pandas`                      |
| 🧠 Backend    | `LangChain`, `OpenAI GPT-4o-mini`, `Prompt Engineering` |
| 🔍 Retrieval  | `FAISS` (or `Chroma`), `OpenAI Embeddings`              |
| 📁 Data       | `PDF`, `CSV`, `JSON` from `data/` & `static_content/`   |


## 🚀 Quickstart

```bash
# ✅ Step 1: Install Python dependencies
pip install -r requirements.txt

# ✅ Step 2: Add your OpenAI API key to a .env file
# (Do NOT share this key publicly)
echo "OPENAI_API_KEY=sk-..." > .env

# ✅ Step 3: Index your documents into the FAISS vector store
python phase1.py

# ✅ Step 4: Launch the Streamlit web app
streamlit run phase3.py

# 4. Run project phases
Step 1 : python phase1.py  # ➤ Index data into FAISS
Step 2 : streamlit run phase3.py  # ➤ Launch Streamlit app
