# 📘 PDF Q&A App using Streamlit + LangChain + HuggingFace

An intelligent app that lets you upload multiple PDFs, ask questions from them, and even generate 20 question-answer pairs automatically.

---

## 🚀 Features
- 📂 Upload and process multiple PDFs
- 🤖 Ask questions from the uploaded content
- 🧠 Generate 20 Q&A pairs using Llama 3.1
- 📊 View analytics of key topics in the PDFs
- 📥 Download generated Q&A as PDF

---

## 🧰 Tech Stack
- **Streamlit** – Interactive frontend
- **LangChain** – Document chunking & retrieval
- **HuggingFace** – LLM inference & embeddings
- **FAISS** – Vector similarity search
- **ReportLab** – PDF generation

---

## 🧑‍💻 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/itsaaryanshetty/pdf_qa.git
cd pdf_qa

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
