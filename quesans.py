from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import re

import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)
embedding_function = HuggingFaceEmbeddings()

st.title("Understanding PDFs better")

# uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_files = st.file_uploader("Upload a PDF", type=['pdf'], accept_multiple_files=True)

text=""
if uploaded_files:
    for uploaded_file in uploaded_files:

        st.sidebar.success(f"{uploaded_file.name} uploaded")
        pdf = PdfReader(uploaded_file)
        st.sidebar.write(f"Pages: {len(pdf.pages)}")
        st.sidebar.write(f"Size: {round(len(uploaded_file.getvalue())/1024, 2)} KB")
        for page in pdf.pages:
            text+=page.extract_text() +"\n"
    
    st.write("Combined text from all PDFs:")
    st.text_area("Extracted Text", text, height=300)

    with st.spinner("📖 Processing PDF..."):
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
        )
        docs = [Document(page_content=text)]
        split_docs = text_splitter.split_documents(docs)
        pdf_text = " ".join([doc.page_content for doc in split_docs])


        # Create FAISS vector DB
        vector_store = FAISS.from_documents(split_docs, embedding=embedding_function)
        vector_store.save_local("my_qa_db")
        vector_store = FAISS.load_local(
        "my_qa_db",
        embedding_function,
        allow_dangerous_deserialization=True
        )
    st.success("✅ PDF processed and indexed!")
            # user_input = st.text_input("Type 1 for full pdf question answer formation")

    tab1, tab2, tab3 = st.tabs(["Generate Q&A", "Ask Questions", "Analytics Tab"])
    theme = st.get_option("theme.base")
    
    # if(user_input=="1"):
    with tab1:
        st.subheader("📌 Generate 20 Question & Answers")
        if st.button("Generate 20 Q&A"):
            # run Q&A generation code
            system_instruction = SystemMessage(
                content=("Form 20 question answers in the format \n"
                         "Q) ...... \n"
                         "A) ...... \n" 
                         "Each Question answer should be on separate lines \n"
                         "And use some text generation. The questions should be formed with the help of the content present in the pdf\n"
                         "The questions created should not be vague and use text generation and provide a little big answer\n"
                         "Try giving some good level questions related to the topics\n"
                         "Keep in mind not to create irrelevant questions which might not be helpful at all even if the number of questions reduces."
                        )
            )

            messages_for_model = [
                system_instruction,
                HumanMessage(content=pdf_text)
            ]
            result = model.invoke(messages_for_model)
            st.markdown("### Questions and Answers")
            # st.write(result.content)
            # theme = st.get_option("theme.base")
            if theme == "dark":
                card_bg = "#2b2b2b"
                text_color = "#ffffff"
            else:
                card_bg = "#f9f9f9"
                text_color = "#000000"

            qa_pairs = re.split(r"Q\)", result.content)
            for qa in qa_pairs[1:]:
                parts = qa.strip().split("A)")
                if len(parts) == 2:
                    question, answer = parts
                    st.markdown(f"""
                    <div style="padding:15px; margin-bottom:10px; border-radius:10px; background-color:{card_bg}; color:{text_color}; box-shadow:0px 2px 5px rgba(0,0,0,0.1);">
                    <b>Q)</b> {question.strip()}<br>
                    <b>A)</b> {answer.strip()}
                    </div>
                    """, unsafe_allow_html=True)

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            # text = c.beginText(50, height - 50)
            # text.setFont("Helvetica", 12)

            # for line in result.content.split("\n"):
            #     text.textLine(line)
    
            # c.drawText(text)
            # c.save()
            # buffer.seek(0)
            for line in result.content.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["Normal"]))
                    story.append(Spacer(1, 6))  # spacing between lines

            doc.build(story)
            buffer.seek(0)

        # --- Streamlit download button ---
            st.download_button(
                label="📥 Download Q&A as PDF",
                data=buffer,
                file_name="questions_answers.pdf",
                mime="application/pdf"
            )


    with tab2:
        st.subheader("Ask question from concepts in PDF")
        user_input = st.text_input("Type your question here...")
        if user_input:
        # run similarity search + answer


    
            with st.spinner("🤔 Searching for the answer..."):
                relevant_docs = vector_store.similarity_search(user_input, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])


                system_instruction = SystemMessage(
                content=f"You are a helpful assistant. Only answer using the following document context:\n\n{context}. Dont exactly copy paste the content from the \n"
                "from the pdf. Use some text generation provide a different answer with some considerable amount of content."
                
                )


                messages_for_model = [
                system_instruction,
                HumanMessage(content=user_input)
                ]
                result = model.invoke(messages_for_model)

            st.markdown("### ✅ Answer")
            if theme == "dark":
                ans_bg = "#2b2b2b"
                ans_text = "#ffffff"
                border_color = "#4CAF50"
            else:
                ans_bg = "#f1f8f4"
                ans_text = "#000000"
                border_color = "#4CAF50"

            st.markdown(f"""
            <div style="padding:15px; border-left:5px solid {border_color}; background-color:{ans_bg}; border-radius:5px; color:{ans_text};">
            {result.content}
            </div>
            """, unsafe_allow_html=True)


            with st.expander("📖 Show context used for answer"):
                for i, doc in enumerate(relevant_docs, 1):
                    st.markdown(f"**Chunk {i}:** {doc.page_content}")


    with tab3:
        stop_words = set(stopwords.words('english'))
        full_text = " ".join([doc.page_content for doc in split_docs])
        full_text = full_text.lower()
        full_text = re.sub(f"[{string.punctuation}]", " ", full_text)

        tokens = word_tokenize(full_text)

        filtered_tokens = [w for w in tokens if w not in stop_words and len(w)>2]

        word_counts = Counter(filtered_tokens)

        top_5 = word_counts.most_common(5)
        labels, values = zip(*top_5)

        fig, ax = plt.subplots()
        with plt.style.context('dark_background'):
            ax.bar(labels, values)
            ax.set_title("Top 5 important topics")
            ax.set_ylabel("Frequency")
            plt.xticks(rotation=30, ha='right')

            st.pyplot(plt.gcf())
            plt.clf()

        # --- Pie Chart ---
        fig2, ax2 = plt.subplots()
        wp = {'linewidth': 1, 'edgecolor': "black"}
        explode = (0.1, 0.0, 0.2, 0.3, 0.0)
        colors = ("orange", "cyan", "beige",
          "grey", "indigo")

        wedges, texts, autotexts = ax2.pie(
            values,
            explode=explode,
            labels=labels,
            shadow = True,
            startangle=90,
            wedgeprops=wp,
            colors=colors,
            textprops=dict(color="black"),
            autopct='%1.1f%%'  # ✅ Added back to show percentages
        )

        ax2.set_title("Top 5 Topics in PDF")
        st.pyplot(plt.gcf())
        plt.clf()
