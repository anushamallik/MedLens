import streamlit as st
import fitz  # PyMuPDF
import re
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from deep_translator import GoogleTranslator
from gtts import gTTS
import pygame
import time
from langchain.chains import StuffDocumentsChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.llm import LLMChain

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="MedLens", page_icon="🧠")
st.title(":brain: MedLens: Making Complex Medical Reports Crystal Clear")

# ------------------ SESSION INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ------------------ PATIENT NAME EXTRACTION ------------------
def extract_patient_name(text):
    match = re.search(r'(?i)(Patient|Patient\s+Name|Name)\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    return match.group(2).strip() if match else "Patient name not found."

# ------------------ FINDINGS EXTRACTION ------------------
def extract_findings_section(text):
    match = re.search(r'(?i)findings\s*[:\-]*\s*(.+?)(?=\n[A-Z][A-Za-z ]{2,}[:\-])', text, re.DOTALL)
    return match.group(1).strip() if match else "Findings section not found."

# ------------------ CLEAN GIBBERISH ------------------
def clean_findings_text(text):
    lines = text.splitlines()
    cleaned = [line for line in lines if len(line.strip()) > 3 and not re.search(r'[^a-zA-Z0-9\s.,:;()%\-]', line)]
    return '\n'.join([line for line in cleaned if not line.strip().isupper()]).strip()

# ------------------ LOAD EXISTING SUMMARIES DB ------------------
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    df = pd.read_csv("C:/Users/Abhilash/OneDrive/Desktop/Mini Project/ReportSummary.csv", encoding='unicode_escape')
    df = df[df['Summary'].notna()]
    documents = [Document(page_content=f"Report:\n{row['Report']}\n\nSummary:\n{row['Summary']}", metadata={"id": i}) for i, row in df.iterrows()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return Chroma.from_documents(docs, embedding=GPT4AllEmbeddings(), persist_directory="./medical_summaries_dB")

vectorstore = load_vectorstore()

# ------------------ PROMPT ------------------
summary_prompt = ChatPromptTemplate.from_template("""
You are a medical assistant that summarizes complex radiology findings into simple, easy-to-understand language for patients and their families.

Instructions:
- Do NOT use medical jargon.
- Explain technical terms using everyday words.
- Mention what the findings mean for the patient.
- Include relevant organs or areas affected.
- Make it conversational and friendly, like a doctor talking to a patient.
- Keep the summary detailed but not overwhelming. Length should be roughly 60–75% of the original findings.

Context (previous medical summaries for reference):
{context}

Report:
{question}

Now write the summary in simple and clear language that anyone can understand:
""")

qa_chain = RetrievalQA.from_chain_type(
    llm=Ollama(model="phi3"),
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": summary_prompt}
)

# ------------------ PDF UPLOAD ------------------
st.subheader(":page_facing_up: Upload a Radiology Report PDF")
uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")

if uploaded_pdf is not None:
    with open("temp_uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    doc = fitz.open("temp_uploaded.pdf")
    full_text = "".join(page.get_text() for page in doc)

    patient_name = extract_patient_name(full_text)
    raw_findings = extract_findings_section(full_text)
    clean_findings = clean_findings_text(raw_findings)

    st.write(f"**:bust_in_silhouette: Patient Name:** {patient_name}")
    st.write("**:receipt: Extracted Findings:**")
    st.code(clean_findings, language="text")

    if st.button(":brain: Generate Summary"):
        with st.spinner("Summarizing..."):
            summary = qa_chain.run(clean_findings)
            st.session_state.summary = summary
        st.success("✅ Summary Generated:")
        st.markdown(st.session_state.summary)

# ------------------ Language Selection ------------------
st.markdown("---")
language_map = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn"
}
selected_language = st.selectbox(":globe_with_meridians: Select language for translation and speech", list(language_map.keys()))

# ------------------ Read Aloud Button ------------------
if st.button(":loud_sound: Read Aloud Summary"):
    try:
        summary = st.session_state.get("summary", "")
        if summary:
            translated = GoogleTranslator(source="auto", target=language_map[selected_language]).translate(summary)
            st.markdown(f"**:memo: Summary in {selected_language}:**")
            st.markdown(translated)

            tts = gTTS(translated, lang=language_map[selected_language])
            audio_path = "summary_audio.mp3"
            tts.save(audio_path)

            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.5)

            pygame.mixer.music.stop()
            pygame.mixer.quit()
            os.remove(audio_path)
        else:
            st.warning("Please generate a summary first.")
    except Exception as e:
        st.error(f"Translation/Speech Error: {e}")

# ------------------ Chatbot ------------------
st.markdown("---")
st.subheader(":robot_face: Chat with MedLens")

chat_input = st.text_input("Ask a follow-up question about the report or findings:")

if chat_input and st.session_state.summary:
    with st.spinner("Thinking..."):
        followup_prompt = ChatPromptTemplate.from_template("""
        You are a medical assistant. Use the context to answer user questions clearly in layman's terms.

        Context:
        {context}

        Question:
        {question}

        Answer in 2-3 sentences.
        """)

        llm_chain = LLMChain(llm=Ollama(model="phi3"), prompt=followup_prompt)

        combine_docs_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )

        chatbot_chain = RetrievalQA(
            retriever=vectorstore.as_retriever(),
            combine_documents_chain=combine_docs_chain
        )

        response = chatbot_chain.run({"query": chat_input})
        st.markdown(f"**:robot_face: Answer:** {response}")
elif chat_input:
    st.warning("Please generate a summary first to enable the chatbot.")
