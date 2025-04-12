# MedLens
MedLens is an AI-powered tool built with Streamlit that simplifies complex radiology reports into clear, detailed, and patient-friendly summaries. Designed for non-medical users, MedLens extracts key findings from radiology PDFs and generates easy-to-understand summaries, making medical jargon accessible to everyone.

Features
1.Radiology Report Summarization: Uses Ollama Phi-3 for generating detailed, layman-friendly summaries from complex medical texts.
2.Document Processing with LangChain: Utilizes LangChain to process and split the documents, ensuring accurate extraction of relevant information.
3.Vector Storage and Retrieval: Chroma and GPT4All embeddings help retrieve and generate contextually relevant summaries using pre-existing medical data.
4.Multi-language Support: Summaries can be translated into regional Indian languages like Hindi, Telugu, Tamil, and more, using GoogleTranslator.
5.Text-to-Speech: Summaries can be read aloud for enhanced accessibility with gTTS (Google Text-to-Speech).
6.Interactive Chatbot: Users can ask follow-up questions about the report and get simplified answers via LangChain’s RetrievalQA.

Installation
To run MedLens locally, follow these steps:
1. Clone the repository:
git clone https://github.com/your-username/MedLens.git
cd MedLens

2.Install required dependencies:
pip install -r requirements.txt

3.Run the app:
streamlit run app.py
