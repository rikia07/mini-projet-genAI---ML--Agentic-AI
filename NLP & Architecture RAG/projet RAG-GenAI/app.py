import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback

# === CONFIG PAGE ===
st.set_page_config(page_title="Gemini PDF Chatbot", layout="wide")
st.title(" Gemini PDF Chatbot")

# === CHARGER VARIABLES D'ENVIRONNEMENT ===
load_dotenv()

# === BARRE LATERALE : UPLOAD DE PDF ===
st.sidebar.header(" Téléversement du PDF")
if st.sidebar.button(" Réinitialiser la conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

uploaded_file = st.sidebar.file_uploader(
    "Choisissez un fichier PDF",
    type=["pdf"],
    accept_multiple_files=False
)

# === INITIALISER L'HISTORIQUE DU CHAT ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === AFFICHER L'HISTORIQUE DU CHAT ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === TRAITEMENT SI PDF UPLOADE ===
if uploaded_file:
    st.sidebar.success(f" Fichier chargé : {uploaded_file.name}")

    try:
        # === EXTRACTION TEXTE PDF ===
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

        if not raw_text.strip():
            st.error(" Aucun texte n'a été extrait du PDF. Essayez un autre fichier.")
        else:
            st.write(" Aperçu du texte extrait :")
            st.write(raw_text[:1000])  # Aperçu 1000 caractères

            # === CHUNKING ===
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(raw_text)
            st.success(f" Texte divisé en {len(chunks)} chunks")

            # === EMBEDDINGS + FAISS ===
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

            documents = [Document(page_content=chunk) for chunk in chunks]
            vector_store = FAISS.from_documents(documents, embeddings)
            st.success(" Vector store créé avec succès ! Vous pouvez maintenant poser des questions.")

            # === ZONE DE CHAT ===
            user_input = st.chat_input("Posez votre question sur le PDF...")

            if user_input:
                # Afficher question dans l'historique
                st.chat_message("user").markdown(user_input)
                st.session_state.messages.append({"role": "user", "content": user_input})

                # === SPECIAL CASE : "Call me" → Formulaire ===
                if "call me" in user_input.lower() or "appelle-moi" in user_input.lower():
                    with st.form("contact_form"):
                        st.info(" Merci de compléter ce formulaire pour être recontacté.")
                        name = st.text_input("Nom complet")
                        phone = st.text_input("Téléphone")
                        email = st.text_input("Email")
                        submitted = st.form_submit_button("Envoyer")

                        if submitted:
                            # Crée le fichier avec entête s'il n'existe pas encore
                            if not os.path.exists("contacts.csv"):
                                with open("contacts.csv", "w", newline="", encoding="utf-8") as f:
                                    writer = csv.writer(f)
                                    writer.writerow(["timestamp", "name", "phone", "email"])

                            # Ajoute une ligne avec les données
                            with open("contacts.csv", "a", newline="", encoding="utf-8") as f:
                                writer = csv.writer(f)
                                writer.writerow([datetime.now().isoformat(), name, phone, email])

                            st.success(" Vos informations ont été enregistrées.")
    
                    # Empêche d'appeler le LLM si c'était juste pour remplir le formulaire
                    st.stop()

                # === INITIALISER LLM GEMINI + PROMPT ===
                llm = ChatGoogleGenerativeAI(
                    model="models/gemini-1.5-pro-001",
                    temperature=0.2,
                    google_api_key=os.getenv("GOOGLE_API_KEY")
                )

                prompt_template = """
                Tu es un assistant expert en recherche clinique.
                Réponds de manière claire uniquement à partir du contexte fourni.
                Si l'information n'est pas dans le document, dis : "Je ne sais pas."

                CONTEXTE :
                {context}

                QUESTION :
                {question}
                """

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template=prompt_template
                )

                # === CREER LA CHAINE QA ===
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vector_store.as_retriever(),
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True # Pour afficher les sources
                )

                # === RÉPONDRE À LA QUESTION ===
                response = qa_chain.invoke({"query": user_input})
                answer = response["result"]

                # Afficher réponse dans l'historique
                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # === SOURCES AFFICHÉES ===
                sources = response.get("source_documents", [])
                if sources:
                    with st.expander("📚 Sources utilisées (chunks du PDF)"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(doc.page_content[:500])
                            st.markdown("---")

    except Exception as e:
        st.error(" Une erreur est survenue.")
        st.exception(e)
