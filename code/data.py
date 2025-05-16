#!/usr/bin/env python3

import os
import logging
import pickle
from pathlib import Path
import pdfplumber
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import TokenTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_openai_key() -> str:
    """Load OPENAI_KEY from .env."""
    load_dotenv()  
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Please set OPENAI_API_KEY in your .env")
    return key


def filter_dog_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV and keep only AnimalName == 'Dog'."""
    df = pd.read_csv(csv_path)
    df = df.query("AnimalName == 'Dog'")
    print("Filtered to Dog rows:", len(df))
    return df


def silence_pdfminer():
    """Suppress pdfminer warnings just like in Colab."""
    logging.getLogger("pdfminer").setLevel(logging.ERROR)


def preview_pdfs(pdf_paths: list[Path], pages: int = 3):
    """Exactly your Colab preview of first 3 pages."""
    for path in pdf_paths:
        fn = path.name
        with pdfplumber.open(path) as pdf:
            texts = [(page.extract_text() or "").strip() for page in pdf.pages[:pages]]
        combined = "\n\n--- Page Break ---\n\n".join(texts).strip()
        if combined:
            snippet = combined[:500].replace("\n", " ")
            print(f"\n=== Preview of {fn} (first {pages} pages) ===\n{snippet}")
        else:
            print(f"\n=== No text on first {pages} pages of {fn} ===")


def load_documents(df: pd.DataFrame, pdf_paths: list[Path]) -> list[Document]:
    """
    Build your `documents` list exactly as in Colab:
     1) full-text PDFs
     2) CSV rows with all columns
    """
    docs: list[Document] = []

    # PDFs → one Document per file
    for p in pdf_paths:
        with pdfplumber.open(p) as pdf:
            full = "\n".join(page.extract_text() or "" for page in pdf.pages)
        docs.append(Document(page_content=full, metadata={"source": str(p)}))

    # CSV rows → one Document per row
    for idx, row in df.iterrows():
        content = " ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(page_content=content,
                             metadata={"source": "conditioncls.csv", "row": idx}))

    print(f"Loaded {len(pdf_paths)} PDFs + {len(df)} CSV rows = {len(docs)} documents")
    return docs


def view_doc(documents: list[Document]):
    """Same as your DataFrame preview in Colab."""
    previews = []
    for d in documents:
        src = d.metadata["source"]
        snippet = d.page_content.replace("\n", " ")[:200]
        previews.append({"source": src, "preview": snippet})
    dfp = pd.DataFrame(previews)
    print(dfp.head())


def split_documents(
    documents: list[Document],
    chunk_size: int = 300,
    chunk_overlap: int = 10,
    encoding_name: str = "cl100k_base"
) -> list[Document]:
    """TokenTextSplitter exactly as you used it."""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name=encoding_name
    )
    out: list[Document] = []
    for doc in documents:
        for i, chunk in enumerate(splitter.split_text(doc.page_content)):
            out.append(Document(
                page_content=chunk,
                metadata={"source": doc.metadata["source"], "chunk": i}
            ))
    print("Total token-based chunks:", len(out))
    return out


def save_docs(documents: list[Document], path: Path):
    with open(path, "wb") as f:
        pickle.dump(documents, f)
    print(f"\nSaved {len(documents)} docs to {path}")


def load_docs(path: Path) -> list[Document]:
    with open(path, "rb") as f:
        docs = pickle.load(f)
    print(f"\nLoaded {len(docs)} docs from {path}")
    return docs


def build_faiss_index(
    documents: list[Document],
    embedder: OpenAIEmbeddings,
    index_path: Path
) -> FAISS:
    """FAISS.from_documents + save_local exactly as you did."""
    vs = FAISS.from_documents(documents, embedder)
    vs.save_local(str(index_path))
    print(f"\nSaved faiss_index to {index_path}")
    return vs


def load_faiss_index(
    index_path: Path,
    embedder: OpenAIEmbeddings
) -> FAISS:
    print(f"\nLoaded faiss_index from {index_path}")
    return FAISS.load_local(
        str(index_path),
        embedder,
        allow_dangerous_deserialization=True
    )


def create_qa_chain(
    vector_store: FAISS,
    openai_key: str,
    model_name: str = "gpt-4o",
    k: int = 5
) -> RetrievalQA:
    """RetrievalQA.from_chain_type exactly as in Colab."""
    llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def create_conversational_chain(vs, openai_key):
    llm = ChatOpenAI(openai_api_key=openai_key)
    retriever = vs.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

ROOT = Path(__file__).parent.parent

def main():
    openai_key = get_openai_key()
    print(openai_key)

    # — your original paths —
    ROOT = Path(__file__).parent.parent
    CSV_PATH = ROOT / "data" / "conditioncls.csv"
    DATA_DIR = ROOT / "data"
    PDF_FILES = [
        "Approved-Home-Remedies-for-Dog.pdf",
        "dog-facts-the-pet-parent's-atoz-home-care-encyclopedia.pdf",
        "veterinary-clinical-pathology-a-casebased-approach.pdf",
    ]
    PDF_PATHS = [DATA_DIR / fn for fn in PDF_FILES]
    PICKLE_PATH = ROOT / "data" / "documents.pkl"
    INDEX_PATH = ROOT / "data" / "faiss_index"

    # 1) CSV → df
    df = filter_dog_csv(CSV_PATH)

    # 2) Silence warnings
    silence_pdfminer()

    # 3) Preview PDFs
    preview_pdfs(PDF_PATHS)

    # 4) Load & preview Documents
    docs = load_documents(df, PDF_PATHS)
    view_doc(docs)

    # 5) Split
    split_docs = split_documents(docs)

    # 6) load doc/save if needed 
    if not PICKLE_PATH.exists():
        save_docs(split_docs, PICKLE_PATH)
    # split_docs = load_docs(PICKLE_PATH)

    # 7) Embed & FAISS
    embedder = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_key
    )
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        vs = build_faiss_index(split_docs, embedder, INDEX_PATH)
    vs = load_faiss_index(INDEX_PATH, embedder)

    # # 8) Retriever + QA
    # qa = create_qa_chain(vs, openai_key)
    # # qa = create_conversational_chain(vs, openai_key)

    # # 9) Test
    # history = []
    # while True:
    #     query = input("\nAsk a question (or blank to quit): ").strip()
    #     if not query:
    #         print("Goodbye! 👋")
    #         break

    #     resp = qa({"query": query})
    #     answer = resp.get("result", "")
    #     history.append((query, answer))

    #     # print the latest answer
    #     print(f"\n→ {answer}")

    #     # optionally, show full chat so far
    #     print("\n–– Chat history ––")
    #     for q,a in history:
    #         print(f"You: {q}\nBot: {a}\n")
    
    
    # -------------------------------------------------------------------------------------------------
    # 8) Create the conversational chain
    qa = create_conversational_chain(vs, openai_key)

    # 9) Multi-turn REPL
    print("\nHello! Ask me anything about your dog. (press Enter on empty line to quit)\n")
    while True:
        user_q = input("You: ").strip()
        if not user_q:
            print("Goodbye! 👋")
            break

        # Pass your question under the 'question' key
        resp = qa.invoke({"question": user_q})
        answer = resp["answer"]

        # Show latest turn
        print(f"Assistant: {answer}\n")

# if __name__ == "__main__":
#     main()
