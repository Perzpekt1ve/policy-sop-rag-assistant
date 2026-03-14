from pathlib import Path
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HF_EMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    RETRIEVAL_K,
)
from prompts import SYSTEM_PROMPT


def load_pdf_documents(file_paths: Iterable[Path]) -> List[Document]:
    docs = []

    for path in file_paths:
        loader = PyPDFLoader(str(path))
        file_docs = loader.load()

        for doc in file_docs:
            doc.metadata["file_name"] = path.name
            doc.metadata["file_path"] = str(path)

        docs.extend(file_docs)

    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)


def build_vectorstore(chunks: List[Document], persist_directory: Path = CHROMA_DIR):
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory),
    )

    return vectorstore


def load_vectorstore(persist_directory: Path = CHROMA_DIR):
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )


def get_retriever(vectorstore):
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def format_context(docs: List[Document]) -> str:
    formatted_chunks = []

    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("file_name", "Unknown file")
        page = doc.metadata.get("page", "N/A")
        text = doc.page_content.strip()

        chunk = f"[Source {idx}] File: {source} | Page: {page}\n{text}"
        formatted_chunks.append(chunk)

    return "\n\n".join(formatted_chunks)


def answer_question(vectorstore, question: str):
    retriever = get_retriever(vectorstore)
    retrieved_docs = retriever.invoke(question)
    context = format_context(retrieved_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Use the context below to answer the user question.\n\nContext:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    llm = ChatOllama(
        model=OLLAMA_CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return {
        "answer": answer,
        "source_documents": retrieved_docs,
    }