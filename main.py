import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough


def build_or_load_vectorstore(speech_path: str, persist_dir: str = "chroma_db") -> Chroma:
    speech_file = Path(speech_path)
    if not speech_file.exists():
        raise FileNotFoundError(f"Speech file not found: {speech_file}")

    if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        print(f"Using existing ChromaDB at '{persist_dir}'...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("Building new vectorstore from input text...")

    docs = TextLoader(str(speech_file), encoding="utf8").load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return db


def create_qa_chain(db):
    llm = Ollama(model="llama3.2:1b-instruct-q8_0")
    retriever = db.as_retriever()

    prompt = PromptTemplate.from_template(
        """
        Use only the provided context to answer.

        Context:
        {context}

        Question: {question}

        Respond concisely and rely solely on the above context.
        """
    )

    chain = (
        RunnableParallel(
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
    )

    return chain


def main():
    speech_path = "speech.txt"
    persist_dir = "chroma_db"

    db = build_or_load_vectorstore(speech_path, persist_dir)
    qa_chain = create_qa_chain(db)

    print("\n=== Ambedkar Speech Q&A ===")
    print("Type a question related to the speech.")
    print("Enter 'exit' to close.\n")

    while True:
        question = input("Question: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            answer = qa_chain.invoke(question)
        except Exception as e:
            print("Error:", e)
            continue

        print("\n--- Answer ---")
        print(answer)
        print("\n")


if __name__ == "__main__":
    main()
