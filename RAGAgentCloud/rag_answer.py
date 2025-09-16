from langchain.chains import ConversationalRetrievalChain
from RAGAgentCloud.llm import llm
from RAGAgentCloud.retriever import get_collection_retriever
from RAGAgentCloud.prompt import prompt, condense_prompt
import logging

logger = logging.getLogger(__name__)

def get_rag_answer(question: str, chat_history=None):
    if chat_history is None:
        chat_history = []
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=get_collection_retriever(),
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    result = rag_chain({"question": question, "chat_history": chat_history})
    answer = result.get("answer") or result.get("result") or ""
    sources = result.get("source_documents", []) or []
    return {
        "answer": answer,
        "sources": sources
    }