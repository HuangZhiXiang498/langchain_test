"""Create a ChatVectorDBChain for question/answering."""
from langchain import PromptTemplate
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import (
    CONDENSE_QUESTION_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore, VectorStoreRetriever

prompt_template = """Use the following pieces of context to answer the question in Chinese at the end.

{context}

Question: {question}
Helpful Answer:"""

prompt_template_Chinese = """Given the following conversation and a follow up question ,use  answer in Chinese .

Chat History:
{chat_history}
Follow Up Input: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
QA_PROMPT_Chinese = PromptTemplate(
    template=prompt_template_Chinese, input_variables=["chat_history", "question"]
)


def get_chain(
        vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering. ConversationalRetrievalChain"""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0.7,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager,
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager,
    )

    qa = ConversationalRetrievalChain(
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        retriever=VectorStoreRetriever(vectorstore=vectorstore),
    )
    return qa
