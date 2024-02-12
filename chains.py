from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.output_parsers.boolean import BooleanOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prompts import ROUTER_PROMPT, RAG_PROMPT, CONTEXT_PROMPT, CALL_SUPPORT_PROMPT, EXPLAIN_PROMPT

load_dotenv()

llm = ChatOpenAI(temperature=0.0)

# RAG

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

db_general = FAISS.load_local('faiss_general', embeddings=embeddings)
db_products = FAISS.load_local('faiss_products', embeddings=embeddings)
db_services = FAISS.load_local('faiss_services', embeddings=embeddings)

general_retriever = db_general.as_retriever(search_type='mmr')
products_retriever = db_products.as_retriever(search_type='mmr')
services_retriever = db_services.as_retriever(search_type='mmr')

def _combine_documents(retrieved_docs: list) -> str:
    docs = [doc.page_content for doc in retrieved_docs]
    return '\n'.join(docs)


### ROUTER

router_chain = (
    ROUTER_PROMPT
    | llm
    | StrOutputParser()
)

### RAG Retrievers

general_chain = (
    RAG_PROMPT
    | llm
    | StrOutputParser()
    | general_retriever
    | _combine_documents
)

products_chain = (
    RAG_PROMPT
    | llm
    | StrOutputParser()
    | products_retriever
    | _combine_documents
)

services_chain = (
    RAG_PROMPT
    | llm
    | StrOutputParser()
    | services_retriever
    | _combine_documents
)


rag_chain = RunnableBranch(
    (lambda x: "products" in x['topic'].lower(), products_chain),
    (lambda x: "services" in x['topic'].lower(), services_chain),
    general_chain
)

### CONTEXT ANALYZER

context_chain = (
    CONTEXT_PROMPT
    | llm
    | BooleanOutputParser()
)

### FINAL CHAIN


call_support_chain = (
    CALL_SUPPORT_PROMPT
    | llm
    | StrOutputParser()
)


explain_chain = (
    EXPLAIN_PROMPT
    | llm
    | StrOutputParser()
)

chatbot_chain = RunnableBranch(
    (lambda x: x['context']==False, call_support_chain),
    explain_chain
)
    

def call_chain(user_input: str, user_id: str):
    
    ts_request = datetime.now().replace(microsecond=0)
    
    global router_chain, rag_chain, context_chain, chatbot_chain
    
    history_db = pd.read_csv('chat_history.csv')
    
    print(user_id)
    
    print(history_db[history_db['user_id']==user_id])
    
    user_history = history_db[history_db['user_id']==user_id]
    
    ### Build memory
    
    def memory_factory(row, chat_history: ChatMessageHistory):
        
        if row['author'] == 'human':
            chat_history.add_user_message(row['content'])
            
        elif row['author'] == 'ai':
            chat_history.add_ai_message(row['content'])
            
            
    
    def build_memory(user_history: pd.DataFrame) -> ChatMessageHistory:
        
        chat_history = ChatMessageHistory()
        
        if user_history.empty:
            return chat_history
        
        user_history = user_history.sort_values('timestamp')
        
        user_history.apply(lambda row: memory_factory(row, chat_history), axis=1)
        
        return chat_history
        
        
    chat_history = build_memory(user_history=user_history)
    
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        memory_key='chat_history',
        input_key='user_input',
        return_messages=True
    )
    
    inputs = {
        'user_input': user_input,
        'chat_history': memory.buffer 
    }
    
    inputs['topic'] = router_chain.invoke(inputs)
    inputs['rag_output'] = rag_chain.invoke(inputs)
    inputs['context'] = context_chain.invoke(inputs)
    
    output = chatbot_chain.invoke(inputs)
    
    ts_response = datetime.now().replace(microsecond=0)
    
    new_memory = pd.DataFrame(
        {
            'timestamp': [ts_request, ts_response],
            'user_id': [user_id, user_id],
            'author': ['human', 'ai'],
            'content': [user_input, output]
        } 
    )
    
    history_db = pd.concat([history_db, new_memory]).reset_index(drop=True)
    history_db.to_csv('chat_history.csv', index=False)
    
    return output