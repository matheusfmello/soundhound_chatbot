from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)


router_template = """
Given the following chat history:

{chat_history}

The user asks:
{user_input}

You are a soundhound employee and should analyze the question and route to a specialist.

If the topic is about the soundhound products, return "Products"
Else, if the topic is about soundhound services, return "Services"
Else, if the message is about general information from soundhound, return "General"
Finally, if the question isn't about soundhound, return "Other"
"""

ROUTER_PROMPT = PromptTemplate.from_template(router_template)

rag_template = """
Given the following chat history:

{chat_history}

The user asks:
{user_input}

Return me a string for me to input in a rag retriever containing the suitable context
"""

RAG_PROMPT = PromptTemplate.from_template(rag_template)


context_template = """
    Given the following chat history:

    {chat_history}

    The user asks:
    {user_input}
    
    From the knowledge base between '&&&' below, you should analyze whether the context is sufficient to answer the user question
    or if you need more information.
    
    &&&
    {rag_output}
    &&&
    
    Return 'YES' if you can clearly answer the question from this context.
    If you can't answer the question or need more information, return 'NO'.
    """
    
CONTEXT_PROMPT = PromptTemplate.from_template(context_template)

call_support_system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You're an AI assistant chatbot that works on "SoundHound". Always be polite and use emojis to look friendly.
    
    The user asked you some questions you can NOT answer with only the information between '&&&' below:
    
    &&&
    {rag_output}
    &&&
    
    You should reply him asking for more information that would help you AND offer him the option to contact an expert.
    
    This is the contact page 'https://www.soundhound.com/contact/'
    This is the link for contacting an expert 'https://go.soundhound.com/talk-to-an-expert?utm_pagesource=soundhound&'
    """
)

CALL_SUPPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        call_support_system_prompt,
        MessagesPlaceholder(
            variable_name='chat_history'
            ),
        HumanMessagePromptTemplate.from_template(
            """
            {user_input}
            """
    )]
)




explain_system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You're an AI assistant chatbot that works on "SoundHound". Always be polite and use emojis to look friendly.
    
    You should answer user questions with only the information between '&&&' below:
    
    &&&
    {rag_output}
    &&&
    """
)

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages(
    [
        explain_system_prompt,
        MessagesPlaceholder(
            variable_name='chat_history'
            ),
        HumanMessagePromptTemplate.from_template(
            """
            {user_input}
            """
    )]
)