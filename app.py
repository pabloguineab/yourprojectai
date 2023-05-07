# Bring in deps
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] =  st.secrets["apikey"]

# App framework
st.title('🦜🔗 Final undergraduate Project Generator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a project degree title about {topic}'
)
index_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me an extended index project degree  about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'index', 'wikipedia_research'], 
    template='write me an introduction section for my project based on this title TITLE: {title} and this index: {index} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
index_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
index_chain = LLMChain(llm=llm, prompt=index_template, verbose=True, output_key='index', memory=index_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    index = index_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, index=index, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(index)
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)
   
    with st.expander('Index History'): 
        st.info(index_memory.buffer)
        
    with st.expander('Introduction History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
