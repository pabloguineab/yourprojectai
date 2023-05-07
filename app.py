# Bring in deps
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]

# App framework
st.title('🦜🔗 Final undergraduate Project Generator')

# User inputs
prompt_title = st.text_input('Enter the project title:')
num_sections = st.number_input('Enter the number of sections in the index:', min_value=1, value=3, step=1)

sections = []
for i in range(num_sections):
    section = st.text_input(f'Enter section {i+1} of the index:')
    sections.append(section)

# Check if both inputs are provided
if prompt_title and sections:
    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me a project degree title about {topic}'
    )
    index_template = PromptTemplate(
        input_variables=['sections'], 
        template='write me an extended index project degree with the following sections: {sections}'
    )
    script_template = PromptTemplate(
        input_variables=['title', 'index', 'wikipedia_research'], 
        template='write me an introduction section for my project based on this title TITLE: {title} and this index: {index} while leveraging this wikipedia research: {wikipedia_research}'
    )
    
    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    index_memory = ConversationBufferMemory(input_key='sections', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    
    # Llms
    llm = OpenAI(temperature=0.9) 
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    index_chain = LLMChain(llm=llm, prompt=index_template, verbose=True, output_key='index', memory=index_memory)
    script_chain = LLMChain(
        llm=llm,
        prompt=script_template,
        verbose=True,
        output_key='script',
        memory=script_memory
    )
    
    wiki = WikipediaAPIWrapper()
    
    # Generate project based on user inputs
    title = title_chain.run(topic=prompt_title)
    index = index_chain.run(sections=[str(section) for section in sections])    
    wiki_research = wiki.run(prompt_title)  
    script = script_chain.run(title=title, index=index, wikipedia_research=wiki_research)
    
    # Display project details
    st.write("Generated Project:")
    st.write("Title:", title)
    st.write("Index:", index)
    st.write("Script:", script)
    
    # Display history
    with st.expander('Title History'): 
        st.info(title_memory.buffer)
   
    with st.expander('Index History'): 
        st.info(index_memory.buffer)
        
    with st.expander('Introduction History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
