# Bring in deps
import os 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = st.secrets["apikey"]

# App framework
st.title('🦜🔗 Final undergraduate Project Generator')

# User inputs
prompt_title = st.text_input('Enter the project title:')
num_sections = st.number_input('Enter the number of sections in the index:', min_value=1, value=3, step=1)

sections = []
section_templates = []

for i in range(num_sections):
    section_title = st.text_input(f'Enter title for section {i+1}:')
    section_index = st.text_input(f'Enter index for section {i+1}:')
    section_template = PromptTemplate(
        input_variables=['title', 'index', 'wiki_research'],
        template='write me a section for my project with title: {title}, index: {index}, and leveraging this wikipedia research: {wikipedia_research}'
    )
    sections.append((section_title, section_index))
    section_templates.append(section_template)

# Check if both inputs are provided
if prompt_title and all(sections):
    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me a project degree title about {topic}'
    )
    index_template = PromptTemplate(
        input_variables=['sections'], 
        template='write me an extended index project degree with the following sections: {sections}'
    )
    
    script_templates = []
    for i, (section_title, section_index) in enumerate(sections):
        script_template = PromptTemplate(
            input_variables=['title', 'index', 'wikipedia_research'], 
            template=f'write me section {i+1} with title: {section_title}, index: {section_index}, and leveraging this wikipedia research: {wikipedia_research}'
        )
        script_templates.append(script_template)
    
    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    index_memory = ConversationBufferMemory(input_key='sections', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    
    # Llms
    llm = OpenAI(temperature=0.9) 
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    index_chain = LLMChain(llm=llm, prompt=index_template, verbose=True, output_key='index', memory=index_memory)
    script_chains = []
    for i, script_template in enumerate(script_templates):
        script_chain = LLMChain(
            llm=llm,
            prompt=script_template,
            verbose=True,
            output_key=f'script{i+1}',
            memory=script_memory
        )
        script_chains.append(script_chain)
    
    wiki = WikipediaAPIWrapper()
    
    # Generate project based on user inputs
    title = title_chain.run(topic=prompt_title)
    index = index_chain.run(sections=[(str(section[0]), str(section[1])) for section in sections])
    wiki_research = wiki.run(prompt_title)
    scripts = [script_chain.run(title=title, index=index, wikipedia_research=wiki_research) for script_chain, (section_title, section_index) in zip(script_chains, sections)]
    
    # Display project details
    st.write("Generated Project:")
    st.write("Title:", title)
    st.write("Index:", index)
    
    for i, (section_title, section_index) in enumerate(sections):
        with st.expander(f'Section {i+1} - {section_title}'):
            st.write("Title:", section_title)
            st.write("Index:", section_index)
            st.write("Script:", scripts[i])
    
    # Display history
    with st.expander('Title History'): 
        st.info(title_memory.buffer)
   
    with st.expander('Index History'): 
        st.info(index_memory.buffer)
        
    with st.expander('Introduction History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)
