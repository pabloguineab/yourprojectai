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
st.title('🦜🔗 Final Project Degree Generator')

# User inputs
prompt_title = st.text_input('Enter the project title:')
prompt_index = st.text_input('Enter the project index:')

# Check if both inputs are provided
if prompt_title and prompt_index:
    # Prompt templates
    title_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me a project degree title about {topic}'
    )
    index_template = PromptTemplate(
        input_variables=['topic'], 
        template='write me an extended index project degree about {topic}'
    )
    script_template = PromptTemplate(
        input_variables=['title', 'index', 'wikipedia_research'], 
        template='write me an introduction section for my project based on this title TITLE: {title} and this index: {index} while leveraging this wikipedia research: {wikipedia_research}'
    )
    section_template = PromptTemplate(
        input_variables=['title', 'index', 'wikipedia_research', 'previous_sections'],
        template='write me a section for my project with the following title: {title}, index: {index}, wikipedia research: {wikipedia_research}, and the previous sections: {previous_sections}'
    )
    
    # Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    index_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    section_memory = ConversationBufferMemory(input_key='previous_sections', memory_key='chat_history')
    
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
    section_chain = LLMChain(
        llm=llm,
        prompt=section_template,
        verbose=True,
        output_key='section',
        memory=section_memory
    )
    
    wiki = WikipediaAPIWrapper()
    
    # Generate project based on user inputs
    title = title_chain.run(topic=prompt_title)
    index = index_chain.run(topic=prompt_index)
    wiki_research = wiki.run(prompt_title)
    section_outputs = []
    section_prompts = [s.strip() for s in prompt_index.split('\n')]
    section_prompts = [s for s in section_prompts if len(s) > 0]
    previous_sections = ''
    for i, section_prompt in enumerate(section_prompts):
        section_title, section_index = section_prompt.split('-', 1)
        section_title = section_title.strip()
        section_index = section_index.strip()
        section_output = section_chain.run(title=section_title, index=section_index, wikipedia_research=wiki_research, previous_sections=previous_sections)
        section_outputs.append(section_output)
        previous_sections += section_output + '\n'

        # Output generated project
        st.subheader('Project Degree')
        st.write(f"# {title}")
        st.write(index)
        st.write(script_chain.run(title=title, index=index, wikipedia_research=wiki_research, previous_sections=previous_sections))
        for i, section_output in enumerate(section_outputs):
            st.write(f"## {section_prompts[i].split('-')[0].strip()}")
            st.write(section_output)
