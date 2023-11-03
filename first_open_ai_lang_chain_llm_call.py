# Integrate our code with OpenAI API
import os
import streamlit as st
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = openai_key

# streamlit framework
st.title('Langchain Demo with OpenAI API')
input_text = st.text_input("Search the topic you are looking for:")

person_memory = ConversationBufferMemory(input_key = 'object', memory_key = 'chat_history')
country_memory = ConversationBufferMemory(input_key = 'person_details', memory_key = 'chat_history')
places_memory = ConversationBufferMemory(input_key = 'country', memory_key = 'chat_history')

first_template = PromptTemplate(
    input_variables = ['object'],
    template = "Describe more about {object}"
)

## OpenAI LLMs
llm = OpenAI(temperature=0.8)
chain1 = LLMChain(llm = llm, prompt = first_template, verbose = True, output_key='person_details', memory = person_memory)

second_template = PromptTemplate(
    input_variables = ['person_details'],
    template = "Which country the person {person_details} was born?"
)

chain2 = LLMChain(llm = llm, prompt = second_template, verbose = True, output_key='country', memory = country_memory)

third_template = PromptTemplate(
    input_variables = ['country'],
    template = "Tell me 5 famous things of country {country}"
)

chain3 = LLMChain(llm = llm, prompt = third_template, verbose = True, output_key='things', memory=places_memory)

parent_chain = SequentialChain(chains = [chain1, chain2, chain3], input_variables = ['object'], output_variables = ['person_details', 'country', 'things'], verbose = True)

if input_text:
    st.write(parent_chain({'object': input_text}))

    with st.expander('Person'):
        st.info(person_memory.buffer)
    
    with st.expander('Country'):
        st.info(country_memory.buffer)

    with st.expander('Places'):
        st.info(places_memory.buffer)