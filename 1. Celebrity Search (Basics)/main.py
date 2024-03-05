from dotenv import load_dotenv
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Streamlit framework
st.title("Search Celebrity")
input_text = st.text_input("Search the celebrity you want...")

# Prompt Template
first_input_template = PromptTemplate(
    input_variables=['name'],
    template='Tell me about celebrity {name}'
)

second_input_template = PromptTemplate(
    input_variables=['person'],
    template='When was {person} born?'
)

third_input_template = PromptTemplate(
    input_variables=['dob'],
    template='Mention 5 major events happened around {dob} in the world'
)

# Memory
person_memory = ConversationBufferMemory(
    input_key='name',
    memory_key='chat_history'
)

dob_memory = ConversationBufferMemory(
    input_key='person',
    memory_key='chat_history'
)

events_memory = ConversationBufferMemory(
    input_key='dob',
    memory_key='events_history'
)

# OpenAI
llm = OpenAI(temperature=0.8)

# LLM Chain
chain = LLMChain(
    prompt=first_input_template,
    llm=llm,
    verbose=True,
    output_key='person',
    memory=person_memory
)

chain2 = LLMChain(
    prompt=second_input_template,
    llm=llm,
    verbose=True,
    output_key='dob',
    memory=dob_memory
)

chain3 = LLMChain(
    prompt=third_input_template,
    llm=llm,
    verbose=True,
    output_key='events',
    memory=events_memory
)

# Use SequentialChain for complex workflows with conditional logic.
# Use SimpleSequentialChain for straightforward, linear sequences.

# parent_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)
parent_chain = SequentialChain(
    chains=[chain, chain2, chain3],
    input_variables=['name'],
    output_variables=['person', 'dob', 'events'],
    verbose=True
)

if input_text:
    # For SimpleSequentialChain
    # st.write(parent_chain.run(input_text))
    
    # For SequentialChain
    st.write(parent_chain({'name': input_text}))
    
    with st.expander('Person Memory'):
        st.write(person_memory.buffer)
        
    with st.expander('DOB Memory'):
        st.write(dob_memory.buffer)
        
    with st.expander('Events Memory'):
        st.write(events_memory.buffer)