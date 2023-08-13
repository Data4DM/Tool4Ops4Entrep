from langchain.agents.agent_toolkits.amadeus.toolkit import AmadeusToolkit

from langchain.memory import ConversationBufferMemory, ConversationEntityMemory
from langchain.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool

load_dotenv('.env')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")    
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from main import create_agent_executor

llm = OpenAI(temperature=0, verbose=True, openai_api_key=OPENAI_API_KEY)

search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
db_chain = create_agent_executor()

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="Lobbying DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about lobbying industry."
    )
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm_chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

agent_chain = initialize_agent(tools, 
                               llm_chat, 
                               agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                               verbose=True, 
                               memory=memory)


agent_chain.run(input="Which lobbying firm would be the best for Novo Nordisk to hire? Novo Nordisk is the drugmaker that produces weight loss medications Ozempic and Wegovy. The company wants to make the drugs to be eligible for coverage of Medicare bills. Please provide your list of good candidates of lobbying firms with the reason. The reason is good when it's backed with data.")

agent_chain.run(input="Can you find the contact of that lobbying firm you just recommneded?")

print(memory.load_memory_variables({}))

