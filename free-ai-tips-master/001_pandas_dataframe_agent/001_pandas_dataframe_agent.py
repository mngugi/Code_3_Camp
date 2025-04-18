# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 001 | HOW TO CREATE YOUR FIRST DATA ANALYSIS AGENT ----

# GOALS: 
# - Create a data analysis agent that can answer questions about a dataset
# - Use OpenAI's API to interact with the data

# Libraries:
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

import os
from pathlib import Path
import sys
import yaml
from pprint import pprint
import pandas as pd


# SETUP PATHS

PATH_ROOT = '001_pandas_dataframe_agent'

# Add the directory to the system path if not already present
current_dir = Path.cwd() / PATH_ROOT
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))
    
# Import from utils.parsers module
from utils.parsers import parse_json_to_dataframe

# 1.0 SET UP OPENAI API ----

# OPENAI SETUP
model = 'gpt-4o-mini'

# Set your OpenAI API key
# os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('../credentials.yml'))['openai']

# 2.0 CREATE THE DATA ANALYSIS AGENT ----
#  - Load a CSV file of your data
#  - Create an agent that can answer questions about the data
#  - Set the return format to JSON
#  - Run the agent
#  - Parse the JSON

# Load your dataset
df = pd.read_csv(PATH_ROOT + '/data/customer_data.csv')
df

# Initialize the LLM
llm = ChatOpenAI(
    model_name=model,
    temperature=0
)

# Create an agent that can interact with the Pandas DataFrame
data_analysis_agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    agent_type=AgentType.OPENAI_FUNCTIONS,
    suffix = "Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
    verbose=True,
    allow_dangerous_code=True
)
data_analysis_agent

# Run the agent
response = data_analysis_agent.invoke("What are the total sales by geography?")
response

pprint(response['output'])

# Parse the JSON

response_df = parse_json_to_dataframe(response['output'])

# Make a visualization

import plotly.express as px

fig = px.bar(response_df, x='Geography', y='Sales', title='Sales by Geography')
fig


# 5.0 WANT TO LEARN HOW TO USE GENERATIVE AI AND LLMS FOR DATA SCIENCE? ----
# - Join My Live 8-Week AI For Data Scientists Bootcamp
# - Live Cohorts are happening once per quarter. Schedule:
#       -   Week 1: Live Kickoff Clinic + Local LLM Training + AI Fast Track
#       -   Week 2: Retrieval Augmented Generation (RAG) For Data Scientists
#       -   Week 3: Business Intelligence AI Copilot (SQL + Pandas Tools)
#       -   Week 4: Customer Analytics Agent Team (Multi-Agent Workflows)
#       -   Week 5: Time Series Forecasting Agent Team (Multi-Agent Machine Learning Workflows)
#       -   Week 6: LLM Model Deployment With AWS Bedrock
#       -   Week 7: Fine-Tuning LLM Models & RAG Deployments With AWS Bedrock
#       -   Week 8: AI App Deployment With AWS Cloud (Docker, EC2, NGINX)
# 
# Enroll here: https://learn.business-science.io/generative-ai-bootcamp-enroll
