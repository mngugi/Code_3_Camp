# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 006 | How To Automate SQL With AI ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Create an AI Copilot to automate SQL Database Queries
# 3. Run the SQL Agent on the Northwind Database (Sample ERP) and ask it questions

# PROBLEM: SQL is a time-consuming process that keeps us from analyzing data, making ML models, and getting business insights.

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import sqlalchemy as sql
import os
import yaml
from pprint import pprint

from ai_data_science_team.agents import make_sql_database_agent

# PATHS
PATH_ROOT = "006_automate_sql_copilot/"

# Create Connection

engine = sql.create_engine(f'sqlite:///data/northwind.db')

conn = engine.connect()

# List all tables in the database

pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)

# 1.0 AGENT SETUP 

# LLM
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL    = "gpt-4o-mini"

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")


# 2.0 CREATE THE SQL AGENT

llm = ChatOpenAI(model = MODEL)

sql_agent = make_sql_database_agent(
    model = llm, 
    connection=conn, 
    n_samples=1, # Needed for large databases to avoid token limits
    log=LOG, 
    log_path=LOG_PATH,
    bypass_explain_code=True,
    bypass_recommended_steps=True,
)

sql_agent

# 3.0 RUN THE SQL AGENT

# What tables are in the database?

response = sql_agent.invoke({
    "user_instructions": "What tables are in the database?",
    "max_retries":3, 
    "retry_count":0
})

response.keys()

pd.DataFrame(response['data_sql'])

response['sql_query_code']


# What are the sales for each product?

response = sql_agent.invoke({
    "user_instructions": "What are the sales for each product?",
    "max_retries":3, 
    "retry_count":0
})

pd.DataFrame(response['data_sql'])

pprint(response['sql_query_code'])

# What are the top sales for each product **by year**?

response = sql_agent.invoke({
    "user_instructions": "What are the sales for each product by year?",
    "max_retries":3, 
    "retry_count":0
})

pd.DataFrame(response['data_sql'])

pprint(response['sql_query_code'])

# 5.0 NEXT STEPS

# - My goal is to have a team of AI Copilots that can automate common data science tasks.
# - I'm looking for feedback on the AI Copilots. So if you try it out and if something doesn't work, please let me know.
# - I'm also looking for ideas on what other AI Copilots I can create.
# - To Send Feedback, file Github Issues here: https://github.com/business-science/ai-data-science-team/issues


# 6.0 WANT TO LEARN HOW TO USE GENERATIVE AI AND LLMS FOR DATA SCIENCE? ----
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