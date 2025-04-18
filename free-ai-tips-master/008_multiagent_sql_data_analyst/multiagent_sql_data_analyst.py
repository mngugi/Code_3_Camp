# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 008 | Multi-Agent SQL Data Analyst ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Create an AI Copilot to automate SQL Data Analysis
# 3. Run the SQL Data Analyst Agent on the Northwind Database (Sample ERP) and ask it questions

# PROBLEM: Writing SQL code to making Data Visualizations are a time-consuming process that keeps us from making ML models and getting business insights.

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import sqlalchemy as sql
import os
import yaml
from pprint import pprint

from ai_data_science_team.multiagents import SQLDataAnalyst
from ai_data_science_team.agents import SQLDatabaseAgent, DataVisualizationAgent


# PATHS
PATH_ROOT = "008_multiagent_sql_data_analyst/"

# 1.0 DATABASE SETUP

# Create Connection

engine = sql.create_engine(f'sqlite:///data/northwind.db')

conn = engine.connect()

# 2.0 LLM SETUP 

# LLM API KEY
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")

# INSTANTIATE LLM
MODEL    = "gpt-4o-mini"

llm = ChatOpenAI(model = MODEL)


# 3.0 SQL DATA ANALYST

sql_data_analyst = SQLDataAnalyst(
    model = llm,
    sql_database_agent = SQLDatabaseAgent(
        model = llm,
        connection = conn,
        n_samples = 1,
        log = LOG,
        log_path = LOG_PATH,
        bypass_recommended_steps=True,
        bypass_explain_code=True,
    ),
    data_visualization_agent = DataVisualizationAgent(
        model = llm,
        n_samples = 10,
        log = LOG,
        log_path = LOG_PATH,
        bypass_explain_code = True,
    )
)

sql_data_analyst.show()

sql_data_analyst.show(xray=1)

sql_data_analyst.get_state_keys()



# * Easy questions: What Tables?

sql_data_analyst.invoke_agent(
    user_instructions = "What tables are in the database?",
)

sql_data_analyst.get_data_sql()

sql_data_analyst.get_sql_query_code(markdown=True)

sql_data_analyst.get_sql_database_function(markdown=True)


# * Donut Chart

sql_data_analyst.invoke_agent(
    user_instructions = "Make a donut chart of sales revenue by territory for the top 5 territories.",
)

sql_data_analyst.get_sql_query_code(markdown=True)

sql_data_analyst.get_data_sql()

sql_data_analyst.get_data_visualization_function(markdown=True)

sql_data_analyst.get_plotly_graph()


# * Time Series Plot

sql_data_analyst.invoke_agent(
    user_instructions = "Make a plot of sales revenue by month by territory. Make a dropdown for the user to select the territory.",
)

sql_data_analyst.get_sql_query_code(markdown=True)

sql_data_analyst.get_data_sql()

sql_data_analyst.get_plotly_graph()



# 4.0 NEXT STEPS

# - My goal is to have a team of AI Copilots that can automate common data science tasks.
# - I'm looking for feedback on the AI Copilots. So if you try it out and if something doesn't work, please let me know.
# - I'm also looking for ideas on what other AI Copilots I can create.
# - To Send Feedback, file Github Issues here: https://github.com/business-science/ai-data-science-team/issues


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

