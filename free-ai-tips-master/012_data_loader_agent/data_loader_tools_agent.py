# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 012 | 5 Ways To Load CSV Files with AI ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Introduce you to the Data Loader Agent for automating data loading and exploration
# 3. Use the Data Loader Agent to interact with data loading tools
# 4. Load data, search for files, and explore data structures

# * Project Github: https://github.com/business-science/ai-data-science-team

# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import os
import yaml

from IPython.display import Markdown

from ai_data_science_team.agents import DataLoaderToolsAgent


# LLM SETUP
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model="gpt-4o-mini")
llm

# 1.0 CREATE THE DATA LOADER AGENT

# Make a data loader agent
data_loader_agent = DataLoaderToolsAgent(
    llm, 
    invoke_react_agent_kwargs={"recursion_limit": 10},
)

data_loader_agent

# 2.0 RUN THE AGENT (5 Examples)

# Example 1: What tools do you have access to? Return a table.
data_loader_agent.invoke_agent("What tools do you have access to? Return a table.")

data_loader_agent.get_ai_message(markdown=True)


# Example 2: What folders and files are available?
data_loader_agent.invoke_agent("What folders and files are available at the root of my directory? Return the file folder structure as code formatted block with the root path at the top and just the top-level folders and files.")

data_loader_agent.get_ai_message(markdown=True)

data_loader_agent.get_artifacts(as_dataframe=True)


# Example 3: What is in the data folder?
data_loader_agent.invoke_agent("What is in the data folder?")

data_loader_agent.get_ai_message(markdown=True)


# Example 4: Load the bike_sales_data.csv file from the data folder.
data_loader_agent.invoke_agent("Load the churn data file from the data folder.")

data_loader_agent.get_ai_message(markdown=True)

data_loader_agent.get_artifacts(as_dataframe=True)


# Example 5: Search for 'csv' files recursively in my current working directory. 
data_loader_agent.invoke_agent("Search for 'csv' files recursively in my current working directory. Return the file folder structure showing where the files are located.")

data_loader_agent.get_ai_message(markdown=True)

data_loader_agent.get_artifacts(as_dataframe=True)

# Then just point the data loader agent to the path to load the data.
data_loader_agent.invoke_agent("Load the audi csv at path 005_automate_data_wrangling/data")

data_loader_agent.get_artifacts(as_dataframe=True)


# 3.0 NEXT STEPS + PROJECT ROADMAP

# - My goal is to have a team of AI Copilots that can automate common data science tasks.
# - I'm looking for feedback on the AI Copilots. So if you try it out and if something doesn't work, please let me know.
# - I'm also looking for ideas on what other AI Copilots I can create.
# - To Send Feedback, file Github Issues here: https://github.com/business-science/ai-data-science-team/issues


# 4.0 WANT TO LEARN HOW TO USE GENERATIVE AI AND LLMS FOR DATA SCIENCE? ----
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
