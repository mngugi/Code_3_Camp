# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 004 | How To Automate Feature Engineering With AI ----

# PROBLEM: Feature Engineering is a time-consuming process that keeps us from analyzing data, making ML models, and getting business insights.

# GOALS: 
# - Expose you to my new AI Data Science Team of Copilots
# - Create an AI Copilot to automate feature engineering
# - Use the AI Copilot to feature engineer a Customer Churn dataset

# Project Github: https://github.com/business-science/ai-data-science-team

# Installation: pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

# Libraries

from langchain_openai import ChatOpenAI

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from pprint import pprint

from ai_data_science_team.agents import make_feature_engineering_agent

# 1.0 SETUP 

# PATHS
PATH_ROOT = "004_automate_feature_engineering_copilot/"

# LLM
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

MODEL    = "gpt-4o-mini"

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")

# Data set
df = pd.read_csv("https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv")

df.info()


# 2.0 CREATE THE AI COPILOT

# Create the AI Copilot

llm = ChatOpenAI(model = MODEL)

feature_engineering_agent = make_feature_engineering_agent(
    model = llm, 
    log=LOG, 
    log_path=LOG_PATH
)

feature_engineering_agent

# Run feature engineer agent on the data

response = feature_engineering_agent.invoke({
    "target_variable": "Churn",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})


# 3.0 RESULTS

# Evaluate the response
list(response.keys())

# Print the feature engineered data
df.info()

pd.DataFrame(response['data_engineered']).info()

# What feature engineering steps were taken?

pprint(response['messages'][0].content)

# What does the data cleaner function look like?

pprint(response['feature_engineer_function'])

# How can I reuse the data cleaning steps as I get new data?

current_dir = Path.cwd() / PATH_ROOT
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from ai_functions.feature_engineer import feature_engineer

df.info()

feature_engineer(df).info()

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