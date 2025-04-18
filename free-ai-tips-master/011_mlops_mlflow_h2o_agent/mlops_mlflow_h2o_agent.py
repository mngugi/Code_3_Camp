# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 010 | MLOps with AI: H2O + MLflow Agent ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Introduce an AI ML Agent for automating 32+ Machine Learning Models in 30 seconds
# 3. Combine the H2O ML Agent with MLflow for MLOps
# 4. Use MLflow to make predictions in production and manage ML projects (experiments, runs, and artifacts)

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import h2o 
import mlflow
import os
import yaml

from ai_data_science_team.ml_agents import H2OMLAgent, MLflowToolsAgent

# DATA
df = pd.read_csv("data/churn_data.csv")
df

# LLM SETUP
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# Define constants for model
MODEL = "gpt-4o-mini"

# Initialize the language model
llm = ChatOpenAI(model=MODEL)
llm

# 1.0 CREATE THE MACHINE LEARNING AGENT
ml_agent = H2OMLAgent(
    model=llm, 
    enable_mlflow=True, # Use this to log to MLflow 
)
ml_agent


# RUN THE AGENT
ml_agent.invoke_agent(
    data_raw=df.drop(columns=["customerID"]),
    user_instructions="Please do classification on 'Churn'. Use a max runtime of 30 seconds. Use mlflow to log the experiment.",
    target_variable="Churn"
)

ml_agent.get_leaderboard()

# 2.0 CREATE MLFLOW AGENT
mlflow_agent = MLflowToolsAgent(llm)
mlflow_agent

# what tools do you have access to?
mlflow_agent.invoke_agent(
    user_instructions="What tools do you have access to?",
)
mlflow_agent.get_ai_message(markdown=True)

# launch the mflow UI
mlflow_agent.invoke_agent(user_instructions="launch the mflow UI")
mlflow_agent.get_ai_message(markdown=True)

# what runs are available?
mlflow_agent.invoke_agent(user_instructions="What runs are available in the H2O AutoML experiment?")
mlflow_agent.get_ai_message(markdown=True)
mlflow_agent.get_mlflow_artifacts(as_dataframe=True)

# Make predictions using a specific run ID
mlflow_agent.invoke_agent(
    user_instructions="Make churn predictions on the data set provided using Run ID b19bb206a13644748bb601de3b7b34d5.",
    data_raw=df, # Provide the raw data to the agent for predictions
)
mlflow_agent.get_mlflow_artifacts(as_dataframe=True)

# shut down the mflow UI
mlflow_agent.invoke_agent("shut down the mflow UI on port 5001")
mlflow_agent.get_ai_message(markdown=True)


# 5.0 NEXT STEPS + PROJECT ROADMAP

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

