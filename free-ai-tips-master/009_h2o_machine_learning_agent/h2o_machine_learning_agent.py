# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 009 | H2O Machine Learning Agent ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Introduce an AI ML Agent for automating 32+ Machine Learning Models in 30 seconds
# 3. Execute the H2O ML SQL Data Analyst Agent on the Customer Churn Dataset

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import h2o # pip install h2o
import os
import yaml

from ai_data_science_team.ml_agents import H2OMLAgent

# PATHS
PATH_ROOT = "009_h2o_machine_learning_agent/"

# DATA
df = pd.read_csv("data/churn_data.csv")

# 1.0 LLM SETUP 

# LLM API KEY
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# Define constants for model, logging, and paths
MODEL    = "gpt-4o-mini"
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")
MODEL_PATH = os.path.join(os.getcwd(), PATH_ROOT, "h2o_models/")

# Initialize the language model
llm = ChatOpenAI(model=MODEL)
llm

# 2.0 CREATE THE AGENT

ml_agent = H2OMLAgent(
    model=llm, 
    log=True, 
    log_path=LOG_PATH,
    model_directory=MODEL_PATH, 
)
ml_agent

# 3.0 RUN THE AGENT

ml_agent.invoke_agent(
    data_raw=df.drop(columns=["customerID"]),
    user_instructions="Please do classification on 'Churn'. Use a max runtime of 30 seconds.",
    target_variable="Churn"
)

# Retrieve and display the leaderboard of models
ml_agent.get_leaderboard()

# Get the H2O training function in markdown format
ml_agent.get_h2o_train_function(markdown=True)

# Get the recommended machine learning steps in markdown format
ml_agent.get_recommended_ml_steps(markdown=True)

# Get a summary of the workflow in markdown format
ml_agent.get_workflow_summary(markdown=True)

# Get a summary of the logs in markdown format
ml_agent.get_log_summary(markdown=True)

# Get the path to the saved model
model_path = ml_agent.get_model_path()
model_path

# 4.0 LOAD THE MODEL
# Initialize H2O and load the saved model

h2o.init()

model = h2o.load_model(model_path)
model

# Evaluate the model's performance
model.model_performance()

# Make predictions using the loaded model
model.predict(h2o.H2OFrame(df))

# Generate explanations for the model's predictions
expl = model.explain(h2o.H2OFrame(df), )
expl

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

