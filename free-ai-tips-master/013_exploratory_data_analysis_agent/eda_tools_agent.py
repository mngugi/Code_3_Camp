# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 013 | Exploratory Data Analysis with AI: EDA Tools Agent ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Introduce you to the EDA Tools Agent for automating exploratory data analysis
# 3. Use the EDA Tools Agent to interact with EDA tools
# 4. Describe datasets, visualize missing data, and analyze correlations, and generate Sweetviz EDA reports

# * Project Github: https://github.com/business-science/ai-data-science-team

# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import os
import yaml

# Agent
from ai_data_science_team.ds_agents import EDAToolsAgent

# Helper functions
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team.utils.html import open_html_file_in_browser

# LLM SETUP
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model="gpt-4o-mini")

# DATA
df = pd.read_csv("data/churn_data.csv")

# 1.0 CREATE THE EDA TOOLS AGENT

exploratory_agent = EDAToolsAgent(
    llm, 
    invoke_react_agent_kwargs={"recursion_limit": 10},
)
exploratory_agent

# 2.0 USING THE AGENT

# Example 1: What tools do you have access to? Return a table.
exploratory_agent.invoke_agent("What tools do you have access to? Return a table.")
exploratory_agent.get_ai_message(markdown=True)

# Example 2: Give me information on the correlation funnel tool.
exploratory_agent.invoke_agent("Give me information on the correlation funnel tool.")
exploratory_agent.get_ai_message(markdown=True)

# Example 3: Describe the dataset.
exploratory_agent.invoke_agent(
    user_instructions="Describe the dataset.",
    data_raw=df,
)
exploratory_agent.get_ai_message(markdown=True)
exploratory_agent.get_artifacts().keys()
pd.DataFrame(exploratory_agent.get_artifacts()['describe_df'])

# Example 4: Visualize missing data in the dataset.
#  * Note - Requires missingno
#    pip install missingno
exploratory_agent.invoke_agent(
    user_instructions="Visualize missing data in the dataset.",
    data_raw=df,
)
exploratory_agent.get_ai_message(markdown=True)

exploratory_agent.get_artifacts().keys()

matplotlib_from_base64(exploratory_agent.get_artifacts()['matrix_plot'])

matplotlib_from_base64(exploratory_agent.get_artifacts()['bar_plot'])

matplotlib_from_base64(exploratory_agent.get_artifacts()['heatmap_plot'])

# Example 5: Use the correlation funnel tool to analyze the dataset. Use the Churn feature as the target.
#  * Note - Requires pytimetk
#    pip install pytimetk
exploratory_agent.invoke_agent(
    user_instructions="Use the correlation funnel tool to analyze the dataset. Use the Churn feature as the target.",
    data_raw=df,
)
exploratory_agent.get_ai_message(markdown=True)

exploratory_agent.get_artifacts(as_dataframe=False).keys()

pd.DataFrame(exploratory_agent.get_artifacts(as_dataframe=False)['correlation_data'])

matplotlib_from_base64(exploratory_agent.get_artifacts(as_dataframe=False)['plot_image'])

plotly_from_dict(exploratory_agent.get_artifacts(as_dataframe=False)['plotly_figure'])

# Example 6: Generate a Sweetviz report for the dataset. Use the Churn feature as the target.
#  * Note - Requires sweetviz. 
#    pip install sweetviz
exploratory_agent.invoke_agent(
    user_instructions="Generate a Sweetviz report for the dataset. Use the Churn feature as the target.",
    data_raw=df,
)
exploratory_agent.get_ai_message(markdown=True)

exploratory_agent.get_internal_messages()

exploratory_agent.get_artifacts(as_dataframe=False).keys()

open_html_file_in_browser(
    file_path=exploratory_agent.get_artifacts(as_dataframe=False)['report_file']
)

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
