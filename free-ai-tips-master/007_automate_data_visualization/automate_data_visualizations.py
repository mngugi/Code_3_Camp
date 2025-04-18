# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 007 | How To Automate Data Visualizations With AI ----

# WHAT WE COVER TODAY: 
# 1. Expose you to my new AI Data Science Team of Copilots
# 2. Create an AI Copilot to automate Data Visualizations
# 3. Run the SQL Agent on the Northwind Database (Sample ERP) and ask it questions

# PROBLEM: Writing code to make Data Visualizations are a time-consuming process that keeps us from making ML models and getting business insights.

# * Project Github: https://github.com/business-science/ai-data-science-team


# LIBRARIES
# * pip install git+https://github.com/business-science/ai-data-science-team.git --upgrade

from langchain_openai import ChatOpenAI
import pandas as pd
import sqlalchemy as sql
import os
import yaml
from pprint import pprint

from ai_data_science_team.agents import make_data_visualization_agent
from ai_data_science_team.utils.plotly import plotly_from_dict

# PATHS
PATH_ROOT = "007_automate_data_visualization/"

# 1.0 DATABASE SETUP

# Create Connection

engine = sql.create_engine(f'sqlite:///data/northwind.db')

conn = engine.connect()

# Query the database

def sql_database_pipeline(connection):
    import pandas as pd
    import sqlalchemy as sql
    
    # Create a connection if needed
    is_engine = isinstance(connection, sql.engine.base.Engine)
    conn = connection.connect() if is_engine else connection

    sql_query = '''
    SELECT strftime('%Y-%m', OrderDate) AS Month, SUM(OrderDetails.UnitPrice * OrderDetails.Quantity * (1 - OrderDetails.Discount)) AS TotalSales
FROM Orders
JOIN "Order Details" AS OrderDetails ON Orders.OrderID = OrderDetails.OrderID
GROUP BY Month
ORDER BY Month;
    '''
    
    return pd.read_sql(sql_query, connection)

df = sql_database_pipeline(conn)
df


# 2.0 LLM SETUP 

# LLM API KEY
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
# os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

# LOGGING
LOG      = True
LOG_PATH = os.path.join(os.getcwd(), PATH_ROOT, "ai_functions/")

# INSTANTIATE LLM
MODEL    = "gpt-4o-mini"

llm = ChatOpenAI(model = MODEL)


# 3.0 CREATE THE DATA VISUALIZATION AGENT
#   user_instructions: Please plot the time series.

data_visualization_agent = make_data_visualization_agent(
    model=llm, 
    n_samples=10,
    log=LOG,
    log_path=LOG_PATH,
    bypass_explain_code=True,
)

data_visualization_agent


# 4.0 RUN THE DATA VISUALIZATION AGENT

response = data_visualization_agent.invoke({
    "user_instructions": "Please plot the time series.",
    "data_raw": df.to_dict(),
    "max_retries":3, 
    "retry_count":0
})

list(response.keys())


response['plotly_graph']

plotly_from_dict(response['plotly_graph'])



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
