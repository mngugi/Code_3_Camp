# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 002 | CSV SEMANTIC SEARCH ----

# GOALS: 
# - Perform semantic search on a CSV file of businesses using a pre-trained sentence transformer model
# - Display the top 5 results for a given query

# Libraries:
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1.0 LOAD DATA ---- 

# Load the business data
business_data = pd.read_csv("001_csv_semantic_search/data/business_data.csv")

business_data

# 2.0 GENERATE EMBEDDINGS ----

# Load the pre-trained sentence transformer model
# - The first time this will take a few seconds to download the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the business descriptions
business_data['description_embedding'] = list(model.encode(business_data['description'].tolist()))

business_data

# 3.0 PERFORM SEMANTIC SEARCH ----

# Define a function to perform semantic search
def semantic_search(query, data, top_k=5):
    # Generate the embedding for the query
    query_embedding = model.encode([query])[0]
    
    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], data['description_embedding'].tolist())
    
    # Get the indices of the top_k most similar descriptions
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    
    # Return the top_k results
    return data.iloc[top_indices]

# Perform a semantic search
query = "Find healthy organic food businesses"
k = 5
results = semantic_search(query, business_data, top_k=k)

# Display the results
print(f"Top {k} results for your query:")
print(results[['business_name', 'description', 'category']])

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
# - Enroll here: https://learn.business-science.io/generative-ai-bootcamp-enroll

