# BUSINESS SCIENCE GENERATIVE AI/ML TIPS ----
# AI-TIP 002 | AI/ML FOR CUSTOMER CHURN ----

# GOALS: 
# - Use Generative AI to improve feature engineering of Customer Churn Models 
# - Use LLMs to generate summaries of customer tickets
# - Use Text Embeddings to convert text to vectors
# - Use XGBoost to predict customer churn with AI features

# Libraries

from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd
import numpy as np
import yaml
import os
import ast
import matplotlib.pyplot as plt

# ---------------------------
# 1. Setup
# ---------------------------

# PATHS
PATH_ROOT = "002_customer_churn_ai_ml/"

# MODELS
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# OPENAI SETUP
os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
os.environ['OPENAI_API_KEY'] = yaml.safe_load(open('../credentials.yml'))['openai']

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# DATASET 
df = pd.read_csv(PATH_ROOT + "/data/customer_churn.csv")

# ---------------------------
# 2. Generate Summaries with an LLM
# ---------------------------

# This will take a few minutes and will incur charges to your OpenAI account

def summarize_ticket(ticket_text):
    prompt = f"Summarize the following customer ticket focusing on the main complaint or request:\n\n{ticket_text}\n\nSummary:"
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    # Use message.content for chat responses
    return response.choices[0].message.content.strip()

df['ticket_summary'] = df['ticket_notes'].apply(summarize_ticket)

# df.to_csv(PATH_ROOT + "/data/customer_churn_summary.csv", index=False)

df = pd.read_csv( PATH_ROOT + "/data/customer_churn_summary.csv")
df

# ---------------------------
# 3. Get Embeddings for Summaries
# ---------------------------

# This will take a few minutes and will incur charges to your OpenAI account

def get_embeddings(text):
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    # Access data as attributes instead of dict indexing:
    embedding = response.data[0].embedding
    return embedding

df['summary_embedding'] = df['ticket_summary'].apply(get_embeddings)

# df.to_csv(PATH_ROOT + "/data/customer_churn_summary_embeddings.csv", index=False)

df = pd.read_csv(PATH_ROOT + "/data/customer_churn_summary_embeddings.csv")

df

# ---------------------------
# 4. Prepare Features
# ---------------------------
df['plan_type_encoded'] = LabelEncoder().fit_transform(df['plan_type'])

# If embeddings are stored in a pandas Series
df['summary_embedding'] = df['summary_embedding'].apply(ast.literal_eval)

embeddings_df = pd.DataFrame(df['summary_embedding'].tolist(), index=df.index)

# Combine your original numeric features with the embedding columns
X_df = pd.concat([df[['age','tenure','spend_rate','plan_type_encoded']], embeddings_df], axis=1)

# Ensure all columns are numeric
X_df = X_df.astype('float32')

y = df['churn'].values

# You can now split into train and test sets while keeping X as a DataFrame
X_train, X_test, y_train, y_test = train_test_split(X_df, y, stratify=y, random_state=42)


# ---------------------------
# 5. Train an XGBoost Model
# ---------------------------
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred_proba))

# X_test is now still a DataFrame
print("X_test (DataFrame) head:")
print(X_test.head())

# ---------------------------
# 6. Interpret the ML Model
# ---------------------------

# Convert model to xgboost Booster
booster = model.get_booster()

# Plot feature importance
xgb.plot_importance(booster)
plt.show()

# If you want a numeric array of feature importances:
importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
importance_df.sort_values('importance', ascending=False, inplace=True)
print(importance_df)

X_df[422]

df_top_feat_importance = df.copy()

df_top_feat_importance[422] = X_df[422]

df_top_feat_importance.sort_values(by=422, ascending=False).head(10)


