import os
import json
import time
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Required for server-side plotting (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

# --- CONFIGURATION ---
app = Flask(__name__)
FEATHER_FILE = 'data.feather'
STATIC_FOLDER = 'static/plots'
os.makedirs(STATIC_FOLDER, exist_ok=True)

# API Configuration
genai.configure(api_key="YOUR_GEMINI_API_KEY") 
model = genai.GenerativeModel('gemini-1.5-flash')

# --- DATA LOADING ---
def get_dataframe():
    try:
        # Utilizing pandas 1.4.4 & pyarrow 12.0.1
        return pd.read_feather(FEATHER_FILE)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return None

# --- VISUALIZATION ENGINE ---
def create_chart(chart_type, x_col, y_col, df):
    """
    Generates a matplotlib chart based on AI parameters.
    """
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid") # Uses seaborn 0.11.2 styling
    
    filename = f"plot_{uuid.uuid4().hex}.png"
    filepath = os.path.join(STATIC_FOLDER, filename)

    try:
        if chart_type == 'bar':
            # Aggregate if too many unique values
            if df[x_col].nunique() > 20:
                data = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(15)
                sns.barplot(data=data, x=x_col, y=y_col, palette="viridis")
            else:
                sns.barplot(data=df, x=x_col, y=y_col, palette="viridis")
                
        elif chart_type == 'line':
            sns.lineplot(data=df, x=x_col, y=y_col, marker='o')
            
        elif chart_type == 'scatter':
            sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.7)
            
        elif chart_type == 'histogram':
            sns.histplot(data=df, x=x_col, kde=True)
            
        plt.title(f"{chart_type.title()} Chart of {y_col} vs {x_col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        return f"/{STATIC_FOLDER}/{filename}"
        
    except Exception as e:
        print(f"Plotting Error: {e}")
        plt.close()
        return None

# --- AI LOGIC ---
def process_query(user_query):
    df = get_dataframe()
    if df is None: return {"text": "Error: Data source unavailable."}

    # 1. Ask Gemini to decide: Text Answer OR Visualization?
    # We ask for a structured JSON response to parse intent safely.
    columns = list(df.columns)
    stats = df.describe().to_string()
    
    prompt = f"""
    You are a Data Assistant. Use the dataset metadata below.
    
    COLUMNS: {columns}
    STATS: {stats}
    
    USER QUERY: "{user_query}"
    
    TASK:
    Determine if the user wants a visualization (chart/plot) or a text answer.
    
    RESPONSE FORMAT (Strict JSON):
    If Visualization:
    {{
        "intent": "plot",
        "chart_type": "bar" | "line" | "scatter" | "histogram",
        "x_axis": "exact_column_name_from_list",
        "y_axis": "exact_column_name_from_list" (or null for histogram),
        "explanation": "Brief description of the chart"
    }}
    
    If Text Answer:
    {{
        "intent": "text",
        "response": "Your detailed answer here based on the stats..."
    }}
    """
    
    try:
        ai_resp = model.generate_content(prompt)
        # Clean the response to ensure valid JSON (remove markdown backticks if any)
        clean_json = ai_resp.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        
        if data['intent'] == 'plot':
            # Generate the plot
            image_url = create_chart(data['chart_type'], data['x_axis'], data['y_axis'], df)
            if image_url:
                return {
                    "text": data.get('explanation', "Here is the visualization you requested."),
                    "image": image_url
                }
            else:
                return {"text": "I tried to generate a chart, but encountered a technical issue with the data format."}
        else:
            # Return text response
            return {"text": data['response']}
            
    except Exception as e:
        print(f"AI Processing Error: {e}")
        return {"text": "I'm having trouble analyzing that request right now."}

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_msg = request.json.get("message")
    if not user_msg: return jsonify({"error": "Empty message"}), 400
    
    response_data = process_query(user_msg)
    return jsonify(response_data)

if __name__ == '__main__':
    from waitress import serve
    print("AI Data Dashboard running on http://localhost:8080")
    serve(app, host='0.0.0.0', port=8080)