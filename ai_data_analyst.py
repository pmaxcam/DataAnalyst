import os
import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
import re
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to set API key if not already set
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    # Try to get from Streamlit secrets
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = openai_key
    except:
        st.error("Please set your OpenAI API key in the app settings")
        st.stop()

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None, None

        # Create temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            
        # Generate column metadata for the semantic model
        column_metadata = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            # Convert sample values to strings to handle timestamps and other non-JSON serializable types
            sample_values = df[col].dropna().unique()[:5]
            sample_values = [str(val) for val in sample_values]  # Convert all values to strings
            
            metadata = {
                "name": col,
                "type": col_type,
                "sample_values": sample_values,
                "description": f"Contains {col_type} values like {', '.join(map(str, sample_values[:3]))}"
            }
            column_metadata.append(metadata)

        # Update semantic model with detailed column information
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                    "columns": column_metadata
                }
            ]
        }

        # Save the DataFrame
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df, semantic_model
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Retrieve the API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")
else:
    st.success("API key loaded from environment variable.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Proceed only if both the file is uploaded and the API key is provided
if uploaded_file is not None and openai_key:
    # Preprocess and save the uploaded file
    temp_path, columns, df, semantic_model = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)

        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)

        # Initialize the DuckDbAgent with enhanced system prompt
        duckdb_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=openai_key),
            semantic_model=json.dumps(semantic_model),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="""You are an expert data analyst. When analyzing data:
1. First examine the column metadata and sample values to understand the data format
2. Use LIKE or ILIKE for string matching to handle variations in text data
3. Consider common data variations (spaces, underscores, case sensitivity)
4. Generate SQL queries that are robust to these variations
Return only the SQL query, enclosed in ```sql ``` and give the final answer."""
        )

        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")

        # Add info message about terminal output
        # st.info("💡 Check your terminal for a clearer output of the agent's response")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    with st.spinner('Processing your query...'):
                        # Create a placeholder for the streaming output
                        response_placeholder = st.empty()
                        
                        # Capture the streamed response
                        response = duckdb_agent.run(user_query, stream=True)  # Assuming run supports streaming
                        
                        # If response is an iterator or generator
                        full_response = ""
                        for chunk in response:
                            if hasattr(chunk, 'content'):
                                content = chunk.content
                            else:
                                content = str(chunk)
                            full_response += content
                            response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")
                    st.write("Debug: Exception details:")
                    st.write(traceback.format_exc())

else:
    if not openai_key:
        st.warning("Please set the OPENAI_API_KEY environment variable to your OpenAI API key.")
    if uploaded_file is None:
        st.warning("Please upload a CSV or Excel file to proceed.")