import os
import json
import tempfile
import csv
import streamlit as st
import pandas as pd
import uuid

from openai import OpenAI
from pathlib import Path
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
import re
import traceback
from dotenv import load_dotenv
from utils import preprocess_csv_file, guess_column_descriptions

# Add this class before your main code
class PathJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

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

openai_client = OpenAI(api_key=openai_key)


def _preprocess_and_save_csv(file, tmp_dir: Path):
    df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
    df = preprocess_csv_file(df, openai_client=openai_client)

    # Generate column metadata for the semantic model
    column_metadata = []
    column_descriptions = guess_column_descriptions(df, openai_client=openai_client)
    for col in df.columns:
        col_type = str(df[col].dtype)
        # Convert sample values to strings to handle timestamps and other non-JSON serializable types
        sample_values = df[col].dropna().unique()[:5]
        sample_values = [
            str(val) for val in sample_values
        ]  # Convert all values to strings

        metadata = {
            "name": col,
            "type": col_type,
            "sample_values": sample_values,
            "description": f"Contains {col_type} values like {', '.join(map(str, sample_values[:3]))}. {column_descriptions[col]}",
        }

        column_metadata.append(metadata)

    # Update semantic model with detailed column information
    temp_path = tmp_dir / f"{uuid.uuid4()}.csv"
    semantic_model = {
        "tables": [
            {
                "name": "uploaded_data",
                "description": "Contains the uploaded dataset.",
                "path": temp_path,
                "columns": column_metadata,
            }
        ]
    }

    # Save the DataFrame
    df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
    return [temp_path], [df.columns.tolist()], [df], semantic_model


def _preprocess_and_save_xlsx(file, tmp_dir: Path):
    excel_file = pd.ExcelFile(file)

    tables = []
    temp_paths = []
    columns_per_df = []
    dfs = []

    for sheet_name in excel_file.sheet_names:
        assert isinstance(sheet_name, str)

        df = pd.read_excel(
            file,
            sheet_name=sheet_name,
            na_values=["NA", "N/A", "missing"],
            # US date format
            date_format="%m-%d-%Y",
        )

        df = preprocess_csv_file(df, openai_client=openai_client)

        # Generate column metadata for the semantic model
        column_metadata = []
        column_descriptions = guess_column_descriptions(df, openai_client=openai_client)
        for col in df.columns:
            col_type = str(df[col].dtype)
            # Convert sample values to strings to handle timestamps and other non-JSON serializable types
            sample_values = df[col].dropna().unique()[:5]
            sample_values = [
                str(val) for val in sample_values
            ]  # Convert all values to strings

            description = column_descriptions.get(col, "")
            metadata = {
                "name": col,
                "type": col_type,
                "sample_values": sample_values,
                "description": f"Contains {col_type} values like {', '.join(map(str, sample_values[:3]))}. {description}",
            }

            column_metadata.append(metadata)

        temp_path = tmp_dir / f"{uuid.uuid4()}.csv"
        tables.append(
            {
                "name": sheet_name.replace(" ", "_")
                .replace("-", "_")
                .replace(".", "_"),
                "description": f"Contains the data from the sheet: {sheet_name}",
                "path": temp_path,
                "columns": column_metadata,
            }
        )
        temp_paths.append(temp_path)
        columns_per_df.append(df.columns.tolist())
        dfs.append(df)
        df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

    semantic_model = {"tables": tables}
    return temp_paths, columns_per_df, dfs, semantic_model


# Function to preprocess and save the uploaded file
def preprocess_and_save(file, tmp_dir: Path):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith(".csv"):
            return _preprocess_and_save_csv(file, tmp_dir)
        elif file.name.endswith(".xlsx"):
            return _preprocess_and_save_xlsx(file, tmp_dir)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None, None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None, None

def create_visualization(x_values, y_values, title=None):
    try:
        if not x_values or not y_values:
            st.error("No data available for visualization")
            return

        # Create a DataFrame from the x and y values
        chart_data = pd.DataFrame({
            'x': x_values,
            'y': y_values
        })
        
        # Set the index for the line chart
        chart_data.set_index('x', inplace=True)

        # Add title if provided
        if title:
            st.subheader(title)
        
        # Create the line chart
        st.subheader("Line chart")
        st.line_chart(chart_data)
        st.subheader("Bar chart")
        st.bar_chart(chart_data)
        
            
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        st.write("Debug info:")
        st.write(f"X values: {x_values[:5]} ...")
        st.write(f"Y values: {y_values[:5]} ...")


# Streamlit app
st.title("📊 Data Analyst Agent")

# Retrieve the API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error(
        "Please set the OPENAI_API_KEY environment variable to your OpenAI API key."
    )
else:
    st.success("API key loaded from environment variable.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# Proceed only if both the file is uploaded and the API key is provided
if uploaded_file is not None and openai_key:
    # Preprocess and save the uploaded file
    with tempfile.TemporaryDirectory() as _tmp_dir:
        tmp_dir = Path(_tmp_dir)
        temp_paths, columns_per_df, dfs, semantic_model = preprocess_and_save(
            uploaded_file, tmp_dir
        )
        if temp_paths and columns_per_df and dfs is not None:
            with st.expander("Raw Uploaded Data"):
                for df, table in zip(dfs, semantic_model["tables"]):
                    st.write(table["name"])
                    st.dataframe(df.head())
            # Initialize the DuckDbAgent with enhanced system prompt
            duckdb_agent = DuckDbAgent(
                model=OpenAIChat(model="gpt-4o", api_key=openai_key),
                semantic_model=json.dumps(semantic_model, cls=PathJSONEncoder),
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
    Return only the SQL query, enclosed in ```sql ``` and give the final answer.""",
            )
            
            duckdb_visualizer = DuckDbAgent(
                model=OpenAIChat(model="gpt-4", api_key=openai_key),
                semantic_model=json.dumps(semantic_model, cls=PathJSONEncoder),
                tools=[PandasTools()],
                markdown=True,
                add_history_to_messages=False,
                followups=False,
                read_tool_call_history=False,
                system_prompt="""You are an expert data analyst and visualization specialist. When analyzing data:
    1. First examine the column metadata and sample values to understand the data format
    2. Analyze the data and prepare it for visualization
    3. MAKE SUER YOU Return the data in the following format:
    ```viz
    {
        "type": "line",
        "x_values": [...], // array of x-axis values
        "y_values": [...], // array of y-axis values
        "title": "Chart Title"
    }
    ```
    Make sure to return only numeric values for y_values and appropriate values for x_values."""
            )

            # Main query input widget
            user_query = st.text_area("Ask a query about the data:")
            submit_query = st.button("Submit Query")
            # Add info message about terminal output
            # st.info("💡 Check your terminal for a clearer output of the agent's response")
            if submit_query:
                if user_query.strip() == "":
                    st.warning("Please enter a query.")
                else:
                    try:
                        with st.spinner("Processing your query..."):
                            response_placeholder = st.empty()
                            full_response = ""
                            
                            
                            if "@plot" in user_query:
                                user_query = user_query + """    
                                MAKE SUER YOU Return the data in the following format:
                                ```viz
                                {
                                    "type": "line",
                                    "x_values": [...], // array of x-axis values
                                    "y_values": [...], // array of y-axis values
                                    "title": "Chart Title"
                                }
                                ```
                                Make sure to return only numeric values for y_values and appropriate values for x_values."""

                                response = duckdb_visualizer.run(user_query,show_function_calls=True, stream=True)
                                counter= 0
                        
                                # Process the response to find visualization config
                                for chunk in response:
                                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                    full_response += content
                                    
                                    # Look for viz configuration.
                                    viz_match = re.search(r'```viz\n(.*?)```', full_response, re.DOTALL)
                                    if viz_match and counter == 0:
                                        counter += 1
                                        # st.write("Visualization configuration found")
                                        try:
                                            viz_config = json.loads(viz_match.group(1))
                                            
                                            # Create visualization directly from the provided data
                                            create_visualization(
                                                x_values=viz_config.get('x_values'),
                                                y_values=viz_config.get('y_values'),
                                                title=viz_config.get('title')
                                            )
                                        except json.JSONDecodeError:
                                            st.error("Invalid visualization configuration")
                                        except Exception as e:
                                            st.error(f"Error creating visualization: {e}")
                                    
                                    response_placeholder.markdown(full_response)
                            else:
                                # Capture the streamed response
                                response = duckdb_agent.run(
                                    user_query, stream=True
                                )  # Assuming run supports streaming

                                # If response is an iterator or generator
                                for chunk in response:
                                    if hasattr(chunk, "content"):
                                        content = chunk.content
                                    else:
                                        content = str(chunk)
                                    full_response += content
                                    response_placeholder.markdown(full_response)

                    except Exception as e:
                        st.error(f"Error generating response from the DuckDbAgent: {e}")
                        st.error(
                            "Please try rephrasing your query or check if the data format is correct."
                        )
                        st.write("Debug: Exception details:")
                        st.write(traceback.format_exc())

        else:
            if not openai_key:
                st.warning(
                    "Please set the OPENAI_API_KEY environment variable to your OpenAI API key."
                )
            if uploaded_file is None:
                st.warning("Please upload a CSV or Excel file to proceed.")
