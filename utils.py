import pandas as pd
from pydantic import BaseModel
from openai import OpenAI

class BooleanResponse(BaseModel):
    result: bool

def preprocess_csv_file(dataframe: pd.DataFrame, openai_client: OpenAI) -> pd.DataFrame:
    try:
        df = dataframe
        df = df.replace(r'^\s*$', pd.NA, regex=True)
        df = df.dropna(how='all', axis=0)  
        df_removed_blank_rows = df.dropna(how='all', axis=1)  

        sample_df = df.head(10).iloc[:, :10]
        csv_preview = sample_df.to_string()
        
        formatted_content = (
            "Analyze this CSV data structure:\n\n"
            f"{csv_preview}\n\n"
            "Determine if this data needs to be transposed. "
            "Return true if the column headers are currently horizontal "
            "(meaning they should be vertical for better readability), "
            "false if the headers are already vertical."
        )
        
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a data structure analyzer. Evaluate if the CSV data needs transposition."},
                {"role": "user", "content": formatted_content},
            ],
            response_format=BooleanResponse,
        )
        
        needs_transpose = completion.choices[0].message.parsed.result
        
        if str(needs_transpose).lower() == 'true':
            df = df_removed_blank_rows
            df_transposed = df.transpose()
            
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed.iloc[1:]
            
            columns = df_transposed.columns.fillna('')
            
            # handle duplicate column names
            seen = {}
            new_columns = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
            
            df_transposed.columns = new_columns
            return df_transposed
            
        return df_removed_blank_rows
        
    except Exception as e:
        print(f"Error during preprocessingf: {e}")
        return dataframe

