import pandas as pd
import json
import re

def parse_json_to_dataframe(json_string):
    """
    Parse a JSON string containing data and return it as a Pandas DataFrame.
    
    Parameters:
    json_string (str): String containing JSON data enclosed in triple backticks.
    
    Returns:
    pd.DataFrame: Parsed data as a Pandas DataFrame.
    
    Examples:
    json_string = '''Here is the total sales by geography:

    ```json
    [
        {"Geography": "East", "Sales": 3070.0},
        {"Geography": "North", "Sales": 3440.0},
        {"Geography": "South", "Sales": 3825.0},
        {"Geography": "West", "Sales": 3830.0}
    ]
    ```'''

    df = parse_json_to_dataframe(json_string)
    print(df)
    """
    # Extract the JSON portion using regular expressions
    match = re.search(r'```json\n(.*?)\n```', json_string, re.DOTALL)
    if not match:
        raise ValueError("No JSON content found in the provided string.")
    
    json_content = match.group(1)
    
    # Parse the JSON string to Python objects
    data = json.loads(json_content)
    
    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    return df


