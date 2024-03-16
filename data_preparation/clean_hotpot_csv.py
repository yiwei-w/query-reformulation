import ast
import pandas as pd

df = pd.read_csv('hotpot_requests_and_queries.csv')

# Function to convert string representation of tuples/lists to actual tuples/lists
def convert_string_to_tuple(s):
    try:
        # Convert string to actual tuple/list
        return ast.literal_eval(s)
    except ValueError:
        # Return original string if conversion fails
        return s

# Apply the conversion function to the relevant columns
df['0'] = df['0'].apply(convert_string_to_tuple)
df['1'] = df['1'].apply(convert_string_to_tuple)

# Extracting inner_monologue and queries into separate columns
df['inner_monologue'] = df['0'].apply(lambda x: x[1] if type(x) == tuple and x[0] == 'inner_monologue' else None)
df['queries'] = df['1'].apply(lambda x: x[1] if type(x) == tuple and x[0] == 'queries' else None)

# Dropping the original tuple columns
cleaned_df = df.drop(columns=['0', '1'])

print(cleaned_df.head())

# Save the cleaned DataFrame to a new CSV file
cleaned_df.to_csv('cleaned_hotpot_requests_and_queries.csv', index=False)


