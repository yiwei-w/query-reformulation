import pandas as pd
import jsonlines
import ast

# Path to the input CSV file
mixtral_csv_file_path = './cleaned_mixtral_requests_and_queries.csv'
hotpot_csv_file_path = './cleaned_hotpot_requests_and_queries.csv'

# Path to the output JSONLines file
jsonl_file_path = './finetune_data.jsonl'

# Read the selected columns of the CSV file into a DataFrame
mixtral_df = pd.read_csv(mixtral_csv_file_path, usecols=['request', 'queries'])
mixtral_df = mixtral_df.rename(columns={'request': 'prompt', 'queries': 'completion'})

hotpot_df = pd.read_csv(hotpot_csv_file_path, usecols=['request', 'queries'])
hotpot_df = hotpot_df.rename(columns={'request': 'prompt', 'queries': 'completion'})

merged_df = pd.concat([mixtral_df, hotpot_df])

# deduplicate
merged_df = merged_df.drop_duplicates()

# shuffle
merged_df = merged_df.sample(frac=1).reset_index(drop=True)

# Print total number of rows in the DataFrame
print("Total number of samples:", len(merged_df))

merged_df['completion'] = merged_df['completion'].apply(lambda x: '; '.join(ast.literal_eval(x)))


# Open the JSONLines file for writing
with jsonlines.open(jsonl_file_path, mode='w') as writer:
    # Iterate over each row in the DataFrame
    for _, row in merged_df.iterrows():
        # Write the row (as a dictionary with 'prompt' and 'completion') to the JSONLines file
        writer.write(row.to_dict())
