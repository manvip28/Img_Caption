import pandas as pd

# Paths
input_file = "dataset/captions.txt"   # your current captions file
output_file = "dataset/captions.csv"  # the CSV we’ll create

# Read the txt file and convert to dataframe
df = pd.read_csv(input_file, header=None, names=['image', 'caption'])

# Save as CSV
df.to_csv(output_file, index=False)
print(f"Saved {len(df)} captions to {output_file}")
