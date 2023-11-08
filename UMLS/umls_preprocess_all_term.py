import json
import jsonlines
import pandas as pd

# Define the column names for the UMLS concepts.
concept_column_names = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
    "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
    "SUPPRESS", "CVF"
]

# Read the raw UMLS concepts data from a RRF file into a DataFrame.
df_raw = pd.read_table('MRCONSO.RRF', sep='|', index_col=False, names=concept_column_names)

# Export the raw DataFrame as a CSV file for further use.
df_raw.to_csv('MRCONSO.RRF.csv', index=False)

# Load the CSV file into a DataFrame.
df = pd.read_csv("MRCONSO.RRF.csv")

# Filter the DataFrame to keep only the columns necessary for the output.
filtered_data = df[["CUI", "SAB", "CODE", "STR", "LAT"]]

# Save the filtered data to a new CSV file.
filtered_data.to_csv('MRCONSO.RRF.filtered.csv', index=False)

# Open a JSON lines file for writing the processed UMLS concepts.
output_file = jsonlines.open("./umls_files/umls_all_terms.json", 'w')

# Read the filtered data from the CSV file.
df_filtered = pd.read_csv("MRCONSO.RRF.filtered.csv")

# Group the filtered data by the Concept Unique Identifier (CUI).
grouped_by_cui = df_filtered.groupby("CUI")

# Iterate over each group of concepts sharing the same CUI.
for cui, group in grouped_by_cui:
    output_record = {"CUI": cui}  # Start the output record with the CUI.

    # Since CUI is already included in the output_record, we can drop it.
    group = group.drop('CUI', axis=1)

    # Further group the data by Source Abbreviation (SAB) and Language (LAT).
    grouped_by_sab = group.groupby("SAB")
    grouped_by_lat = group.groupby("LAT")

    terms_all_list = []

    # Process groups by SAB and serialize to JSON.
    for sab, sab_group in grouped_by_sab:
        sab_group = sab_group.drop('SAB', axis=1)  # SAB is redundant in the group.
        sab_group_json = json.loads(sab_group.to_json(orient='records'))
        output_record[sab] = sab_group_json

        # Collect all terms for a concatenated string.
        terms_all_list.extend(sab_group['STR'].astype(str))

    # Concatenate terms from each language group.
    for lat, lat_group in grouped_by_lat:
        terms_lat_list = lat_group['STR'].tolist()
        output_record['terms_' + lat] = " ".join(terms_lat_list)

    # Combine all terms into a single string.
    output_record['terms_all'] = " ".join(terms_all_list)

    # Write the structured data to the JSON lines file.
    output_file.write(output_record)

# Close the JSON lines file after writing.
output_file.close()
