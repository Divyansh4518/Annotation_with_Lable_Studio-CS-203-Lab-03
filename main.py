import pandas as pd
import json as json
def create_annotation_dataframe(csv_file_path):
    """
    Creates a Pandas DataFrame from Label Studio exported CSV data,
    parsing the 'label' column to extract words and their tags.

    Args:
        csv_file_path (str): Path to the CSV file containing Label Studio export.

    Returns:
        pandas.DataFrame: DataFrame with columns 'word' and 'tag'.
    """
    df = pd.read_csv(csv_file_path)
    annotation_data = []

    for index, row in df.iterrows():
        label_json_str = row['label']
        try:
            label_json = json.loads(label_json_str)
            if isinstance(label_json, list): # Handle cases where label is a list of annotations
                for annotation in label_json:
                    if 'text' in annotation and 'labels' in annotation and annotation['labels']:
                        word = annotation['text']
                        tag = annotation['labels'][0] # Assuming only one label per word
                        annotation_data.append({'word': word, 'tag': tag})
            elif isinstance(label_json, dict) and 'text' in label_json and 'labels' in label_json and label_json['labels']: # Handle cases where label is a single annotation dict
                word = label_json['text']
                tag = label_json['labels'][0]
                annotation_data.append({'word': word, 'tag': tag})

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON in row {index}, annotation_id {row.get('annotation_id', 'N/A')}")
            continue # Skip to the next row if JSON is invalid

    annotation_df = pd.DataFrame(annotation_data)
    return annotation_df

# Example usage:
csv_file1 = 'data/nlp_data1.csv'
csv_file2 = 'data/nlp_data2.csv'

annotation_df = create_annotation_dataframe(csv_file1)
annotation_df = create_annotation_dataframe(csv_file2)

annotation_data2= annotation_df.to_csv('annotation_data1.csv', index = True)
annotation_data2= annotation_df.to_csv('annotation_data2.csv', index = True)

def compare_annotation_datasets(csv_file1_path, csv_file2_path):
    """
    Compares two NLP annotated datasets to find common words and their POS tags.

    Args:
        csv_file1_path (str): Path to the CSV file for the first annotation dataset.
        csv_file2_path (str): Path to the CSV file for the second annotation dataset.

    Returns:
        tuple: A tuple containing two lists:
            - list1: List of (word, pos_tag) pairs from annotation_data1 for common words.
            - list2: List of (word, pos_tag) pairs from annotation_data2 for common words.
    """

    df1 = pd.read_csv(csv_file1_path, index_col=0)  # Assuming first unnamed column is index
    df2 = pd.read_csv(csv_file2_path, index_col=0)  # Assuming first unnamed column is index

    list1 = []
    list2 = []

    words_df2 = set(df2['word']) # For faster lookups

    for index, row1 in df1.iterrows():
        word_df1 = row1['word']
        tag_df1 = row1['tag']

        if word_df1 in words_df2:
            # Find the corresponding row in df2 for the same word
            row_df2 = df2[df2['word'] == word_df1].iloc[0] # Assuming unique words in each df
            tag_df2 = row_df2['tag']

            list1.append((word_df1, tag_df1))
            list2.append((word_df1, tag_df2))

    return list1, list2


csv_file1 = 'annotation_data1.csv'
csv_file2 = 'annotation_data2.csv'

common_words_list1, common_words_list2 = compare_annotation_datasets(csv_file1, csv_file2)

print("List 1 (from annotation_data1):")
print(common_words_list1)
print("\nList 2 (from annotation_data2):")
print(common_words_list2)
from sklearn.metrics import cohen_kappa_score

def calculate_cohen_kappa(list1, list2):
    """
    Calculates Cohen's Kappa score for inter-annotator agreement
    based on two lists of (word, tag) pairs for common words.

    Args:
        list1 (list): List of (word, tag) pairs from the first annotation dataset.
        list2 (list): List of (word, tag) pairs from the second annotation dataset.

    Returns:
        float: Cohen's Kappa score.
    """

    tags1 = [pair[1] for pair in list1] # Extract tags from list1
    tags2 = [pair[1] for pair in list2] # Extract tags from list2

    if not tags1 or not tags2: # Handle empty lists case
        print("Warning: One or both lists are empty. Cohen's Kappa cannot be calculated.")
        return None # Or return a specific value like -2 to indicate error

    if len(tags1) != len(tags2): # Sanity check for list lengths
        raise ValueError("Error: Lists of tags must have the same length for Cohen's Kappa calculation.")

    kappa_score = cohen_kappa_score(tags1, tags2)
    return kappa_score


kappa = calculate_cohen_kappa(common_words_list1, common_words_list2)

if kappa is not None:
    print(f"Cohen's Kappa Score: {kappa}")

    # Interpretation of Kappa Score (common guidelines):
    if kappa < 0:
        interpretation = "Less than chance agreement"
    elif kappa <= 0.20:
        interpretation = "Slight agreement"
    elif kappa <= 0.40:
        interpretation = "Fair agreement"
    elif kappa <= 0.60:
        interpretation = "Moderate agreement"
    elif kappa <= 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    print(f"Interpretation: {interpretation}")
    

def extract_truck_info(csv_file_path):
    """
    Extracts 'truck', 'no truck', and index information from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with columns 'index', 'truck', 'no_truck'.
    """

    df = pd.read_csv(csv_file_path)
    # Create new columns 'truck' and 'no_truck' based on 'choice' column
    df['truck'] = df['choice'].apply(lambda x: 1 if x == 'Trucks' else 0)
    df['no_truck'] = df['choice'].apply(lambda x: 1 if x == 'No Trucks' else 0)
    # Reset index to get a separate index column
    df = df.reset_index()
    # Select desired columns
    result_df = df[['truck', 'no_truck']]

    return result_df

# Example usage:
file_path1 = 'mine_cv.csv'
file_path2 = 'other_cv.csv'
file_path3 = 'teammate_cv.csv'

truck_info_df1 = extract_truck_info(file_path1)
truck_info_df2 = extract_truck_info(file_path2)
truck_info_df3 = extract_truck_info(file_path3)

mine_trucks = truck_info_df1['truck'].tolist()
teammate_trucks = truck_info_df3['truck'].tolist()

kappa = cohen_kappa_score(mine_trucks, teammate_trucks)
print(f"Cohen's Kappa between mine and teammate: {kappa}")
def fleiss_kappa(ratings):
    """
    Calculates Fleiss' Kappa for a set of ratings.

    Args:
        ratings: A list of lists, where each inner list represents the ratings
                 given by each rater to a single subject.
                 Each rating should be a category label (e.g., 0 or 1 for binary).

    Returns:
        The Fleiss' Kappa value (float).
    """
    n_subjects = len(ratings)  # Number of subjects
    n_raters = len(ratings[0])  # Number of raters (assuming all subjects have the same number of raters)
    categories = sorted(list(set(sum(ratings, [])))) # Unique categories
    n_categories = len(categories)

    # 1. Calculate P_j (proportion of assignments to each category)
    P_j = {}
    for j in categories:
        P_j[j] = sum(row.count(j) for row in ratings) / (n_subjects * n_raters)

    # 2. Calculate P_i (agreement level for each subject)
    P_i = []
    for row in ratings:
        p_i_row = 0
        for j in categories:
            n_ij = row.count(j)
            p_i_row += n_ij * (n_ij - 1)
        P_i.append(p_i_row / (n_raters * (n_raters - 1)))

    # 3. Calculate overall agreement (P_bar)
    P_bar = sum(P_i) / n_subjects

    # 4. Calculate expected agreement by chance (P_bar_e)
    P_bar_e = sum(P_j[j]**2 for j in categories)

    # 5. Calculate Fleiss' Kappa
    kappa = (P_bar - P_bar_e) / (1 - P_bar_e)

    return kappa

# Combine the data from the three annotators for Fleiss' Kappa
combined_ratings = []
for i in range(len(truck_info_df1)):
    combined_ratings.append([
        truck_info_df1['truck'][i],
        truck_info_df2['truck'][i],
        truck_info_df3['truck'][i]
    ])

# Calculate Fleiss' Kappa
kappa_fleiss = fleiss_kappa(combined_ratings)
print(f"Fleiss' Kappa: {kappa_fleiss}")
