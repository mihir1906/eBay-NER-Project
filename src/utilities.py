
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def prepare_data(df):
    """
    Prepare the DataFrame by sorting it based on 'Record Number' and resetting the index.

    Args:
        df (pd.DataFrame): The input DataFrame to be prepared.

    Returns:
        pd.DataFrame: The prepared DataFrame sorted by 'Record Number' with a reset index.
    """
    # Check if 'Record Number' column exists
    if 'Record Number' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'Record Number' column.")

    # Ensure 'Record Number' is of numeric type
    df['Record Number'] = pd.to_numeric(df['Record Number'], errors='coerce')

    # Drop rows with missing 'Record Number' values
    df = df.dropna(subset=['Record Number'])

    # Sort the DataFrame by 'Record Number'
    df_sorted = df.sort_values(by='Record Number', ascending=True)

    # Reset the index of the sorted DataFrame
    df_sorted.reset_index(drop=True, inplace=True)

     # Rename columns for improved clarity
    df_sorted.rename(columns={'Record Number': 'sentence_id', 'Aspect Value': 'words', 'Aspect Name': 'labels'}, inplace=True)
    

    return df_sorted




def plot_countplot(dataframe: pd.DataFrame) -> None:
    """
    Plots a horizontal count plot of the 'Tag' column in the provided DataFrame, showing the 
    distribution of different tags. Includes data labels for each count, a custom color 
    palette, and formatted labels.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the 'Tag' column to plot.

    Returns:
    None
    """
    # Set up figure with custom dimensions
    plt.figure(figsize=(14, 8))

    # Set a dark grid theme for the plot
    sns.set_theme(style="whitegrid")

    # Get the count of each unique tag, including NaN values
    tag_counts = dataframe['Aspect Name'].value_counts(dropna=False)

    # Create a horizontal count plot of 'Tag' values
    sns.countplot(
        y='Aspect Name', 
        data=dataframe, 
        order=tag_counts.index, 
        hue='Aspect Name',
        palette="viridis",    # Optional color palette for aesthetics
        edgecolor="black",
        legend=False                   # Adds a border around bars for clarity
    )

    # Add title and axis labels with custom formatting
    plt.title('Aspect Name Distribution', fontsize=18, weight='bold', pad=20)
    plt.xlabel('Count', fontsize=14, labelpad=10)
    plt.ylabel('Aspect Name', fontsize=14, labelpad=10)

    # Add data labels to each bar for easier visualization
    for i, count in enumerate(tag_counts.values):
            plt.text(count + 170, i, str(count), color='black', ha='center', va='center', fontsize=12)

    # Remove unnecessary top and right spines for a cleaner look
    sns.despine(left=True, bottom=True)

    # Display the plot
    plt.show()
    

def fix_aspect_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the DataFrame to concatenate 'Token' values in rows where 'Tag' is NaN, 
    merging them with the previous row's 'Token'. Drops rows with NaN 'Tag' values 
    after merging and renames columns for clarity. Resets the index after modifications.

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'Token' and 'Tag' to process.

    Returns:
    pd.DataFrame: Processed DataFrame with merged 'Token' values, renamed columns, 
                  and reset indices.
    """
    
    # Iterate over DataFrame rows in reverse order to handle row merging and deletion
    for i in range(len(df) - 1, -1, -1):
        # Check if the 'Tag' value is NaN in the current row
        if pd.isna(df.loc[i, 'Tag']):
            # Concatenate the 'Token' of the current row to the 'Token' of the previous row
            df.loc[i - 1, 'Token'] = df.loc[i - 1, 'Token'] + ' ' + df.loc[i, 'Token']
            # Drop the current row as it has been merged with the previous row
            df = df.drop(i)
    
    # Reset the index to reorder the indices sequentially after row deletions
    df = df.reset_index(drop=True)
    
    # Rename columns for improved clarity
    df.rename(columns={'Token': 'Aspect Value', 'Tag': 'Aspect Name'}, inplace=True)
    
    # Return the modified DataFrame
    return df

def convert_to_iob(df):
    """
    Converts aspect labels in a DataFrame to the IOB (Inside-Outside-Beginning) format.

    Parameters:
        df (pd.DataFrame): A DataFrame containing labeled data. It must have at least two columns:
                           - 'Record Number': Groups sentences or records together.
                           - 'Aspect Name': Contains labels for each token, where 'O' denotes no label.

    Returns:
        list: A list of converted labels in IOB format corresponding to the order of tokens in the DataFrame.
    """

    # Initialize an empty list to store the converted IOB labels
    converted_labels = []

    # Iterate through groups of sentences, grouped by the 'Record Number' column
    for sentence_id, group in df.groupby('Record Number'):
        # Variable to track the label of the previous token, initialized as 'O'
        previous_label = 'O'

        # Iterate through the labels in the 'Aspect Name' column for the current group
        for label in group['Aspect Name']:
            if label == 'O':
                # If the label is 'O', mark the token as outside and reset the previous label
                converted_labels.append('O')
                previous_label = 'O'
            else:
                # If the current label differs from the previous label, mark it as a beginning (B-)
                if label != previous_label.replace('I-', ''):
                    converted_labels.append(f'B-{label}')
                else:
                    # Otherwise, mark it as an inside (I-) of the current label
                    converted_labels.append(f'I-{label}')

                # Update the previous label for the next token
                previous_label = f'I-{label}'

    # Return the list of converted labels
    return converted_labels


