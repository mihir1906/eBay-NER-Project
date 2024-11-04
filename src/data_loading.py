import pandas as pd

def load_listing_titles(file_path):
    """
    Load Listing Titles from a gzipped TSV file.

    Args:
        file_path (str): Path to the Listing_Titles.tsv.gz file.

    Returns:
        pd.DataFrame: DataFrame containing Record Number and Title.
    """
    print("Loading Listing Titles...")
    try:
        df = pd.read_csv(
            file_path,
            compression='gzip',
            sep='\t',
            encoding='utf-8',
            on_bad_lines='skip',  # Updated for newer pandas versions
            dtype={'Record Number': int, 'Title': str}
        )
        print(f"Listing Titles loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading Listing Titles: {e}")
        return pd.DataFrame()

def load_tagged_titles(file_path):
    """
    Load Train Tagged Titles from a gzipped TSV file.

    Args:
        file_path (str): Path to the Train_Tagged_Titles.tsv.gz file.

    Returns:
        pd.DataFrame: DataFrame containing Record Number, Title, Token, and Tag.
    """
    print("Loading Train Tagged Titles...")
    try:
        df = pd.read_csv(
            file_path,
            compression='gzip',
            sep='\t',
            encoding='utf-8',
            on_bad_lines='skip',
            dtype={'Record Number': int, 'Title': str, 'Token': str, 'Tag': str}
        )
        print(f"Train Tagged Titles loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading Train Tagged Titles: {e}")
        return pd.DataFrame()