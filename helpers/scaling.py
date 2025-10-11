import pandas as pd
import numpy as np
import os

def minmax_scale_distance_matrix(csv_filepath):
    """
    Reads a distance matrix from a CSV and normalizes it as a percentage.

    The min-max scaling formula is used and multiplied by 100:
        x_scaled = ((x - min_val) / (max_val - min_val)) * 100
    
    Args:
        csv_filepath (str): The path to the CSV file containing the distance matrix.
                            The file should have a header row and an index column.

    Returns:
        pandas.DataFrame: A DataFrame containing the normalized distance matrix,
                          preserving the original headers and index.
                          Returns None if the file is not found or is empty.
    """
    try:
        # Read the CSV file into a pandas DataFrame.
        # index_col=0 tells pandas to use the first column as the DataFrame index.
        df = pd.read_csv(csv_filepath, index_col=0)

        if df.empty:
            print("Error: The CSV file is empty.")
            return None

        # The DataFrame already contains the numerical data.
        matrix = df.to_numpy()

        # Find the minimum and maximum values in the entire matrix.
        min_val = np.min(matrix)
        max_val = np.max(matrix)

        # Handle the case where all values are the same to avoid division by zero.
        if max_val == min_val:
            # Return a matrix of zeros with the same shape.
            normalized_matrix_np = np.zeros(matrix.shape)
        else:
            # Apply the min-max scaling formula and multiply by 100 for percentage.
            normalized_matrix_np = ((matrix - min_val) / (max_val - min_val)) * 100

        # Create a new DataFrame with the normalized data, keeping original index and columns.
        normalized_df = pd.DataFrame(normalized_matrix_np, index=df.index, columns=df.columns)

        return normalized_df

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Specify the path to your CSV file.
    csv_file = "composite_distance_matrix.csv"
    
    # Check if the file exists before proceeding.
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found in the current directory.")
    else:
        print(f"Reading data from: '{csv_file}'")
        
        # 2. Call the function to normalize the matrix.
        normalized_df = minmax_scale_distance_matrix(csv_file)
        
        # 3. Print the result.
        if normalized_df is not None:
            print("\nOriginal Matrix (first 5 rows/columns):")
            print(pd.read_csv(csv_file, index_col=0).iloc[:5, :5])
            
            print("\nNormalized Matrix (first 5 rows/columns, as a percentage):")
            # Use .iloc to show a slice of the large matrix for readability.
            print(normalized_df.iloc[:5, :5])

            # Optionally, save the normalized matrix to a new CSV file.
            # output_filename = "normalized_distance_matrix.csv"
            # normalized_df.to_csv(output_filename)
            # print(f"\nNormalized matrix saved to '{output_filename}'")