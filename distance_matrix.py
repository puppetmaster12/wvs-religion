import pandas as pd
import numpy as np
from itertools import combinations
from tqdm.auto import tqdm
import argparse
import json

class CFSTDistanceCalculator:
    """
    Calculates CFST-based cultural distance matrices.

    This class provides methods to compute pairwise cultural fixation statistics (CFST)
    between groups based on survey question responses. It can generate distance
    matrices for individual questions, composite matrices for a set of questions,
    and also break down the analysis by cultural dimensions.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the survey data.
        scaling (str, optional): The scaling method to apply to the final distance
                                 matrices. Can be 'minmax', 'zscore', or None.
    """
    def __init__(self, df: pd.DataFrame, scaling: str = None):
        """
        Initializes the calculator with data and scaling options.

        Args:
            df: DataFrame with a group column and value columns for questions.
            scaling: Scaling method ('minmax', 'zscore', or None).
        """
        self.df = df
        self.scaling = scaling
    
    @staticmethod
    # def calculate_cfst(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    #     """
    #     Calculates the Cultural Fixation Index (CFST) for a single question.

    #     CFST is calculated as the ratio of between-group variance to total variance.

    #     Args:
    #         df: DataFrame containing the data for the calculation.
    #         value_col: The column with the response values for the question.
    #         group_col: The column identifying the groups to compare.

    #     Returns:
    #         The calculated CFST score as a float.
    #     """
    #     working_df = df[[value_col, group_col]].copy()
    #     working_df = working_df[working_df[value_col] >= 0].dropna()

    #     if working_df.empty or working_df[group_col].nunique() < 2:
    #         print(f"Warning: Not enough data or groups for '{value_col}'. Skipping.")
    #         return np.nan

        
    #     min_val = working_df[value_col].min()
    #     max_val = working_df[value_col].max()
    #     value_col_scaled = (working_df[value_col] - min_val) / (max_val - min_val)
        
    #     total_variance = value_col_scaled.var(ddof=1)
    #     if pd.isna(total_variance) or total_variance == 0:
    #         return 0.0

    #     group_means = working_df.groupby(group_col)[value_col_scaled].mean()
    #     between_group_variance = group_means.var(ddof=1)
    #     if pd.isna(between_group_variance):
    #         return 0.0

    #     cfst_score = between_group_variance / total_variance
    #     return cfst_score
    @staticmethod
    def calculate_cfst(df: pd.DataFrame, value_col: str, group_col: str) -> float:
        working_df = df[[value_col, group_col]].copy()
        working_df = working_df[working_df[value_col] >= 0].dropna()


        if working_df.empty or working_df[group_col].nunique() < 2:
            return np.nan


        p_total = working_df[value_col].mean()
        H_T = 2 * p_total * (1 - p_total)
        if H_T == 0:
            return 0.0


        H_G = 0
        for _, g in working_df.groupby(group_col):
            p_g = g[value_col].mean()
            H_G += 2 * p_g * (1 - p_g)
        H_G /= working_df[group_col].nunique()


        return (H_T - H_G) / H_T
    
    @staticmethod
    def _cfst_variance(df: pd.DataFrame, value_col: str, group_col: str) -> float:
        working_df = df[[value_col, group_col]].copy()
        working_df = working_df[working_df[value_col] >= 0].dropna()

        if working_df.empty or working_df[group_col].nunique() < 2:
            return np.nan

        min_val = working_df[value_col].min()
        max_val = working_df[value_col].max()
        if max_val == min_val:
            return 0.0

        value_col_scaled = (working_df[value_col] - min_val) / (max_val - min_val)
        total_variance = value_col_scaled.var(ddof=1)
        if pd.isna(total_variance) or total_variance == 0:
            return 0.0
    
    def calculate_pairwise_cfst(self, value_col: str, group_col: str) -> dict:
        """
        Calculates CFST scores for all unique pairs of groups for a single question.

        Args:
            value_col: The column with response values.
            group_col: The column with group identifiers.

        Returns:
            A dictionary mapping group pairs (tuples) to their CFST score.
        """
        results = {}
        groups = self.df[group_col].dropna().unique()
        group_pairs = combinations(groups, 2)

        for group1, group2 in tqdm(group_pairs, desc=f"Pairwise for '{value_col}'", leave=False):
            pair_df = self.df[self.df[group_col].isin([group1, group2])]
            score = self.calculate_cfst(pair_df, value_col, group_col)
            results[(group1, group2)] = score

        return results

    def scale_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the specified scaling method to a distance matrix.

        Args:
            matrix: The distance matrix (DataFrame) to scale.

        Returns:
            The scaled distance matrix.
        """
        if self.scaling is None:
            return matrix

        values = matrix.values
        # Use np.nanmean and np.nanstd for calculations, ignoring NaNs
        if self.scaling == "minmax":
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            if max_val - min_val == 0:
                return matrix  # avoid divide-by-zero
            scaled_values = (values - min_val) / (max_val - min_val)
        elif self.scaling == "zscore":
            mean_val, std_val = np.nanmean(values), np.nanstd(values)
            if std_val == 0:
                return matrix
            scaled_values = (values - mean_val) / std_val
        else:
            raise ValueError("Scaling method not recognized. Use None, 'minmax', or 'zscore'.")

        return pd.DataFrame(scaled_values, index=matrix.index, columns=matrix.columns)

    def create_single_value_matrix(self, value_col: str, group_col: str) -> pd.DataFrame:
        """
        Creates a distance matrix for a single question.

        Args:
            value_col: The question column to use.
            group_col: The group identifier column.

        Returns:
            A DataFrame representing the distance matrix for the question.
        """
        pairwise_scores = self.calculate_pairwise_cfst(value_col, group_col)
        groups = sorted(self.df[group_col].dropna().unique())
        distance_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)

        for (group1, group2), score in pairwise_scores.items():
            distance_matrix.loc[group1, group2] = score
            distance_matrix.loc[group2, group1] = score

        np.fill_diagonal(distance_matrix.values, 0)
        return distance_matrix

    def create_composite_distance_matrix(self, value_cols: list, group_col: str) -> pd.DataFrame:
        """
        Creates a composite distance matrix by averaging matrices from multiple questions.

        Args:
            value_cols: A list of question columns to include.
            group_col: The group identifier column.

        Returns:
            A single, averaged distance matrix for all specified questions.
        """
        matrix_list = []
        for value_col in tqdm(value_cols, desc="Processing questions", leave=False):
            single_matrix = self.create_single_value_matrix(value_col, group_col)
            if not single_matrix.isna().all().all():
                matrix_list.append(single_matrix.to_numpy())

        if not matrix_list:
            print("Warning: No valid distance matrices could be calculated.")
            return pd.DataFrame()

        # Calculate mean, ignoring NaNs
        mean_matrix_values = np.nanmean(np.array(matrix_list), axis=0)
        groups = sorted(self.df[group_col].dropna().unique())
        composite_matrix = pd.DataFrame(mean_matrix_values, index=groups, columns=groups)

        return composite_matrix
        # return self.scale_matrix(composite_matrix)

    def calculate_dimensional_matrices(self, dimensions: dict, group_col: str) -> dict:
        """
        Calculates a CFST distance matrix for each cultural dimension.

        Args:
            dimensions: A dictionary mapping dimension names to lists of question columns.
            group_col: The group identifier column.

        Returns:
            A dictionary mapping dimension names to their corresponding distance matrix (DataFrame).
        """
        dimensional_matrices = {}
        for dim_name, questions in tqdm(dimensions.items(), desc="Processing Dimensions"):
            print(f"Calculating distance matrix for dimension: {dim_name}")
            dim_matrix = self.create_composite_distance_matrix(questions, group_col)
            dimensional_matrices[dim_name] = dim_matrix
        return dimensional_matrices


# For direct function usage (optional, for backward compatibility)
def calculate_cfst(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    return CFSTDistanceCalculator._cfst_variance(df, value_col, group_col)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate composite and dimensional CFST distance matrices.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("dimensions_path", type=str, help="Path to the JSON file with question dimensions.")
    parser.add_argument("--group_col", type=str, required=False, help="Name of the group column.")
    parser.add_argument("--scaling", type=str, choices=["minmax", "zscore"], default=None, help="Scaling method for distance matrices.")
    args = parser.parse_args()

    # --- Manual argument simulation ---
    # class Args:
        # You would replace 'path/to/your/data.csv' with the actual path to your data file.
        # For this example, we will create a dummy dataframe.
        # csv_path = 'path/to/your/data.csv'
        # dimensions_path = 'questions_dim.json'
        # group_col = 'Religion'
        # scaling = 'zscore'

    # args = Args()
    # ------------------------------------

    df = pd.read_csv(args.csv_path)
    group_col = 'Religion'
    # religions = ['Religion_A', 'Religion_B', 'Religion_C', 'Religion_D']
    # data = {'Religion': np.random.choice(religions, size=500)}
    # with open(args.dimensions_path, 'r') as f:
    #     dims = json.load(f)
    # all_questions = [q for questions in dims.values() for q in questions]
    # for q in all_questions:
    #     data[q] = np.random.randint(1, 11, size=500)
    # df = pd.DataFrame(data)


    # Load dimensions from JSON file
    with open(args.dimensions_path, 'r') as f:
        dimensions = json.load(f)

    # Instantiate the calculator
    calculator = CFSTDistanceCalculator(df, scaling=args.scaling)

    # --- 1. Calculate and save dimensional matrices ---
    dimensional_matrices = calculator.calculate_dimensional_matrices(dimensions, group_col)
    for dim_name, matrix in dimensional_matrices.items():
        if not matrix.empty:
            output_path = f'dimensional_matrix_{dim_name}.csv'
            matrix.to_csv(output_path)
            print(f"Saved dimensional matrix for '{dim_name}' to {output_path}")

    # --- 2. Create and save the dimensional CFST summary ---
    # This provides a "CFST for each dimension for each religion"
    summary_data = {}
    for dim_name, matrix in dimensional_matrices.items():
        if not matrix.empty:
            # The mean distance of a religion to all others in this dimension
            summary_data[dim_name] = matrix.mean(axis=1)

    if summary_data:
        dimensional_summary_df = pd.DataFrame(summary_data).T # Transpose to have dimensions as rows
        dimensional_summary_df.to_csv('dimensional_cfst_summary.csv')
        print("\nSaved dimensional CFST summary to dimensional_cfst_summary.csv")
        print(dimensional_summary_df)


    # --- 3. Calculate and save the total composite distance matrix ---
    all_questions = [q for questions in dimensions.values() for q in questions]
    print("\nCalculating the final composite distance matrix using all questions...")
    composite_matrix = calculator.create_composite_distance_matrix(all_questions, group_col)
    if not composite_matrix.empty:
        composite_matrix.to_csv('composite_distance_matrix.csv')
        print("Saved composite distance matrix to composite_distance_matrix.csv")
        print(composite_matrix)