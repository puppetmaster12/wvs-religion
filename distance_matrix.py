import pandas as pd
import numpy as np
from itertools import combinations
from tqdm.auto import tqdm
import argparse

class CFSTDistanceCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def calculate_cfst(df: pd.DataFrame, value_col: str, group_col: str) -> float:
        working_df = df[[value_col, group_col]].copy()
        working_df = working_df[working_df[value_col] >= 0].dropna()

        if working_df.empty or working_df[group_col].nunique() < 2:
            print(f"Warning: Not enough data or groups for '{value_col}'. Skipping.")
            return np.nan

        total_variance = working_df[value_col].var(ddof=1)
        if pd.isna(total_variance) or total_variance == 0:
            return 0.0

        group_means = working_df.groupby(group_col)[value_col].mean()
        between_group_variance = group_means.var(ddof=1)
        if pd.isna(between_group_variance):
            return 0.0

        cfst_score = between_group_variance / total_variance
        return cfst_score

    def calculate_pairwise_cfst(self, value_col: str, group_col: str) -> dict:
        results = {}
        groups = self.df[group_col].dropna().unique()
        group_pairs = combinations(groups, 2)

        for group1, group2 in tqdm(
            group_pairs, desc=f"Pairwise for '{value_col}'", leave=False
        ):
            pair_df = self.df[self.df[group_col].isin([group1, group2])]
            score = self.calculate_cfst(pair_df, value_col, group_col)
            results[(group1, group2)] = score

        return results

    def create_distance_matrix(self, value_col: str, group_col: str) -> pd.DataFrame:
        pairwise_scores = self.calculate_pairwise_cfst(value_col, group_col)
        groups = self.df[group_col].dropna().unique()
        distance_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)

        for (group1, group2), score in tqdm(
            pairwise_scores.items(), desc="Calculating scores"
        ):
            distance_matrix.loc[group1, group2] = score
            distance_matrix.loc[group2, group1] = score

        np.fill_diagonal(distance_matrix.values, 0)
        return distance_matrix

    def create_single_value_matrix(self, value_col: str, group_col: str) -> pd.DataFrame:
        pairwise_scores = self.calculate_pairwise_cfst(value_col, group_col)
        groups = self.df[group_col].dropna().unique()
        distance_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)

        for (group1, group2), score in pairwise_scores.items():
            distance_matrix.loc[group1, group2] = score
            distance_matrix.loc[group2, group1] = score

        np.fill_diagonal(distance_matrix.values, 0)
        return distance_matrix

    def create_composite_distance_matrix(self, value_cols: list, group_col: str) -> pd.DataFrame:
        matrix_list = []
        for value_col in tqdm(value_cols, desc="Processing all questions"):
            single_matrix = self.create_single_value_matrix(value_col, group_col)
            if not single_matrix.empty:
                matrix_list.append(single_matrix)

        if not matrix_list:
            print("Warning: No valid distance matrices could be calculated.")
            return pd.DataFrame()

        composite_matrix = pd.concat(matrix_list).groupby(level=0).mean()
        return composite_matrix

# For direct function usage (optional, for backward compatibility)
def calculate_cfst(df: pd.DataFrame, value_col: str, group_col: str) -> float:
    return CFSTDistanceCalculator.calculate_cfst(df, value_col, group_col)

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate composite CFST distance matrix from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("--group_col", type=str, required=False, help="Name of the group column.")
    parser.add_argument("--value_cols", type=str, nargs='+', required=False, help="List of value columns to use.")
    args = parser.parse_args()

    # Load DataFrame from CSV
    df = pd.read_csv(args.csv_path)
    questions = [
    "A001", "A002", "A003", "A004", "A005", "A006",
    "F063", "F050", "F051", "F053", "F054", "F202",
    "F203", "F028B", "F028", "F034", "F200", "F201"
    ]
    group_col = 'Religion'

    # Create calculator and compute composite distance matrix
    calculator = CFSTDistanceCalculator(df)
    composite_matrix = calculator.create_composite_distance_matrix(questions, group_col)
    composite_matrix.to_csv('composite_distance_matrix.csv', index=False)
    print("Composite Distance Matrix:")
    print(composite_matrix)
