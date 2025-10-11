import pandas as pd
import numpy as np
from itertools import combinations
from tqdm.auto import tqdm
import argparse
import json


class CFSTDistanceCalculator:
    """
    Calculates CFST-based cultural distance matrices.

    Supports two methods:
        - "heterozygosity": uses H = 1 - sum(p_i^2) (generalized for multi-category data).
        - "variance": ratio of between-group to total variance (on scaled values).

    Attributes:
        df (pd.DataFrame): Input survey dataset.
        scaling (str): Scaling method ("minmax", "zscore", or None).
        method (str): CFST calculation method ("heterozygosity" or "variance").
    """

    def __init__(self, df: pd.DataFrame, scaling: str = None, method: str = "heterozygosity"):
        self.df = df
        self.scaling = scaling
        self.method = method

    # ---------------- CFST METHODS ----------------
    @staticmethod
    def _cfst_heterozygosity(df: pd.DataFrame, value_col: str, group_col: str) -> float:
        """Generalized heterozygosity-based CFST calculation."""
        working_df = df[[value_col, group_col]].dropna()

        if working_df.empty or working_df[group_col].nunique() < 2:
            return np.nan

        # Total heterozygosity
        p_total = working_df[value_col].value_counts(normalize=True)
        H_T = 1 - np.sum(p_total**2)
        if H_T == 0:
            return 0.0

        # Average within-group heterozygosity
        H_G = 0
        for _, g in working_df.groupby(group_col):
            p_g = g[value_col].value_counts(normalize=True)
            H_G += 1 - np.sum(p_g**2)
        H_G /= working_df[group_col].nunique()

        # Fixation index
        Fst = (H_T - H_G) / H_T
        return float(max(0.0, min(1.0, Fst)))  # clamp to [0,1]

    @staticmethod
    def _cfst_variance(df: pd.DataFrame, value_col: str, group_col: str) -> float:
        """Variance-ratio based CFST calculation (using scaled values)."""
        working_df = df[[value_col, group_col]].dropna()
        if working_df.empty or working_df[group_col].nunique() < 2:
            return np.nan

        # Min-max scale this column
        min_val, max_val = working_df[value_col].min(), working_df[value_col].max()
        if max_val == min_val:
            return 0.0
        working_df["_scaled"] = (working_df[value_col] - min_val) / (max_val - min_val)

        total_variance = working_df["_scaled"].var(ddof=1)
        if pd.isna(total_variance) or total_variance == 0:
            return 0.0

        group_means = working_df.groupby(group_col)["_scaled"].mean()
        between_group_variance = group_means.var(ddof=1)

        cfst_score = between_group_variance / total_variance
        return float(max(0.0, min(1.0, cfst_score)))

    def calculate_cfst(self, df: pd.DataFrame, value_col: str, group_col: str) -> float:
        """Dispatch to the chosen CFST method."""
        if self.method == "heterozygosity":
            return self._cfst_heterozygosity(df, value_col, group_col)
        elif self.method == "variance":
            return self._cfst_variance(df, value_col, group_col)
        else:
            raise ValueError("Method must be 'heterozygosity' or 'variance'")

    # ---------------- PAIRWISE + MATRICES ----------------
    def calculate_pairwise_cfst(self, value_col: str, group_col: str) -> dict:
        """Calculate CFST scores for all unique pairs of groups for a single question."""
        results = {}
        groups = self.df[group_col].dropna().unique()
        group_pairs = combinations(groups, 2)

        for g1, g2 in tqdm(group_pairs, desc=f"Pairwise for '{value_col}'", leave=False):
            pair_df = self.df[self.df[group_col].isin([g1, g2])]
            score = self.calculate_cfst(pair_df, value_col, group_col)
            results[(g1, g2)] = score

        return results

    def create_single_value_matrix(self, value_col: str, group_col: str) -> pd.DataFrame:
        """Distance matrix for one question."""
        pairwise_scores = self.calculate_pairwise_cfst(value_col, group_col)
        groups = sorted(self.df[group_col].dropna().unique())
        distance_matrix = pd.DataFrame(index=groups, columns=groups, dtype=float)

        for (g1, g2), score in pairwise_scores.items():
            distance_matrix.loc[g1, g2] = score
            distance_matrix.loc[g2, g1] = score

        np.fill_diagonal(distance_matrix.values, 0)
        return distance_matrix

    def create_composite_distance_matrix(self, value_cols: list, group_col: str) -> pd.DataFrame:
        """Composite distance matrix = mean of per-question matrices."""
        matrix_list = []
        for value_col in tqdm(value_cols, desc="Processing questions", leave=False):
            single_matrix = self.create_single_value_matrix(value_col, group_col)
            if not single_matrix.isna().all().all():
                matrix_list.append(single_matrix.to_numpy())

        if not matrix_list:
            print("Warning: No valid distance matrices could be calculated.")
            return pd.DataFrame()

        mean_matrix_values = np.nanmean(np.array(matrix_list), axis=0)
        groups = sorted(self.df[group_col].dropna().unique())
        return pd.DataFrame(mean_matrix_values, index=groups, columns=groups)

    def calculate_dimensional_matrices(self, dimensions: dict, group_col: str) -> dict:
        """Calculate a CFST distance matrix for each dimension."""
        dimensional_matrices = {}
        for dim_name, questions in tqdm(dimensions.items(), desc="Processing Dimensions"):
            print(f"Calculating distance matrix for dimension: {dim_name}")
            dim_matrix = self.create_composite_distance_matrix(questions, group_col)
            dimensional_matrices[dim_name] = dim_matrix
        return dimensional_matrices


# ---------------- CLI ENTRYPOINT ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate CFST distance matrices from survey data.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing the data.")
    parser.add_argument("dimensions_path", type=str, help="Path to the JSON file with question dimensions.")
    parser.add_argument("--group_col", type=str, default="Religion", help="Column containing group identifiers.")
    parser.add_argument("--scaling", type=str, choices=["minmax", "zscore"], default=None, help="Scaling method.")
    parser.add_argument("--method", type=str, choices=["heterozygosity", "variance"], default="heterozygosity",
                        help="CFST calculation method.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv_path)

    # Load dimensions
    with open(args.dimensions_path, "r") as f:
        dimensions = json.load(f)

    # Instantiate calculator
    calculator = CFSTDistanceCalculator(df, scaling=args.scaling, method=args.method)

    # 1. Dimensional matrices
    dimensional_matrices = calculator.calculate_dimensional_matrices(dimensions, args.group_col)
    for dim_name, matrix in dimensional_matrices.items():
        if not matrix.empty:
            matrix.to_csv(f"dimensional_matrix_{dim_name}.csv")
            print(f"Saved dimensional matrix for '{dim_name}'")

    # 2. Summary (mean distance per religion per dimension)
    summary_data = {}
    for dim_name, matrix in dimensional_matrices.items():
        if not matrix.empty:
            summary_data[dim_name] = matrix.mean(axis=1)
    if summary_data:
        dimensional_summary_df = pd.DataFrame(summary_data).T
        dimensional_summary_df.to_csv("dimensional_cfst_summary.csv")
        print("Saved dimensional CFST summary.")

    # 3. Composite matrix (all questions)
    all_questions = [q for qs in dimensions.values() for q in qs]
    composite_matrix = calculator.create_composite_distance_matrix(all_questions, args.group_col)
    if not composite_matrix.empty:
        composite_matrix.to_csv("composite_distance_matrix.csv")
        print("Saved composite distance matrix.")
