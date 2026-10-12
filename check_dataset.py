
import pandas as pd

DATASET_PATH = "Dataset.csv"

def main():
    df = pd.read_csv(DATASET_PATH)

    print("=" * 55)
    print("NETFLIX AUDIENCE RATING - DATASET ANALYSIS")
    print("=" * 55)

    print(f"\nTotal Records: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nRating Category Distribution:")
    print(df["rating"].value_counts())

    print("\nUnique Rating Categories:")
    print(df["rating"].nunique())

    print("\nDataset Preview:")
    print(df[["title", "type", "release_year", "rating"]].head())

if __name__ == "__main__":
    main()