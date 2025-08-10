import pandas as pd
import numpy as np
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")


def load_datasets():
    """Load both datasets"""
    print("Loading datasets...")

    # Load wiki-sim dataset
    wiki_sim = load_dataset(
        "dleemiller/wiki-sim", "pair-score-sampled-mce-lg", split="train"
    )

    # Load stsb dataset
    stsb = load_dataset("sentence-transformers/stsb", split="train")

    # Convert to pandas DataFrames
    wiki_sim_df = wiki_sim.to_pandas()
    stsb_df = stsb.to_pandas()

    print(f"Wiki-sim dataset: {len(wiki_sim_df)} examples")
    print(f"STSB dataset: {len(stsb_df)} examples")

    return wiki_sim_df, stsb_df


def create_bins(n_bins=11):
    """Create fixed 11 bins for 0-10 rating system, centered on integers"""
    if n_bins != 11:
        print(
            f"Warning: n_bins={n_bins} specified, but using fixed 11 bins for 0-10 rating system"
        )

    # Fixed bin edges for 0-10 rating system
    # Bin 0: 0.00-0.05 (centered on 0.0)
    # Bin 1: 0.05-0.15 (centered on 0.1)
    # Bin 2: 0.15-0.25 (centered on 0.2)
    # ...
    # Bin 9: 0.85-0.95 (centered on 0.9)
    # Bin 10: 0.95-1.00 (centered on 1.0)
    bin_edges = [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
    return np.array(bin_edges)


def assign_bin(score, bin_edges):
    """Assign a score to a bin"""
    # Use digitize to find which bin the score belongs to
    # Subtract 1 because digitize returns 1-indexed bins
    bin_idx = np.digitize(score, bin_edges) - 1
    # Handle edge case where score = 1.0 (would be assigned to bin n_bins)
    bin_idx = min(bin_idx, len(bin_edges) - 2)
    return bin_idx


def bin_to_rating(bin_num):
    """Convert bin number (0-10) to human-readable rating"""
    return bin_num


def rating_to_bin_range(rating):
    """Get the score range for a given rating (0-10)"""
    bin_edges = create_bins()
    if 0 <= rating <= 10:
        return (bin_edges[rating], bin_edges[rating + 1])
    else:
        raise ValueError("Rating must be between 0 and 10")


def print_bin_system():
    """Print information about the bin system"""
    print("Bin System (0-10 Rating Scale):")
    print("=" * 40)
    bin_edges = create_bins()
    for i in range(11):
        start, end = bin_edges[i], bin_edges[i + 1]
        center = i / 10.0
        print(f"Rating {i:2d}: {start:.3f} - {end:.3f} (centered on {center:.1f})")
    print("=" * 40)


def sample_from_bins(df, n_samples_per_dataset, n_bins=11, random_state=42):
    """
    Sample uniformly from fixed 11 bins for 0-10 rating system

    Args:
        df: DataFrame with columns sentence1, sentence2, score
        n_samples_per_dataset: Total number of samples to draw from this dataset
        n_bins: Number of bins (fixed at 11 for 0-10 rating system)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with sampled data including bin numbers
    """
    np.random.seed(random_state)

    # Create fixed bins for 0-10 rating system
    bin_edges = create_bins(n_bins)
    n_bins = 11  # Force to 11 bins
    samples_per_bin = n_samples_per_dataset // n_bins

    print(
        f"Sampling {samples_per_bin} examples per bin (11 bins for 0-10 rating system)"
    )

    # Assign each example to a bin
    df = df.copy()
    df["bin"] = df["score"].apply(lambda x: assign_bin(x, bin_edges))

    # Sample from each bin
    sampled_data = []

    for bin_idx in range(n_bins):
        bin_data = df[df["bin"] == bin_idx]

        if len(bin_data) == 0:
            print(
                f"Warning: Bin {bin_idx} (range {bin_edges[bin_idx]:.3f}-{bin_edges[bin_idx + 1]:.3f}, rating {bin_idx}) has no data"
            )
            continue

        if len(bin_data) < samples_per_bin:
            print(
                f"Warning: Bin {bin_idx} (rating {bin_idx}) has only {len(bin_data)} examples, need {samples_per_bin}"
            )
            sampled_bin = bin_data.sample(
                n=len(bin_data), random_state=random_state + bin_idx
            )
        else:
            sampled_bin = bin_data.sample(
                n=samples_per_bin, random_state=random_state + bin_idx
            )

        sampled_data.append(sampled_bin)

    # Combine all bins
    result_df = pd.concat(sampled_data, ignore_index=True)

    # Keep the required columns including bin number
    result_df = result_df[["sentence1", "sentence2", "score", "bin"]]

    return result_df


def create_uniform_sampled_dataset(
    n_wiki_sim=100, n_stsb=100, n_bins=11, random_state=42
):
    """
    Create a uniformly sampled dataset from both sources using fixed 11 bins (0-10 rating system)

    Args:
        n_wiki_sim: Number of samples to draw from wiki-sim dataset
        n_stsb: Number of samples to draw from stsb dataset
        n_bins: Number of bins (fixed at 11 for 0-10 rating system)
        random_state: Random seed for reproducibility

    Returns:
        Combined DataFrame with samples from both datasets including bin numbers
    """

    # Load datasets
    wiki_sim_df, stsb_df = load_datasets()

    # Check score columns and normalize if needed
    print("\nDataset info:")
    print(
        f"Wiki-sim score range: {wiki_sim_df['score'].min():.3f} - {wiki_sim_df['score'].max():.3f}"
    )
    print(
        f"STSB score range: {stsb_df['score'].min():.3f} - {stsb_df['score'].max():.3f}"
    )

    # Normalize STSB scores to 0-1 if they're on a different scale
    if stsb_df["score"].max() > 1.0:
        print("Normalizing STSB scores to 0-1 range...")
        stsb_df = stsb_df.copy()
        stsb_df["score"] = stsb_df["score"] / stsb_df["score"].max()
        print(
            f"STSB score range after normalization: {stsb_df['score'].min():.3f} - {stsb_df['score'].max():.3f}"
        )

    sampled_datasets = []
    n_bins = 11  # Force to 11 bins for 0-10 rating system

    # Sample from wiki-sim dataset
    if n_wiki_sim > 0:
        print(f"\n--- Sampling from Wiki-sim dataset ---")
        wiki_sim_sampled = sample_from_bins(
            wiki_sim_df, n_wiki_sim, n_bins=n_bins, random_state=random_state
        )
        wiki_sim_sampled["source"] = "wiki-sim"
        sampled_datasets.append(wiki_sim_sampled)
        print(f"Wiki-sim sampled: {len(wiki_sim_sampled)} examples")

    # Sample from STSB dataset
    if n_stsb > 0:
        print(f"\n--- Sampling from STSB dataset ---")
        stsb_sampled = sample_from_bins(
            stsb_df, n_stsb, n_bins=n_bins, random_state=random_state + 1000
        )
        stsb_sampled["source"] = "stsb"
        sampled_datasets.append(stsb_sampled)
        print(f"STSB sampled: {len(stsb_sampled)} examples")

    # Combine datasets
    if sampled_datasets:
        final_dataset = pd.concat(sampled_datasets, ignore_index=True)

        # Keep sentence columns, score, and bin number (remove source column)
        final_dataset = final_dataset[["sentence1", "sentence2", "score", "bin"]]

        print(f"\n--- Final Dataset ---")
        print(f"Total examples: {len(final_dataset)}")
        print(f"Score distribution by bin (0-10 rating system):")

        # Show distribution across bins
        bin_edges = create_bins(n_bins)
        bin_counts = final_dataset["bin"].value_counts().sort_index()

        for bin_idx in range(11):
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            count = bin_counts.get(bin_idx, 0)
            rating = bin_idx
            print(
                f"  Bin {bin_idx} (rating {rating}, range {bin_start:.3f}-{bin_end:.3f}): {count} examples"
            )

        return final_dataset
    else:
        print("No datasets to sample from!")
        return pd.DataFrame()


def save_dataset(df, filename="sts_all_uniform.parquet"):
    """Save the dataset to a parquet file"""
    df.to_parquet(filename, index=False)
    print(f"\nDataset saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Print bin system information
    print_bin_system()

    # Configuration
    N_WIKI_SIM = 2750  # 250 per bin * 11 bins
    N_STSB = 2750  # 250 per bin * 11 bins
    N_BINS = 11  # Fixed at 11 for 0-10 rating system

    # Create the uniformly sampled dataset
    dataset = (
        create_uniform_sampled_dataset(
            n_wiki_sim=N_WIKI_SIM, n_stsb=N_STSB, n_bins=N_BINS, random_state=42
        )
        .sample(frac=1)
        .reset_index(drop=True)
    )

    # Display sample of the results
    print(f"\nSample of final dataset:")
    print(dataset.head(10))

    print(f"\nScore statistics:")
    print(dataset["score"].describe())

    print(f"\nBin distribution (0-10 rating system):")
    bin_dist = dataset["bin"].value_counts().sort_index()
    for bin_num in range(11):
        count = bin_dist.get(bin_num, 0)
        print(f"  Rating {bin_num}: {count} examples")

    # Save to file
    save_dataset(dataset, "sts_all_uniform.parquet")
