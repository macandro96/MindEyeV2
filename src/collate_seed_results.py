import argparse
import os
import pandas as pd

def collate_results(args: argparse.Namespace):
    collated_df = []
    for seed in range(args.seed_range):
        metrics_file = os.path.join(args.input_dir, f"{seed}", "metrics.csv")
        if not os.path.exists(metrics_file):
            print(f"seed={seed} not found in {args.input_dir} folder; skipping..")
        metrics_df = pd.read_csv(metrics_file)
        metrics_df['seed'] = seed
        collated_df.append(metrics_df)

    collated_df = pd.concat(collated_df, ignore_index=True)
    # cols are metric, value, seed
    # compute mean and std for each metric across seeds
    summary_df = collated_df.groupby('metric')['values'].agg(['mean', 'std']).reset_index()
    summary_df.to_csv(args.output_file, index=False)
    print("\n=== Collated Results Summary ===\n")
    print(summary_df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="path to input dir containing seed folders")
    parser.add_argument("--seed_range", type=int, default=5, help="number of seeds to collate")
    parser.add_argument("--output_file", type=str, required=True, help="path to output file for collated results")
    args = parser.parse_args()
    
    collate_results(args)
    
    