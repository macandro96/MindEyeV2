"""Script to run dump_recons on multiple delay and avg combinations"""
import os
import subprocess
from itertools import product


def main():
    base_evals_dir = "/scratch/am10150/projects/MindEyeV2/src/evals"
    delays = [0]
    avgs = [1]
    sessions = [3, 5, 10, 20, 37]
    trials = [139, 277, 693, 1386]
    model_names = [f"finetuned_subj01_{sess}_sessions" for sess in sessions] + [f"finetuned_subj01_{trial}_trials" for trial in trials]
    
    
    # Generate all combinations
    combinations = list(product(delays, avgs, model_names))
    
    for delay, avg, model_name in combinations:
        # Construct the folder path
        delay_dir = f"rtnorm_subj1_datascaling"
        
        input_dir = os.path.join(base_evals_dir, delay_dir, f"{model_name}_avg_{avg}")
        
        # Check if the directory exists
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue
        
        print(f"Processing delay={delay}, avg={avg}, model={model_name}")
        print(f"Input directory: {input_dir}")
        
        # Run the dump_recons script
        cmd = [
            "python",
            "dump_recons.py",
            f"--input_dir={input_dir}",
            "--seed_range=1"
        ]
        
        try:
            result = subprocess.run(cmd, cwd="/scratch/am10150/projects/MindEyeV2/src", check=True)
            print(f"Successfully processed delay={delay}, avg={avg}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error processing delay={delay}, avg={avg}: {e}\n")

if __name__ == "__main__":
    main()
