# Output Directory

This directory is used to store training outputs, checkpoints, and test results.

## Structure

- `fusion/`: Main output directory for fusion model training
  - `best_model.pth`: Best model checkpoint
  - `checkpoint_epoch_*.pth`: Periodic checkpoints
  - `final_model.pth`: Final model checkpoint
  - `test_results/`: Test results directory
    - `test_results.txt`: Test results summary

## Usage

Outputs are automatically saved to this directory during training and testing.

