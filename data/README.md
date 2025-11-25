# Data Directory

This directory is used to store CSV files for training and testing datasets.

## Required Files

- `train.csv`: Training dataset CSV file
- `val.csv`: Validation dataset CSV file (optional)
- `test.csv`: Testing dataset CSV file

## CSV Format

The CSV files should contain at least the following columns:
- `image`: Path to the image file (relative to csv_root_dir)
- `split`: Dataset split ("train", "val", or "test")
- `class`: Class label (0 for real, 1 for fake)

## Usage

The training and testing scripts will automatically look for CSV files in this directory.
If CSV files are not found here, they will also check the parent directory's data folder.

