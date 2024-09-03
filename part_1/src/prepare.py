import zipfile
import os
import argparse

def extract_zip(input_file, output_dir):
    """
    Extracts the contents of a zip file to a specified directory.

    Args:
    input_file (str): Path to the input zip file.
    output_dir (str): Path to the output directory where the contents will be extracted.

    Returns:
    None
    """
    # Check if the input file exists
    if not os.path.isfile(input_file):
        print(f"Error: The file {input_file} does not exist.")
        return

    # Check if the output directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the zip file
    with zipfile.ZipFile(input_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        print(f"Extracted files to {output_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract the contents of a zip file to a specified directory.")
    parser.add_argument('input_file', type=str, help='Path to the input zip file.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where the contents will be extracted.')

    # Parse the arguments
    args = parser.parse_args()

    # Extract the zip file
    extract_zip(args.input_file, args.output_dir)

if __name__ == "__main__":
    main()
