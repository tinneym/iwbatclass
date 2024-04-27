import os
import pandas as pd

#-----------------------------------------------------------------------------------
# FILE FOR CLEANING AUDIO FOLDERS
#-----------------------------------------------------------------------------------

# Path to the directory containing audio files
audio_folder_path = "/n/fs/iwbatclass/audio/train"

# Path to the CSV file containing the list of files to keep
csv_file_path = "/n/fs/iwbatclass/audio/train/metadata.csv"

# Column name in the CSV file that contains the file names
file_column_name = 'file_name'

def clean_audio_directory(folder_path, csv_path, column_name):
    # Read the CSV file to get a list of files to keep
    df = pd.read_csv(csv_path)
    keep_files = set(df[column_name].apply(lambda x: os.path.basename(x)))

    # List all files in the directory
    all_files = set(os.listdir(folder_path))

    # Find files that are not listed in the CSV and delete them
    files_to_delete = all_files - keep_files
    for file in files_to_delete:
        print(file)

    for file_name in files_to_delete:
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith('.WAV'):  # Ensure it's an audio file
            os.remove(file_path)
            print(f"Deleted {file_path}")

# Example usage
clean_audio_directory(audio_folder_path, csv_file_path, file_column_name)

print("success")
