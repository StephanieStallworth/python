import zipfile
import pathlib

def make_archive(filepaths, dest_dir):
    dest_path = pathlib.Path(dest_dir, "compressed.zip")
    with zipfile.ZipFile(dest_path, 'w')  as archive: # creates ZipFile type object
        for filepath in filepaths:
            filepath = pathlib.Path(filepath)
            archive.write(filepath, arcname=filepath.name) # extracts only the file name from filepath

# Test function
# If script is executed directly as the main script, call the function
# Otherwise if imported from another file, don't execute this part
if __name__ == "__main__":
    make_archive(filepaths=["bonus14.py","bonus15.py"], dest_dir ="dest")
