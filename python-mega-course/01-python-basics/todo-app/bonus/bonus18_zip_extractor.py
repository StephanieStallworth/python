import zipfile

def extract_archive(archivepath, dest_dir):
    with zipfile.ZipFile(archivepath, 'r') as archive:
        archive.extractall(dest_dir)

# Right click file in PyCharm file explorer > Copy Path Reference > Absolute Path
# Add escape backlash for Windows path names
if __name__ == "__main__":
    extract_archive("C:\\Users\\sstallworth\\Desktop\\pythonProject\\bonus\\File Extractor\\compressed.zip",
                    "/Module1/bonus\\File Extractor")
