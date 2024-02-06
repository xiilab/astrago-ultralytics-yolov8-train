import os


def make_unique_folder(path):
    path = path.rstrip("/")
    folder_name = os.path.basename(path)
    folder_path = os.path.dirname(path)
    new_folder_path = path
    new_folder_name = folder_name 
    i = 1
    while os.path.exists(new_folder_path):
        new_folder_name = f"{folder_name}_{i}"
        new_folder_path = os.path.join(folder_path, new_folder_name)
        i += 1
    os.makedirs(new_folder_path)
    return new_folder_path, new_folder_name
