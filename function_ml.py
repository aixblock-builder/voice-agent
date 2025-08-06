import os
import time

from aixblock_sdk import Client


def connect_project(url, token, project_id):
    # AIXBLOCK_URL = 'http://127.0.0.1:8080'
    # API_KEY = '4b89a6ffb1f14bba6148d9167218e062b4d029dc'
    # PROJECT_ID = 303
    project = None
    try:
        # connect to AIxBlock
        axb = Client(url=url, api_key=token)
        axb.check_connection()

        project = axb.get_project(project_id)
        return project
    except:
        return project


def download_dataset(project, dataset_id, save_path):
    _, filename = project.download_dataset(dataset_id=dataset_id, save_path=save_path)
    if "zip" not in filename:
        filename = filename + ".zip"
    return filename


from pathlib import Path


def count_files_in_directory(output_dir):
    return len(list(Path(output_dir).rglob("*.*")))


def upload_checkpoint(project, version, output_dir):
    total_file = count_files_in_directory(output_dir)
    index = 0
    for root, dirs, files in os.walk(output_dir):
        for idx, file in enumerate(files):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, output_dir)
            folder_name = os.path.dirname(relative_path)
            if index == total_file - 1:
                project.upload_checkpoint(
                    checkpoint=file_path,
                    version=version,
                    path_file=folder_name,
                    send_mail=True,
                )
                index += 1
            else:
                project.upload_checkpoint(
                    checkpoint=file_path, version=version, path_file=folder_name
                )
                index += 1
