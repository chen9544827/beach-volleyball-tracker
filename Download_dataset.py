# download_datasets.py: 使用 Roboflow API 下载海滩排球和室内排球数据集（YOLOv11 格式）
# 依赖：pip install roboflow

import os
from roboflow import Roboflow

def download_dataset(api_key, workspace, project_name, version_number, save_format):
    """
    下载指定的 Roboflow 数据集版本，并返回本地路径。

    :param api_key: Roboflow API Key
    :param workspace: Roboflow workspace 名称
    :param project_name: Roboflow project 名称
    :param version_number: project 版本号
    :param save_format: 下载格式，例如 'yolov11'
    :return: 本地数据集存储路径
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download(save_format)
    print(f"Downloaded {project_name} v{version_number} to {dataset.location}")
    return dataset.location


def main():
    # 请替换为你的 Roboflow API Key
    API_KEY = "B7ezYh8gELZ4ID6np7dH"
    SAVE_FORMAT = "yolov11"

    # 海滩排球 数据集信息
    BEACH_WORKSPACE = "volleyball-5il1v"
    BEACH_PROJECT = "vollyball-jnlkz"
    BEACH_VERSION = 3

    # 室内排球 数据集信息
    INDOOR_WORKSPACE = "shukur-sabzaliev1"
    INDOOR_PROJECT = "volleyball_v2"
    INDOOR_VERSION = 2

    # 下载数据集
    beach_path = download_dataset(API_KEY, BEACH_WORKSPACE, BEACH_PROJECT, BEACH_VERSION, SAVE_FORMAT)
    indoor_path = download_dataset(API_KEY, INDOOR_WORKSPACE, INDOOR_PROJECT, INDOOR_VERSION, SAVE_FORMAT)

    print("\n=== Datasets downloaded successfully ===")
    print("Beach volleyball dataset path:", beach_path)
    print("Indoor volleyball dataset path:", indoor_path)


if __name__ == "__main__":
    main()
