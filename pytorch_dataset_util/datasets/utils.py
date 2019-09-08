# -*- coding: utf-8 -*-
import os
import tarfile
import zipfile
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def _extract_tar(tarfile_path, dst_dir):

    assert tarfile.is_tarfile(tarfile_path), f'{tarfile_path} is not tar file'
    assert os.path.isdir(dst_dir), f'{dst_dir} is not directory.'

    with tarfile.open(tarfile_path, 'r:gz') as f:
        f.extractall(path=dst_dir)


def _extract_zip(zipfile_path, dst_dir):

    assert zipfile.is_zipfile(zipfile_path), f'{zipfile_path} is not zip file'
    assert os.path.isdir(dst_dir), f'{dst_dir} is not directory.'

    with zipfile.ZipFile(zipfile_path) as f:
        f.extractall(path=dst_dir)


def _download_file(url, directory='data', force_download=False):

    disassembled = urlparse(url)
    file_name = os.path.split(disassembled.path)[1]
    file_path = os.path.join(directory, file_name)

    if not force_download and os.path.exists(file_path):
        print(f'{file_path} is already exist.')
        return

    print(f'Downloading file from "{url}"')

    file_size = int(requests.head(url).headers['content-length'])
    res = requests.get(url, stream=True)
    pbar = tqdm(total=file_size, unit='B', unit_scale=True)

    with open(file_path, 'wb') as f:
        for chunk in res.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
