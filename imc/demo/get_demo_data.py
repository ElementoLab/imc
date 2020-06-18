#!/usr/bin/env python

import shutil
import urllib.request as request
from contextlib import closing
import tarfile
import tempfile

import requests

from imc.types import Path
from imc import Project


DATASET_DB_PATH = Path("~").expanduser() / ".imc" / "demo_datasets"
DATASETS = {
    "jackson_2019_short": "https://wcm.box.com/shared/static/eq1m5j972cf3b5jqoe2vdju3bg9e0r5n",
    "jackson_2019_short_joint": "https://wcm.box.com/shared/static/b8nxku3ywvenghxvvm4wki9znxwbenzb",
}


def _download_file(url: str, output_path: Path, chunk_size=1024) -> None:
    """
    Download a file and write to disk in chunks (not in memory).

    Parameters
    ----------
    url : :obj:`str`
        URL to download from.
    output_path : :obj:`str`
        Path to file as output.
    chunk_size : :obj:`int`
        Size in bytes of chunk to write to disk at a time.
    """
    if url.startswith("ftp://"):
        with closing(request.urlopen(url)) as r:
            with open(output_path, "wb") as f:
                shutil.copyfileobj(r, f)
    else:
        response = requests.get(url, stream=True)
        with open(output_path, "wb") as outfile:
            outfile.writelines(response.iter_content(chunk_size=chunk_size))


def _decompress_file(path: Path, output_root: Path = None) -> None:
    """Decompress a tar.xz file."""
    with tarfile.open(path) as f:
        f.extractall(path.parent if output_root is None else output_root)


def get_dataset(dataset_name: str, output: Path = None) -> Project:
    DATASET_DB_PATH.mkdir()
    dataset_file = DATASET_DB_PATH / dataset_name + ".tar.gz"

    if output is None:
        output = Path(tempfile.TemporaryDirectory().name)

    if not dataset_file.exists():
        _download_file(DATASETS[dataset_name], dataset_file)
    _decompress_file(dataset_file, output)
    return Project(
        name=dataset_name,
        processed_dir=output / dataset_name / "processed",
        subfolder_per_sample="joint" not in dataset_name,
    )
