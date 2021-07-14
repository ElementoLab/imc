#!/usr/bin/env python

import typing as tp
import shutil
import urllib.request as request
from contextlib import closing
import tarfile
import tempfile
import zipfile
import re

import requests
from urlpath import URL
import tifffile
import numpy as np
import pandas as pd

from imc.types import Path
from imc import Project


DATASET_DB_PATH = Path("~").expanduser() / ".imc" / "demo_datasets"
DATASETS = {
    "jackson_2019_short": "https://wcm.box.com/shared/static/eq1m5j972cf3b5jqoe2vdju3bg9e0r5n",
    "jackson_2019_short_joint": "https://wcm.box.com/shared/static/b8nxku3ywvenghxvvm4wki9znxwbenzb",
    "schwabenland_2021_full": "https://zenodo.org/record/5018260/files/COVID19_brain_all_patients_singletiffs_and_cellmasks.zip?download=1",
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


def _decompress_tar_file(path: Path, output_root: Path = None) -> None:
    """Decompress a tar.xz file."""
    with tarfile.open(path) as f:
        f.extractall(path.parent if output_root is None else output_root)


def get_dataset(dataset_name: str, output_dir: Path = None) -> Project:
    DATASET_DB_PATH.mkdir()

    if dataset_name == "schwabenland_2021":
        return get_schwabenland_2021_data(output_dir)
    dataset_file = DATASET_DB_PATH / dataset_name + ".tar.gz"

    if output_dir is None:
        output_dir = Path(tempfile.TemporaryDirectory().name)

    if not dataset_file.exists():
        _download_file(DATASETS[dataset_name], dataset_file)
    _decompress_tar_file(dataset_file, output_dir)
    return Project(
        name=dataset_name,
        processed_dir=output_dir / dataset_name / "processed",
        subfolder_per_sample="joint" not in dataset_name,
    )


def get_schwabenland_2021_data(output_dir: Path = None) -> Project:
    dataset_name = "schwabenland_2021"
    zip_file_url = (
        "https://zenodo.org/record/5018260/files/"
        "COVID19_brain_all_patients_singletiffs_and_cellmasks.zip"
        "?download=1"
    )

    if output_dir is None:
        output_dir = Path(tempfile.TemporaryDirectory().name).mkdir()

    zip_file = output_dir / dataset_name + "_imc_data.zip"

    if not zip_file.exists():
        _download_file(zip_file_url, zip_file)
    with zipfile.ZipFile(zip_file) as zf:
        zf.extractall(output_dir)
    zip_file.unlink()

    for dir_ in filter(lambda x: x.is_dir(), output_dir.iterdir()):
        name = dir_.name
        _stack = list()
        _channel_names = list()
        for file in dir_.iterdir():
            if "_mask.tiff" in file.as_posix():
                mask = tifffile.imread(file)
                continue
            _stack.append(tifffile.imread(file))
            _channel_names.append(file.stem)
        stack = np.asarray(_stack)
        channel_names = pd.Series(_channel_names)
        annotation = (
            channel_names.str.split("_")
            .apply(pd.Series)
            .set_index(channel_names)
            .rename(columns={0: "marker", 1: "metal"})
        )
        annotation["mass"] = annotation["metal"].str.extract(r"(\d+)")[0].astype(int)
        stack = stack[annotation["mass"].rank().astype(int) - 1]
        annotation = annotation.sort_values("mass")
        annotation.index = annotation.index.str.replace("_", "(") + ")"
        labels = annotation.index.to_series().reset_index(drop=True).rename("channel")

        if "ROI" not in name:
            roi_number = "1"
        else:
            roi_number = re.findall(r"_ROI(\d)_", name)[0]
            name = re.sub(r"_ROI(\d)", "", name)

        od = (output_dir / "processed" / name / "tiffs").mkdir()
        output_prefix = od / name + f"-{roi_number}_full"
        tifffile.imwrite(output_prefix + ".tiff", stack)
        tifffile.imwrite(output_prefix + "_mask.tiff", mask)
        labels.to_csv(output_prefix + ".csv")

        shutil.rmtree(dir_)

    return Project(name=dataset_name, processed_dir=output_dir / "processed")


def get_phillips_2021(output_dir: Path = None) -> Project:
    """
    doi:10.3389/fimmu.2021.687673
    """
    if output_dir is None:
        output_dir = Path(tempfile.TemporaryDirectory().name).mkdir()

    (output_dir / "processed").mkdir()

    dataset_name = "phillips_2021"
    base_url = URL("https://immunoatlas.org")
    group_id = "NOLN"
    project_id = "210614-2"
    cases = [f"NOLN2100{i}" for i in range(2, 10)]
    rois = ["A01"]
    markers = [
        "DNA (Hoechst)",
        "T-bet",
        "GATA3",
        "FoxP3",
        "CD56",
        "TCR-γ/δ",
        "Tim-3",
        "CD30",
        "CCR6",
        "PD-L1",
        "TCR-β",
        "CD4",
        "CD2",
        "CD5",
        "Ki-67",
        "CD25",
        "CD134",
        "α-SMA",
        "CD20",
        "LAG3",
        "MUC-1/EMA",
        "CD11c",
        "PD-1",
        "Vimentin",
        "CD16",
        "IDO-1",
        "CD15",
        "EGFR",
        "VISTA",
        "Granzyme B",
        "CD206",
        "ICOS",
        "CD69",
        "CD45RA",
        "CD57",
        "CD3",
        "HLA-DR",
        "CD8",
        "BCL-2",
        "β-catenin",
        "CD7",
        "CD1a",
        "CD45RO",
        "CCR4/CD194",
        "CD163",
        "CD11b",
        "CD34",
        "Cytokeratin",
        "CD38",
        "CD68",
        "CD31",
        "Collagen IV",
        "CD138",
        "Podoplanin",
        "CD45",
        "MMP-9",
        "MCT",
        "CLA/CD162",
        "DNA (DRAQ5)",
    ]

    for case in cases:
        for roi in rois:
            print(case, roi)
            url = base_url / group_id / project_id / case / roi / f"{case}_{roi}.tif"
            roi = roi.replace("A", "")
            od = (output_dir / "processed" / case / "tiffs").mkdir()
            f = od / f"{case}-{roi}_full.tiff"
            if f.exists():
                continue
            # Somehow the _download_file failed a few times
            _download_file(url.as_posix(), f)
            # resp = url.get()
            # with open(f, "wb") as handle:
            #     handle.write(resp.content)
            pd.Series(markers, name="channel").to_csv(f.replace_(".tiff", ".csv"))

    return Project(name=dataset_name, processed_dir=output_dir / "processed")


def get_allam_2021_data(output_dir: Path = None) -> Project:
    if output_dir is None:
        output_dir = Path(tempfile.TemporaryDirectory().name).mkdir()

    base_url = URL("https://raw.githubusercontent.com/coskunlab/SpatialViz/main/data")
    samples = [
        y[0] + str(y[1]) for code in ["DT", "NT"] for y in zip([code] * 6, range(1, 7))
    ]
    markers = [
        "CD20",
        "CD3",
        "CD4",
        "CD45RO",
        "CD68",
        "CD8a",
        "Col1",
        "DNA1",
        "DNA2",
        "Ecadherin",
        "FoxP3",
        "GranzymeB",
        "Histone3",
        "Ki67",
        "PD1",
        "PDL1",
        "Pankeratin",
        "SMA",
        "Vimentin",
    ]

    for sample in samples:
        mask_url = base_url / "cell_masks" / f"{sample}_cell_Mask.tiff"
        for marker in markers:
            channel_url = base_url / "raw" / sample / f"{sample}_{marker}.tiff"
