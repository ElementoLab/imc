"""
Functions for image annotations.
"""

import os, json, typing as tp
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import imc.data_models.roi as _roi
from imc.types import DataFrame, Array, Path


def label_domains(
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    export: bool = True,
    domains: tp.Sequence[str] = ["T", "S", "A", "L", "V", "E"],
    **kwargs,
) -> None:
    """
    Draw shapes outying topological domains in tissue.
    This step is done manually using the `labelme` program.

    $ labelme --autosave --labels metadata/labelme_labels.txt
    """
    if export:
        export_images_for_topological_labeling(rois, output_dir, **kwargs)

    labels_f = (output_dir).mkdir() / "labelme_labels.txt"
    with open(labels_f, "w") as handle:
        handle.write("\n".join(domains))
    os.system(f"labelme --autosave --labels {labels_f} {output_dir}")


def export_images_for_topological_labeling(
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    channels: tp.Sequence[str] = ["mean"],
    overwrite: bool = False,
) -> None:
    """
    Export PNGs for labeling with `labelme`.
    """
    for roi in tqdm(rois):
        f = output_dir / roi.name + ".png"
        if not overwrite and f.exists():
            continue
        array = roi._get_channels(channels, minmax=True, equalize=True)[1].squeeze()
        if array.ndim > 2:
            array = np.moveaxis(array, 0, -1)
        matplotlib.image.imsave(f, array)


def collect_domains(
    input_dir: Path, rois: tp.Sequence[_roi.ROI] = None, output_file: Path = None
) -> tp.Dict[str, tp.Dict]:
    if rois is not None:
        roi_names = [r.name for r in rois]

    filenames = list(input_dir.glob("*.json"))
    if rois is not None:
        filenames = [f for f in filenames if f.stem in roi_names]

    topo_annots = dict()
    for filename in tqdm(filenames):
        annot_f = filename.replace_(".png", ".json")
        if not annot_f.exists():
            continue
        with open(annot_f, "r") as handle:
            annot = json.load(handle)
        topo_annots[filename.stem] = annot["shapes"]
    if output_file is not None:
        with open(output_file, "w") as handle:
            json.dump(topo_annots, handle, indent=4)
    return topo_annots


def illustrate_domains(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    output_dir: Path,
    channels: tp.Sequence[str],
    domain_exclude: tp.Sequence[str] = None,
    cleanup: bool = False,
    cmap_str: str = "Set3",
) -> None:
    """
    Illustrate annotated topological domains of each ROI.
    """
    from imc.utils import polygon_to_mask
    from imc.graphics import legend_without_duplicate_labels
    from shapely.geometry import Polygon

    if domain_exclude is None:
        domain_exclude = []

    labels = list(set(geom["label"] for n, j in topo_annots.items() for geom in j))
    label_color = dict(zip(labels, sns.color_palette(cmap_str)))
    label_order = dict(zip(labels, range(1, len(labels) + 1)))
    cmap = plt.get_cmap(cmap_str)(range(len(labels) + 1))
    cmap[0] = [0, 0, 0, 1]

    for roi_name in tqdm(topo_annots):
        shapes = topo_annots[roi_name]
        roi = [r for r in rois if r.name == roi_name][0]
        annot_mask = np.zeros(roi.shape[1:])
        for shape in shapes:
            if shape["label"] in domain_exclude:
                continue
            region = polygon_to_mask(shape["points"], roi.shape[1:][::-1])
            annot_mask[region > 0] = label_order[shape["label"]]

        ar = roi.shape[1] / roi.shape[2]

        fig, axes = plt.subplots(
            1, 2, figsize=(2 * 4, 4 * ar), gridspec_kw=dict(wspace=0, hspace=0)
        )
        axes[0].set(title=roi.name)
        roi.plot_channels(channels, axes=[axes[0]], merged=True)

        shape_types: Counter[str] = Counter()
        for shape in shapes:
            label: str = shape["label"]
            if label in domain_exclude:
                continue
            shape_types[label] += 1
            c = Polygon(shape["points"]).centroid
            axes[1].text(
                c.x,
                c.y,
                s=f"{label}{shape_types[label]}",
                ha="center",
                va="center",
            )
            axes[0].plot(
                *np.asarray(shape["points"] + [shape["points"][0]]).T,
                label=label,
                color=cmap[label_order[label]],
            )

        m = annot_mask == 0
        annot_mask += 1
        annot_mask[m] = 0
        axes[1].imshow(
            annot_mask,
            cmap=cmap_str,
            vmin=1,
            vmax=len(label_color) + 1,
            interpolation="none",
        )
        axes[1].set(title="Manual annotations")
        legend_without_duplicate_labels(axes[0], title="Domain:")
        for ax in axes:
            ax.axis("off")
        fig.savefig(
            output_dir / roi.name + ".annotations.pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    cmd = f"""pdftk
    {output_dir}/*.annotations.pdf
    cat
    output 
    {output_dir}/topological_domain_annotations.pdf"""
    os.system(cmd.replace("\n", " "))

    if cleanup:
        files = output_dir.glob("*.annotations.pdf")
        for file in files:
            file.unlink()


def get_domains_per_cell(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    exclude_domains: tp.Sequence[str] = None,
    remaining_domain: str = "background",
) -> DataFrame:
    """
    Generate annotation of topological domain each cell is contained in
    based on manual annotated masks.

    Parameters
    ----------
    topo_annots: dict
        Dictionary of annotations for each ROI.
    rois: list
        List of ROI objects.
    exclude_domains: list[str]
        Domains to ignore
    exclude_domains: list[str]
        Domains to ignore
    """
    from imc.utils import polygon_to_mask

    if exclude_domains is None:
        exclude_domains = []

    _full_assigns = list()
    for roi_name, shapes in tqdm(topo_annots.items()):
        roi = [r for r in rois if r.name == roi_name][0]
        mask = roi.mask
        cells = np.unique(mask)[1:]
        td_count: tp.Counter[str] = Counter()
        regions = list()
        _assigns = list()
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            if label in exclude_domains:
                continue
            td_count[label] += 1
            points += [points[0]]
            region = polygon_to_mask(points, roi.shape[1:][::-1])
            regions.append(region)
            assign = (
                pd.Series(np.unique(mask[(mask > 0) & region]), name="obj_id")
                .to_frame()
                .assign(
                    roi=roi.name,
                    sample=roi.sample.name,
                    domain_id=f"{label}{td_count[label]}",
                )
            )
            _assigns.append(assign)

        ## if remaining_domain explicetely annotated, skip
        if remaining_domain in td_count:
            print(f"ROI '{roi.name}' has been manually annotated with remaining domains.")
            _full_assigns += _assigns
            continue

        ## add alveolar region as remaining not overlapping with existing regions
        print(f"ROI '{roi.name}' will be annotated with '{remaining_domain}' by default.")
        # remain = ~polygon_to_mask(np.concatenate(polys), roi.shape[1:][::-1]) # <- wrong
        remain = ~np.asarray(regions).sum(0).astype(bool)
        existing = np.sort(pd.concat(_assigns)["obj_id"].unique())
        remain = remain & (~np.isin(mask, existing))

        assign = (
            pd.Series(np.unique(mask[remain]), name="obj_id")
            .drop(0, errors="ignore")
            .to_frame()
            .assign(
                roi=roi.name,
                sample=roi.sample.name,
                domain_id=remaining_domain + "1",
            )
        )
        _assigns.append(assign)
        _full_assigns += _assigns

        # # To visualize:
        # c = (
        #     pd.concat(_assigns)
        #     .set_index(["roi", "obj_id"])["domain_id"]
        #     .rename("cluster")
        # )
        # c = c.str.replace(r"\d", "", regex=True)
        # domains = set(assigns['topological_domain'])
        # fig = roi.plot_cell_types(
        #     c.replace({k + 1: f"{v} - {k}" for k, v in zip(range(len(domains)), domains)})
        # )

    assigns = pd.concat(_full_assigns)
    assigns["topological_domain"] = assigns["domain_id"].str.replace(
        r"\d", "", regex=True
    )

    # reduce duplicated annotations but for cells annotated with background, make that the primary annotation
    id_cols = ["sample", "roi", "obj_id"]
    assigns = (
        assigns.groupby(id_cols).apply(
            lambda x: x
            if (x.shape[0] == 1)
            else x.loc[x["topological_domain"] == remaining_domain, :]
            if (x["topological_domain"] == remaining_domain).any()
            else x
        )
        # .drop(id_cols, axis=1)
        .reset_index(level=-1, drop=True)
    ).set_index(id_cols)

    # make sure there are no cells with more than one domain that is background
    tpc = assigns.groupby(id_cols)["domain_id"].nunique()
    cells = tpc.index
    assert not assigns.loc[cells[tpc > 1]].isin([remaining_domain]).any().any()

    assigns = (
        assigns.reset_index()
        .drop_duplicates(subset=id_cols)
        .set_index(id_cols)
        .sort_index()
    )

    # expand domains
    for domain in assigns["topological_domain"].unique():
        assigns[domain] = assigns["topological_domain"] == domain

    return assigns


@tp.overload
def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    per_domain: tp.Literal[False],
) -> tp.Dict[Path, float]:
    ...


@tp.overload
def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI],
    per_domain: tp.Literal[True],
) -> DataFrame:
    ...


def get_domain_areas(
    topo_annots: tp.Dict[str, tp.Dict],
    rois: tp.Sequence[_roi.ROI] = None,
    per_domain: bool = False,
) -> tp.Union[tp.Dict[Path, float], DataFrame]:
    """
    Get area of airways per image in microns.
    """
    from shapely.geometry import Polygon

    mpp = 1  # scale
    if rois is not None:
        roi_names = [r.name for r in rois]
        topo_annots = {k: v for k, v in topo_annots.items() if k in roi_names}

    _areas = list()
    for roi_name, shapes in tqdm(topo_annots.items()):
        count: tp.Counter[str] = Counter()
        for shape in shapes:
            label = shape["label"]
            count[label] += 1
            a = Polygon(shape["points"]).area
            _areas.append([roi_name, label + str(count[label]), a * mpp])

    areas = (
        pd.DataFrame(_areas)
        .rename(columns={0: "filename", 1: "domain_domain_obj", 2: "area"})
        .set_index("filename")
    )
    if not per_domain:
        areas = areas.groupby("filename")["area"].sum().to_dict()
    return areas


def get_domain_mask(
    topo_annot: tp.Dict,
    roi: _roi.ROI,
    exclude_domains: tp.Sequence[str],
    fill_remaining: str = None,
    per_domain: bool = False,
) -> Array:
    """ """
    import tifffile
    from imc.utils import polygon_to_mask

    _, h, w = roi.shape
    masks = list()
    region_types = list()
    region_names = list()
    count: tp.Counter[str] = Counter()
    for shape in topo_annot:
        shape["points"] += [shape["points"][0]]
        region = polygon_to_mask(shape["points"], (w, h))
        label = shape["label"]
        count[label] += 1
        masks.append(region)
        region_types.append(label)
        region_names.append(label + str(count[label]))

    for_mask = np.asarray(
        [m for l, m in zip(region_types, masks) if l not in exclude_domains]
    ).sum(0)
    if fill_remaining is not None:
        masks += [for_mask == 0]
        region_types += [fill_remaining]
        for_mask[for_mask == 0] = -1
    exc_mask = np.asarray(
        [m for l, m in zip(region_types, masks) if l in exclude_domains]
    ).sum(0)
    mask: Array = (
        ((for_mask != 0) & ~(exc_mask != 0))
        if isinstance(exc_mask, np.ndarray)
        else for_mask
    ).astype(bool)

    if per_domain:
        nmask = np.empty_like(mask, dtype="object")
        for r, l in zip(masks, region_types):
            if l not in exclude_domains:
                nmask[mask & r] = l
        mask = np.ma.masked_array(nmask, mask=nmask == None)

    return mask
