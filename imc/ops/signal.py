"""
Functions for handling signal intensity in images.
"""

import typing as tp

import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import parmap
from skimage import exposure

import imc.data_models.roi as _roi
from imc.exceptions import cast
from imc.types import DataFrame, Series, Array, Path

FIG_KWS = dict(bbox_inches="tight", dpi=300)


# def check_channel_axis_correlation(
#     arr: Array, channel_labels: tp.Sequence[str], output_prefix: Path
# ) -> DataFrame:
#     # # Plot and regress
#     n, m = get_grid_dims(arr.shape[0])
#     fig, axis = plt.subplots(
#         m, n, figsize=(n * 4, m * 4), squeeze=False, sharex=True, sharey=True
#     )

#     res = list()
#     for channel in range(arr.shape[0]):
#         for axs in [0, 1]:
#             s = arr[channel].mean(axis=axs)
#             order = np.arange(s.shape[0])
#             model = LinearRegression()
#             model.fit(order[:, np.newaxis] / max(order), s)
#             res.append(
#                 [
#                     channel,
#                     axs,
#                     model.coef_[0],
#                     model.intercept_,
#                     pearsonr(order, s)[0],
#                 ]
#             )

#             axis.flatten()[channel].plot(order, s)
#         axis.flatten()[channel].set_title(
#             f"{channel_labels[channel]}\nr[X] = {res[-2][-1]:.2f}; r[Y] = {res[-1][-1]:.2f}"
#         )

#     axis[int(m / 2), 0].set_ylabel("Mean signal along axis")
#     axis[-1, int(n / 2)].set_xlabel("Order along axis")
#     c = sns.color_palette("colorblind")
#     patches = [
#         mpatches.Patch(color=c[0], label="X"),
#         mpatches.Patch(color=c[1], label="Y"),
#     ]
#     axis[int(m / 2), -1].legend(
#         handles=patches,
#         bbox_to_anchor=(1.05, 1),
#         loc=2,
#         borderaxespad=0.0,
#         title="Axis",
#     )
#     fig.savefig(output_prefix + "channel-axis_correlation.svg", **FIG_KWS)

#     df = pd.DataFrame(res, columns=["channel", "axis", "coef", "intercept", "r"])
#     df["axis_label"] = df["axis"].replace(0, "X_centroid").replace(1, "Y_centroid")
#     df["channel_label"] = [x for x in channel_labels for _ in range(2)]
#     df["abs_r"] = df["r"].abs()
#     df.to_csv(output_prefix + "channel-axis_correlation.csv", index=False)
#     return df


def fix_signal_axis_dependency(
    arr: Array, channel_labels: tp.Sequence[str], res: DataFrame, output_prefix: Path
) -> Array:
    # res = pd.read_csv(pjoin("processed", "case_b", "plots", "qc", roi + "_channel-axis_correlation.csv"))
    corr_d = np.empty_like(arr)
    for channel in range(arr.shape[0]):
        r = res.query(f"channel == {channel}")
        x = r.query("axis_label == 'X'")["coef"].squeeze()
        xinter = r.query("axis_label == 'X'")["intercept"].squeeze()
        y = r.query("axis_label == 'Y'")["coef"].squeeze()
        yinter = r.query("axis_label == 'Y'")["intercept"].squeeze()
        # to_reg = pd.DataFrame(arr[channel]).reset_index().melt(id_vars='index').rename(columns=dict(index="X", variable="Y"))

        order = np.arange(arr[channel].shape[0])
        dd = arr[channel]
        m = np.ones_like(dd)
        m = m * (order / max(order) * x) + (xinter)
        m = (m.T * (order / max(order) * y)).T + (yinter)
        ddfix = (dd - m) + dd.mean()
        corr_d[channel] = ddfix

        fig, axis = plt.subplots(1, 7, sharex=True, sharey=False, figsize=(7 * 3, 3 * 1))
        fig.suptitle(channel_labels[channel])
        axis[0].set_title("Original")
        axis[0].imshow(dd)
        axis[1].set_title("Original, equalized")
        axis[1].imshow(exposure.equalize_hist(dd))
        axis[2].set_title("Bias mask")
        axis[2].imshow(m)
        axis[3].set_title("Bias removed")
        axis[3].imshow(ddfix)
        axis[4].set_title("Bias removed, equalized")
        axis[4].imshow(exposure.equalize_hist(ddfix))
        axis[5].set_title("Channel bias")
        axis[5].plot(order, dd.mean(axis=0), label="Original", alpha=0.5)
        axis[5].plot(order, ddfix.mean(axis=0), label="Bias removed", alpha=0.5)
        axis[5].set_xlabel("Position along X axis")
        axis[5].set_ylabel("Signal along X axis")
        axis[5].legend()
        axis[6].set_title("Channel bias")
        axis[6].plot(order, dd.mean(axis=1), label="Original", alpha=0.5)
        axis[6].plot(order, ddfix.mean(axis=1), label="Bias removed", alpha=0.5)
        axis[6].set_xlabel("Position along Y axis")
        axis[6].set_ylabel("Signal along Y axis")
        axis[6].legend()
        for ax in axis[:-2]:
            ax.axis("off")
        fig.savefig(
            output_prefix
            + f"channel-axis_correlation_removal.{channel_labels[channel]}.demonstration.svg",
            **FIG_KWS,
        )
        plt.close("all")
    return corr_d


def channel_stats(roi: _roi.ROI, channels: tp.Sequence[str] = None):
    from skimage.restoration import estimate_sigma

    if channels is None:
        channels = roi.channel_labels.tolist()
    stack = roi._get_channels(channels)[1]
    mask = roi.cell_mask == 0
    res = dict()
    res["wmeans"] = pd.Series(stack.mean(axis=(1, 2)), index=channels)
    res["wstds"] = pd.Series(stack.std(axis=(1, 2)), index=channels)
    res["cmeans"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=mask).mean() for i in range(len(channels))],
        index=channels,
    )
    res["cstds"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=mask).std() for i in range(len(channels))],
        index=channels,
    )
    res["emeans"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=~mask).mean() for i in range(len(channels))],
        index=channels,
    )
    res["estds"] = pd.Series(
        [np.ma.masked_array(stack[i], mask=~mask).std() for i in range(len(channels))],
        index=channels,
    )
    res["noises"] = pd.Series([estimate_noise(ch) for ch in stack], index=channels)
    res["sigmas"] = pd.Series(
        estimate_sigma(np.moveaxis(stack, 0, -1), multichannel=True), index=channels
    )
    return res


def measure_channel_background(
    rois: tp.Sequence[_roi.ROI], plot: bool = True, output_prefix: Path = None
) -> Series:
    from imc.utils import align_channels_by_name
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if plot:
        assert (
            output_prefix is not None
        ), "If `plot` is True, `output_prefix` must be given."

    _channels = pd.DataFrame(
        {r.name: r.channel_labels[~r.channel_exclude.values] for r in rois}
    )
    channels = align_channels_by_name(_channels).dropna().iloc[:, 0].tolist()
    roi_names = [r.name for r in rois]

    res = parmap.map(channel_stats, rois, channels=channels, pm_pbar=True)

    wmeans = pd.DataFrame((x["wmeans"] for x in res), index=roi_names).T
    wstds = pd.DataFrame((x["wstds"] for x in res), index=roi_names).T
    wqv2s = np.sqrt(wstds / wmeans)
    cmeans = pd.DataFrame((x["cmeans"] for x in res), index=roi_names).T
    cstds = pd.DataFrame((x["cstds"] for x in res), index=roi_names).T
    cqv2s = np.sqrt(cstds / cmeans)
    emeans = pd.DataFrame((x["emeans"] for x in res), index=roi_names).T
    estds = pd.DataFrame((x["estds"] for x in res), index=roi_names).T
    eqv2s = np.sqrt(estds / emeans)
    fore_backg: DataFrame = np.log(cmeans / emeans)
    # fore_backg_disp = np.log1p(((cmeans / emeans) / (cmeans + emeans))).mean(1)
    noises = pd.DataFrame((x["noises"] for x in res), index=roi_names).T
    sigmas = pd.DataFrame((x["sigmas"] for x in res), index=roi_names).T

    # Join all metrics
    metrics = (
        wmeans.mean(1)
        .to_frame(name="image_mean")
        .join(wstds.mean(1).rename("image_std"))
        .join(wqv2s.mean(1).rename("image_qv2"))
        .join(cmeans.mean(1).rename("cell_mean"))
        .join(cstds.mean(1).rename("cell_std"))
        .join(cqv2s.mean(1).rename("cell_qv2"))
        .join(emeans.mean(1).rename("extra_mean"))
        .join(estds.mean(1).rename("extra_std"))
        .join(eqv2s.mean(1).rename("extra_qv2"))
        .join(fore_backg.mean(1).rename("fore_backg"))
        .join(noises.mean(1).rename("noise"))
        .join(sigmas.mean(1).rename("sigma"))
    ).rename_axis(index="channel")
    metrics_std = (metrics - metrics.min()) / (metrics.max() - metrics.min())

    if not plot:
        # Invert QV2
        sel = metrics_std.columns.str.contains("_qv2")
        metrics_std.loc[:, sel] = 1 - metrics_std.loc[:, sel]
        # TODO: better decision on which metrics matter
        return metrics_std.mean(1)

    output_prefix = cast(output_prefix)
    if not output_prefix.endswith("."):
        output_prefix += "."

    metrics.to_csv(output_prefix + "channel_background_noise_measurements.csv")
    metrics = pd.read_csv(
        output_prefix + "channel_background_noise_measurements.csv", index_col=0
    )

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.1, 2 * 4), sharex="col")
    axes[0, 0].set_title("Whole image")
    axes[0, 1].set_title("Cells")
    axes[0, 2].set_title("Extracellular")
    for i, (means, stds, qv2s) in enumerate(
        [(wmeans, wstds, wqv2s), (cmeans, cstds, cqv2s), (emeans, estds, eqv2s)]
    ):
        # plot mean vs variance
        mean = means.mean(1)
        std = stds.mean(1) ** 2
        qv2 = qv2s.mean(1)
        fb = fore_backg.mean(1)

        axes[0, i].set_xlabel("Mean")
        axes[0, i].set_ylabel("Variance")
        pts = axes[0, i].scatter(mean, std, c=fb)
        if i == 2:
            div = make_axes_locatable(axes[0, i])
            cax = div.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(pts, cax=cax)

        for channel in means.index:
            lab = "left" if np.random.rand() > 0.5 else "right"
            axes[0, i].text(
                mean.loc[channel], std.loc[channel], channel, ha=lab, fontsize=4
            )
        v = max(mean.max().max(), std.max().max())
        axes[0, i].plot((0, v), (0, v), linestyle="--", color="grey")
        axes[0, i].loglog()

        # plot mean vs qv2
        axes[1, i].set_xlabel("Mean")
        axes[1, i].set_ylabel("Squared coefficient of variation")
        axes[1, i].scatter(mean, qv2, c=fb)
        for channel in means.index:
            lab = "left" if np.random.rand() > 0.5 else "right"
            axes[1, i].text(
                mean.loc[channel], qv2.loc[channel], channel, ha=lab, fontsize=4
            )
        axes[1, i].axhline(1, linestyle="--", color="grey")
        axes[1, i].set_xscale("log")
        # if qv2.min() > 0.01:
        #     axes[1, i].set_yscale("log")
    fig.savefig(output_prefix + "channel_mean_variation_noise.svg", **FIG_KWS)

    fig, axes = plt.subplots(1, 2, figsize=(2 * 6.2, 4))
    p = fore_backg.mean(1).sort_values()
    r1 = p.rank()
    r2 = p.abs().rank()
    axes[0].scatter(r1, p)
    axes[1].scatter(r2, p.abs())
    for i in p.index:
        axes[0].text(r1.loc[i], p.loc[i], s=i, rotation=90, ha="center", va="bottom")
        axes[1].text(
            r2.loc[i], p.abs().loc[i], s=i, rotation=90, ha="center", va="bottom"
        )
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Channel rank")
    axes[1].set_xlabel("Channel rank")
    axes[0].set_ylabel("Cellular/extracellular difference")
    axes[1].set_ylabel("Cellular/extracellular difference (abs)")
    axes[0].axhline(0, linestyle="--", color="grey")
    axes[1].axhline(0, linestyle="--", color="grey")
    fig.savefig(
        output_prefix + "channel_foreground_background_diff.rankplot.svg",
        **FIG_KWS,
    )

    grid = sns.clustermap(
        metrics_std,
        xticklabels=True,
        yticklabels=True,
        metric="correlation",
        cbar_kws=dict(label="Variable (min-max)"),
    )
    grid.fig.savefig(
        output_prefix + "channel_mean_variation_noise.clustermap.svg", **FIG_KWS
    )

    # Invert QV2
    sel = metrics_std.columns.str.contains("_qv2")
    metrics_std.loc[:, sel] = 1 - metrics_std.loc[:, sel]
    # TODO: better decision on which metrics matter
    return metrics_std.mean(1)
