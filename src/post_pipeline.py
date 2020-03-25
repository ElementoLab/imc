from os.path import join as pjoin

import tifffile
import pandas as pd
import numpy as np
import scipy
import scipy.ndimage
import h5py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr


def norm(array):
    # downsample
    if csize == 2 * size:
        array2 = np.empty((array.shape[0], int(array.shape[1] / 2), int(array.shape[2] / 2)))
        for i in range(array2.shape[0]):
            array2[i] = scipy.ndimage.zoom(array[i], 0.5)
        array = array2

    # put the minimum of each channel at zero
    if array.min() > 0:
        array += np.absolute(array.min(axis=(1, 2), keepdims=True))

    # log
    array = np.log10(1 + array)

    # zscore
    m = array.mean(axis=(1, 2), keepdims=True)
    s = array.std(axis=(1, 2), keepdims=True)
    return (array - m) / s


def get_transparent_cmaps(n=3, from_palette="colorblind"):
    import matplotlib
    r = np.linspace(0, 1, 100)
    return [
        matplotlib.colors.LinearSegmentedColormap.from_list(
            '', [p + (c,) for c in r])
        for p in sns.color_palette(from_palette)[:n]]


def get_grid_dims(dims, nstart=None):
    if nstart is None:
        n = min(dims, 1 + int(np.ceil(np.sqrt(dims))))
    else:
        n = nstart
    if (n * n) == dims:
        m = n
    else:
        a = pd.Series(n * np.arange(1, n + 1)) / dims
        m = a[a >= 1].index[0] + 1
    assert n * m >= dims

    if n * m % dims > 1:
        try:
            n, m = get_grid_dims(dims=dims, nstart=n - 1)
        except IndexError:
            pass
    return n, m


def plot_single_channel(d, cmap=None):
    fig, axis = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    axis.imshow(d, cmap=cmap, interpolation="bilinear", rasterized=True)
    axis.axis('off')
    return fig


def plot_overlayied_channels(d, channel_labels, palette=None):

    fig, axis = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    cmaps = get_transparent_cmaps(d.shape[0], from_palette=palette)
    patches = list()
    for i, (m, c) in enumerate(zip(channel_labels, cmaps)):
        x = d[i].squeeze()
        # v = x.std() / 5
        axis.imshow(x, cmap=c, label=m, interpolation="bilinear", rasterized=True, alpha=0.9)
        axis.axis('off')
        patches.append(mpatches.Patch(color=c(256), label=m))
    axis.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return fig


def plot_overlayied_channels_subplots():
    n, m = get_grid_dims(d.shape[0])
    fig, axis = plt.subplots(n, m, figsize=(6 * m, 6 * n), sharex=True, sharey=True, squeeze=False)
    axis = axis.flatten()
    for i, (marker_set, markers) in enumerate(marker_sets.items()):
        patches = list()
        cmaps = get_transparent_cmaps(len(markers))
        for j, (m, c) in enumerate(zip(markers, cmaps)):
            x = d[labels == m, :, :].squeeze()
            v = x.mean() + x.std() * 2
            axis[i].imshow(x, cmap=c, vmin=0, vmax=v, label=m, interpolation="bilinear", rasterized=True)
            axis[i].axis('off')
            patches.append(mpatches.Patch(color=c(256), label=m))
        axis[i].legend(
            handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
            title=marker_set)


def check_channel_axis_correlation(d, output_prefix):
    # # Plot and regress
    res = list()
    n, m = get_grid_dims(d.shape[0])
    fig, axis = plt.subplots(m, n, figsize=(n * 4, m * 4), squeeze=False, sharex=True, sharey=True)
    for channel in range(d.shape[0]):

        for ax in [0, 1]:
            s = d[channel].mean(axis=ax) * 1e4
            s += abs(s.min())
            order = np.arange(s.shape[0])
            axis.flatten()[channel].plot(order, s)
            axis.flatten()[channel].set_yscale("log")

            to_fit = pd.DataFrame([s, order], index=["axis", "signal"]).T
            to_fit['intercept'] = 1
            mod = sm.OLS(to_fit['signal'], to_fit[['axis', 'intercept']])
            fit = mod.fit()
            res.append([channel, ax, fit.params[0], fit.params[1], fit.rsquared_adj, pearsonr(order, s)[0]])
        axis.flatten()[channel].set_title(f"{labels[channel]}\nr[X] = {res[-2][-1]:.2f}; r[Y] = {res[-1][-1]:.2f}")

    axis[int(m / 2), 0].set_ylabel("Log(signal)")
    axis[-1, int(n / 2)].set_xlabel("Order along axis")
    c = sns.color_palette("colorblind")
    patches = [mpatches.Patch(color=c[0], label="X"), mpatches.Patch(color=c[1], label="Y")]
    axis[int(m / 2), -1].legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Axis")
    fig.savefig(output_prefix + "channel-axis_correlation.svg", **fig_kws)

    res = pd.DataFrame(res, columns=['channel', 'axis', 'coef', 'intercept', 'r2', 'r'])
    res['axis_label'] = res['axis'].replace(0, "X").replace(1, "Y")
    res['channel_label'] = [x for x in labels for _ in range(2)]
    res['abs_r'] = res['r'].abs()
    res.to_csv(output_prefix + "channel-axis_correlation.csv", index=False)


fig_kws = dict(bbox_inches="tight", dpi=300)


root_dir = pjoin("processed", "case_b")
# root_dir = pjoin("data", "fluidigm_example_data", "output")

sample = "I13_1285_A1_FL_Jan31_2020"
# sample = "20170906_FluidigmONfinal_SE"


rois = ["s0_p7_r1_a1_ac", "s0_p7_r2_a2_ac", "s0_p7_r4_a4_ac", "s0_p7_r5_a5_ac"]

# roi = "s0_p1_r0_a0_ac"

size = 1000
# size = 800

results_dir = pjoin("processed", "case_b", "plots")


markers = pd.read_csv(
    pjoin("metadata", "panel_markers.panelB.csv"))


for roi in rois:
    metal_order = pd.Series(open(pjoin(root_dir, "tiffs", sample + "_" + roi + "_full.csv"), "r").read().strip().split("\n"), name="Metal Tag")
    metal_order.index.name = "order"
    metal_order = metal_order.reset_index().set_index("Metal Tag")
    labels = markers.set_index("Metal Tag").join(metal_order).sort_values("order").dropna(subset=["order"])['index'].reset_index(drop=True)

    # ilastik_labels = pd.read_csv(pjoin(root_dir, "tiffs", sample + "_" + roi + "_ilastik.csv")).squeeze().tolist()

    files = dict()
    # files['tiff'] = pjoin(root_dir, "ometiff", sample, sample + "_" + roi + ".ome.tiff")
    files['tiff'] = pjoin(root_dir, "tiffs", sample + "_" + roi + "_full.tiff")
    files['channels'] = pjoin(root_dir, "tiffs", sample + "_" + roi + "_ilastik_s2.h5")
    # files['features'] = pjoin(root_dir, "tiffs", sample + "_" + roi + "_ilastik_s2_Features.h5")
    # files['probabilities'] = pjoin(root_dir, "tiffs", sample + "_" + roi + "_ilastik_s2_Probabilities.tiff")
    files['uncertainty'] = pjoin(root_dir, "uncertainty", sample + "_" + roi + '_ilastik_s2_Probabilities_uncertainty.tiff')
    # files['mask'] = pjoin(root_dir, "cpout", sample + "_" + roi + "_ilastik_s2_Probabilities_mask.tiff")

    datas = dict()
    ndatas = dict()
    for ftype, file in files.items():
        csize = size * 2 if "s2" in file else size
        if file.endswith("_mask.tiff"):
            datas[ftype] = tifffile.imread(file) > 0
        elif file.endswith(".ome.tiff"):
            datas[ftype] = tifffile.imread(file, is_ome=True)
        elif file.endswith(".tiff"):
            datas[ftype] = tifffile.imread(file)
        elif file.endswith(".h5"):
            with h5py.File(file, 'r') as f:
                datas[ftype] = np.asarray(f[list(f.keys())[0]])

        if len(datas[ftype].shape) == 3:
            if (min(datas[ftype].shape) == datas[ftype].shape[-1]):
                datas[ftype] = np.moveaxis(datas[ftype], -1, 0)
            datas[ftype] = norm(datas[ftype])
        ndatas[ftype] = datas[ftype]

    # 
    check_channel_axis_correlation(datas["tiff"], pjoin(results_dir, "qc", roi + "_"))

    # Plot all channels
    ftype = "tiff"

    d = ndatas[ftype]
    cmaps = ['viridis', 'plasma', 'magma', 'inferno', 'gist_earth', 'cubehelix']

    kwargs = dict(rasterized=True, cmap="magma", interpolation="bilinear")
    n, m = get_grid_dims(d.shape[0])
    choices = range(d.shape[0])

    if ftype == "tiff":
        assert len(labels) == d.shape[0]
    if ftype == "features":
        choices = np.random.choice(d.shape[0], n * n, replace=False)
    fig, axis = plt.subplots(m, n, sharex=True, sharey=True, figsize=(4 * n, 4 * m), gridspec_kw=dict(wspace=0, hspace=0.1))
    for i, ax in zip(choices, axis.flatten()):
        v = d[i].std() / 2
        if ftype == "tiff":
            ax.set_title(labels[i])
        ax.imshow(d[i], **kwargs, vmin=-v, vmax=v)
        ax.axis('off')
    for ax in axis.flatten()[i:]:
        ax.axis('off')
    fig.savefig(pjoin(results_dir, "segmentation", roi + "_" + ftype + ".svg"), dpi=300, bbox_inches="tight")

    # Channel correlation
    arr_flat = ndatas[ftype].reshape((len(labels), -1))
    xcorr = pd.DataFrame(arr_flat, index=labels).T.corr()
    np.fill_diagonal(xcorr.values, 0)
    for metric in ['euclidean', 'correlation']:
        grid = sns.clustermap(xcorr, cmap="RdBu_r", center=0, metric=metric, cbar_kws=dict(label="Pearson correlation"))
        grid.ax_col_dendrogram.set_title("Pairwise channel correlation")
        grid.savefig(pjoin(results_dir, "qc", roi + "_" + f"channel_pairwise_correlation.{metric}.svg"), **fig_kws)

    # # use channel correlation to pick channels to overlay
    nsubplots = 4
    markers_per_subplot = int(np.ceil(len(labels) / nsubplots))
    c = pd.Series(scipy.cluster.hierarchy.fcluster(grid.dendrogram_col.linkage, nsubplots, criterion='maxclust'), index=xcorr.index)

    marker_sets = dict()
    for sp in range(1, nsubplots + 1):
        marker_sets[sp] = list()
        for i in np.random.choice(np.unique(c), 4, replace=False):
            marker_sets[sp].append(np.random.choice(c[c == i].index, 1, replace=False)[0])

    # # plot channels overlaid based on clustering
    d = ndatas[ftype]
    n, m = get_grid_dims(nsubplots)
    fig, axis = plt.subplots(n, m, figsize=(6 * m, 6 * n), sharex=True, sharey=True, squeeze=False)
    axis = axis.flatten()
    for i, (marker_set, mrks) in enumerate(marker_sets.items()):
        patches = list()
        cmaps = get_transparent_cmaps(len(mrks))
        for j, (m, c) in enumerate(zip(mrks, cmaps)):
            x = d[labels == m, :, :].squeeze()
            v = x.mean() + x.std() * 2
            axis[i].imshow(x, cmap=c, vmin=0, vmax=v, label=m, interpolation="bilinear", rasterized=True)
            axis[i].axis('off')
            patches.append(mpatches.Patch(color=c(256), label=m))
        axis[i].legend(
            handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
            title=marker_set)
    fig.savefig(pjoin(results_dir, "qc", roi + "_" + f"channels_overlayed.random.choice.svg"), **fig_kws)


    # # Plot overlayed channel subset
    structural = ['HLA_Class_I', 'Vimentin', 'DNA2']
    cell_types = ['CD21', 'CD31', 'CD11b', 'Granzyme']
    immune = [r'CD4\(', 'CD8a', 'CD206', 'CD44', 'FoxP3']
    development = ['Vimentin']
    activation = ['ColTypeI', 'Ki-67', 'BCL_6', 'PD_L1']
    marker_sets = {
        "structural": structural,
        # "structural 2": development,
        "cell_types": cell_types,
        "immune": immune,
        "activation": activation}

    fig, axis = plt.subplots(2, 2, figsize=(6 * 2, 6 * 2), sharex=True, sharey=True)
    axis = axis.flatten()
    for i, (marker_set, mrks) in enumerate(marker_sets.items()):
        patches = list()
        cmaps = get_transparent_cmaps(len(mrks))
        for j, (m, c) in enumerate(zip(mrks, cmaps)):
            x = d[labels.str.contains(m), :, :].squeeze()
            # v = x.std() / 5
            axis[i].imshow(x, cmap=c, vmin=-0.2, vmax=x.mean() + 5 * x.std(), label=m, interpolation="bilinear", rasterized=True)
            axis[i].axis('off')
            patches.append(mpatches.Patch(color=c(256), label=m))
        axis[i].legend(
            handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
            title=marker_set)
    fig.savefig(pjoin(results_dir, "qc", roi + "_" + f"channels_overlayed.manual_choice.svg"), **fig_kws)


    # # Plot overlayed channel probabilities
    fig = plot_overlayied_channels(datas["probabilities"], ['nuclei', 'cytoplasm', 'background'], "Set1")
    fig.savefig(pjoin(results_dir, "segmentation", roi + "_" + f"probabilities.overlayed.svg"), **fig_kws)

    # # Plot uncertainty
    fig = plot_single_channel(datas["uncertainty"], "binary")
    fig.savefig(pjoin(results_dir, "segmentation", roi + "_" + f"uncertainty.overlayed.svg"), **fig_kws)


    # # Plot segmentation
    d = datas['mask']
    fig, axis = plt.subplots(1, 1, figsize=(6 * 1, 6 * 1), sharex=True, sharey=True)
    axis.imshow(d, cmap="binary", interpolation="bilinear", rasterized=True)
    axis.axis('off')
    fig.savefig(pjoin(results_dir, "segmentation", roi + "_" + f"segmentation.mask.svg"), **fig_kws)


    # # overlay segmentation on probabilities
    fig = plot_overlayied_channels(datas["probabilities"], ['nuclei', 'cytoplasm', 'background'], "Set1")
    fig.axes[0].imshow(d, cmap="binary_r", interpolation="bilinear", rasterized=True, alpha=0.5)
    fig.savefig(pjoin(results_dir, "segmentation", roi + "_" + f"segmentation.overlayed.svg"), **fig_kws)


# Remove smear effect
ftype = "tiff"
roi = "s0_p7_r4_a4_ac"
res = pd.read_csv(pjoin("processed", "case_b", "plots", "qc", roi + "_channel-axis_correlation.csv"))
for channel in range(datas[ftype].shape[0]):
    r = res.query(f"channel == {channel}")
    x = r.query(f"axis_label == 'X'")['coef'].squeeze()
    xinter = r.query(f"axis_label == 'X'")['intercept'].squeeze()
    y = r.query(f"axis_label == 'Y'")['coef'].squeeze()
    yinter = r.query(f"axis_label == 'Y'")['intercept'].squeeze()
    # to_reg = pd.DataFrame(d[channel]).reset_index().melt(id_vars='index').rename(columns=dict(index="X", variable="Y"))

    dd = d[channel]
    dd = (dd - dd.mean()) / dd.std()

    l = dd.shape[0]
    order = np.arange(l)

    m = np.ones_like(d[channel])
    m = m * (order / l * x)  # + (yinter)
    m = (m.T * (order / l * y)).T  # + (xinter)
    m = (m - m.mean()) / m.std() / 5

    from skimage import exposure
    fig, axis = plt.subplots(1, 7, sharex=True, sharey=False, figsize=(7 * 3, 3 * 1))
    fig.suptitle(labels[channel])
    axis[0].set_title("Original")
    axis[0].imshow(dd)
    axis[1].set_title("Original, equalized")
    axis[1].imshow(exposure.equalize_hist(dd))
    axis[2].set_title("Bias mask")
    axis[2].imshow(m)
    axis[3].set_title("Bias removed")
    axis[3].imshow(dd - m)
    axis[4].set_title("Bias removed, equalized")
    axis[4].imshow(exposure.equalize_hist(dd - m))
    axis[5].set_title("Channel bias")
    axis[5].plot(order, dd.mean(axis=0), label="Original", alpha=0.5)
    axis[5].plot(order, (dd - m).mean(axis=0), label="Bias removed", alpha=0.5)
    axis[5].set_xlabel("Position along X axis")
    axis[5].set_ylabel("Signal along X axis")
    axis[5].legend()
    axis[6].set_title("Channel bias")
    axis[6].plot(order, dd.mean(axis=1), label="Original", alpha=0.5)
    axis[6].plot(order, (dd - m).mean(axis=1), label="Bias removed", alpha=0.5)
    axis[6].set_xlabel("Position along Y axis")
    axis[6].set_ylabel("Signal along Y axis")
    axis[6].legend()
    for ax in axis[:-2]:
        ax.axis('off')
    fig.savefig(pjoin(results_dir, "qc", roi + "_" + f"channel-axis_correlation_removal.{labels[channel]}.demonstration.svg"), **fig_kws)
    plt.close("all")


#


sigy = dd.mean(axis=1)

c = dd + (dd * -y * order)

fig, axis = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(3 * 2, 3 * 1))
axis[0].imshow(dd)
axis[1].imshow(c)


# # exagerate effect 5 time
dde = np.subtract(dd.T, x * 5 * order).T
ex_dd = np.subtract(dde, y * 5 * order)

# # remove effect
ddt = np.subtract(dd.T, -x * 5 * order).T
new_dd = np.subtract(ddt, -y * 5 * order)


dd = np.log1p(dd + abs(dd.min()))
ex_dd = np.log1p(ex_dd + abs(ex_dd.min()))
new_dd = np.log1p(new_dd + abs(new_dd.min()))

fig, axis = plt.subplots(1, 4, sharex=True, sharey=False, figsize=(3 * 4, 3 * 1))
axis[0].plot(order, d[channel].mean(axis=0))
axis[0].plot(order, d[channel].mean(axis=1))
m = dd.mean()
s = dd.std()
axis[1].imshow(dd, vmin=m - s, vmax=m + s)
m = ex_dd.mean()
s = ex_dd.std()
axis[2].imshow(ex_dd, vmin=m - s, vmax=m + s)
m = new_dd.mean()
s = new_dd.std()
axis[3].imshow(new_dd, vmin=m - s, vmax=m + s)
plt.show()


#


#


from skimage.measure import label

plt.imshow(scipy.ndimage.label(d)[0])
plt.imshow(label(d))

# Features

# # Feature extraction (per channel)

# # # Color/Intensity
# # # # Gaussian smoothing
sigmas = [0.3, 0.7, 1.0, 1.6, 3.5, 5, 10]

# # # Edge
# # # # Laplacian of Gaussian

# # # # Gaussian Gradient Magnitude

# # # # Difference of Gaussians

# # # Texture
# # # # Structure of Tensor Eigenvalues

# # # # Hessian of Gaussian Eigenvalues
f = [scipy.ndimage.gaussian_filter(data, s) for s in sigmas]
