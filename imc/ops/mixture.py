"""
Functions for mixtures of signal.
"""

import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from imc.types import DataFrame, Series, Array


@tp.overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: tp.Sequence[str],
    red_func: str,
    return_prediction: tp.Literal[False],
) -> int:
    ...


@tp.overload
def get_best_mixture_number(
    x: Series,
    min_mix: int,
    max_mix: int,
    subsample_if_needed: bool,
    n_iters: int,
    metrics: tp.Sequence[str],
    red_func: str,
    return_prediction: tp.Literal[True],
) -> tp.Tuple[int, Array]:
    ...


def get_best_mixture_number(
    x: Series,
    min_mix: int = 2,
    max_mix: int = 6,
    subsample_if_needed: bool = True,
    n_iters: int = 3,
    metrics: tp.Sequence[str] = [
        "silhouette_score",
        "calinski_harabasz_score",
        "davies_bouldin_score",
    ],
    red_func: str = "mean",
    return_prediction: bool = False,
) -> tp.Union[int, tp.Tuple[int, Array]]:
    from sklearn.mixture import GaussianMixture
    import sklearn.metrics

    def get_means(num: Series, pred: tp.Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: tp.Union[Series, Array]) -> Series:
        means = get_means(x, y)
        repl = dict(zip(means.index, range(len(means))))
        y2 = pd.Series(y, index=x.index).replace(repl)
        new_means = get_means(x, y2.values)
        assert all(new_means.index == range(len(new_means)))
        return y2

    xx = x.sample(n=10_000) if subsample_if_needed and x.shape[0] > 10_000 else x

    if isinstance(xx, pd.Series):
        xx = xx.values.reshape((-1, 1))

    mi = range(min_mix, max_mix)
    mixes = pd.DataFrame(index=metrics, columns=mi)
    for i in tqdm(mi):
        mix = GaussianMixture(i)
        # mix.fit_predict(x)
        for f in metrics:
            func = getattr(sklearn.metrics, "davies_bouldin_score")
            mixes.loc[f, i] = np.mean(
                [func(xx, mix.fit_predict(xx)) for _ in range(n_iters)]
            )
        # mixes[i] = np.mean([silhouette_score(x, mix.fit_predict(x)) for _ in range(iters)])
    mixes.loc["davies_bouldin_score"] = 1 / mixes.loc["davies_bouldin_score"]

    # return best
    # return np.argmax(mixes.values()) + min_mix  # type: ignore
    best = mixes.columns[int(getattr(np, red_func)(mixes.apply(np.argmax, 1)))]
    if not return_prediction:
        return best  # type: ignore

    # now train with full data
    mix = GaussianMixture(best)
    return (best, replace_pred(x, mix.fit_predict(x.values.reshape((-1, 1)))))


def get_threshold_from_gaussian_mixture(
    x: Series, y: Series = None, n_components: int = 2
) -> Array:
    def get_means(num: Series, pred: tp.Union[Series, Array]) -> Series:
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x: Series, y: tp.Union[Series, Array]) -> Series:
        means = get_means(x, y)
        repl = dict(zip(means.index, range(len(means))))
        y2 = pd.Series(y, index=x.index).replace(repl)
        new_means = get_means(x, y2.values)
        assert all(new_means.index == range(len(new_means)))
        return y2

    x = x.sort_values()

    if y is None:
        from sklearn.mixture import GaussianMixture  # type: ignore

        mix = GaussianMixture(n_components=n_components)
        xx = x.values.reshape((-1, 1))
        y = mix.fit_predict(xx)
    else:
        y = y.reindex(x.index).values
    y = replace_pred(x, y).values
    thresh = x.loc[((y[:-1] < y[1::])).tolist() + [False]].reset_index(drop=True)
    assert len(thresh) == (n_components - 1)
    return thresh


def get_probability_of_gaussian_mixture(
    x: Series, n_components: int = 2, population=-1
) -> Series:
    from sklearn.mixture import GaussianMixture  # type: ignore

    x = x.sort_values()
    mix = GaussianMixture(n_components=n_components)
    xx = x.values.reshape((-1, 1))
    mix.fit(xx)
    means = pd.Series(mix.means_.squeeze()).sort_values()
    # assert (means.index == range(n_components)).all()
    # order components by mean
    p = mix.predict_proba(xx)[:, means.index]
    # take requested population
    p = p[:, population]
    return pd.Series(p, index=x.index).sort_index()


def fit_gaussian_mixture(
    x: tp.Union[Series, DataFrame], n_mixtures: tp.Union[int, tp.List[int]] = None
) -> tp.Union[Series, DataFrame]:
    # TODO: paralelize
    from sklearn.mixture import GaussianMixture

    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(n_mixtures, int):
        n_mixtures = [n_mixtures] * x.shape[1]
    expr_thresh = x.astype(int)

    def get_means(num, pred):
        return num.groupby(pred).mean().sort_values()

    def replace_pred(x, y):
        means = get_means(x, y)
        repl = dict(zip(range(len(means)), means.index))
        y2 = y.replace(repl)
        new_means = get_means(x, y2)
        assert all(new_means.index == range(len(new_means)))
        return y2

    for i, ch in enumerate(x.columns):
        if n_mixtures is None:
            n_best = get_best_mixture_number(x, return_prediction=False)  # type: ignore[call-tp.overload]
            mix = GaussianMixture(n_best)
        else:
            mix = GaussianMixture(n_mixtures[i])
        _x = x.loc[:, ch]
        x2 = _x.values.reshape((-1, 1))
        mix.fit(x2)
        y = pd.Series(mix.predict(x2), index=x.index, name="class")
        expr_thresh[ch] = replace_pred(_x, y)
    return expr_thresh.squeeze()


def get_population(
    ser: Series, population: int = -1, plot=False, ax=None, **kwargs
) -> pd.Index:
    if population == -1:
        operator = np.greater_equal
    elif population == 0:
        operator = np.less_equal
    else:
        raise ValueError("Chosen population must be '0' (lowest) or '-1' (highest).")

    # Make sure index is unique
    if not ser.index.is_monotonic:
        ser = ser.reset_index(drop=True)

    # Work only in positive space
    xx = ser  # + abs(ser.min())
    done = False
    while not done:
        try:
            n, y = get_best_mixture_number(xx, return_prediction=True, **kwargs)
        except ValueError:  # "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
            continue
        done = True
    print(f"Chosen mixture of {n} distributions.")
    done = False
    while not done:
        try:
            thresh = get_threshold_from_gaussian_mixture(xx, n_components=n)
        except AssertionError:
            continue
        done = True

    sel = operator(xx, thresh.iloc[population]).values

    if plot:
        ax = plt.gca() if ax is None else ax
        sns.distplot(xx, kde=False, ax=ax)
        sns.distplot(xx.loc[sel], kde=False, ax=ax)
        [ax.axvline(q, linestyle="--", color="grey") for q in thresh]
        ax = None
    return sel
