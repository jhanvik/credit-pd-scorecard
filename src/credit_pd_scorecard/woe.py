"""Weight of Evidence (WoE) binning and Information Value (IV) for numeric features.

Convention used here (the classic credit-scoring one):

    "good" = no default (target 0), "bad" = default (target 1)
    WoE(bin) = ln( share of all goods in the bin / share of all bads in the bin )
    IV       = sum over bins of (share_good - share_bad) * WoE

A POSITIVE WoE means the bin is safer than average; a NEGATIVE WoE means riskier.
Because a logistic regression for default (target 1) fitted on these WoE values
therefore gets NEGATIVE coefficients, "coefficient signs make sense" later means
"all coefficients are negative".

Everything the binner learns (cut points, WoE values) comes from the rows passed
to `fit`. Fit on the training split only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Common rule-of-thumb IV bands (conventions from credit-scoring practice, not laws).
IV_BANDS = [
    (0.02, "not useful"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (np.inf, "suspiciously strong - check for leakage"),
]


def iv_strength(iv: float) -> str:
    for upper, label in IV_BANDS:
        if iv < upper:
            return label
    return IV_BANDS[-1][1]


def _woe_iv(good: np.ndarray, bad: np.ndarray, smoothing: float) -> tuple[np.ndarray, float]:
    """WoE per bin and total IV, with additive smoothing so empty cells never give inf."""
    k = len(good)
    dist_good = (good + smoothing) / (good.sum() + smoothing * k)
    dist_bad = (bad + smoothing) / (bad.sum() + smoothing * k)
    woe = np.log(dist_good / dist_bad)
    return woe, float(((dist_good - dist_bad) * woe).sum())


@dataclass
class WoeBinner:
    """Learn bins for one numeric feature, then map values to WoE.

    max_bins:      upper limit on the number of bins (missing values excluded).
    min_bin_frac:  every bin must hold at least this share of the training rows.
    min_bad_count: and at least this many defaults.
    monotonic:     "auto" (direction picked from the data), "ascending",
                   "descending", or None (keep the natural, possibly bumpy, shape).
    max_unique:    features with at most this many distinct values start with one
                   bin per value (e.g. the PAY_x status codes).
    """

    max_bins: int = 6
    min_bin_frac: float = 0.05
    min_bad_count: int = 20
    monotonic: str | None = "auto"
    max_unique: int = 12
    n_quantiles: int = 20
    smoothing: float = 0.5

    cuts_: np.ndarray = field(default=None, init=False, repr=False)
    woe_: np.ndarray = field(default=None, init=False, repr=False)
    nan_woe_: float = field(default=0.0, init=False)
    iv_: float = field(default=0.0, init=False)
    bin_table_: pd.DataFrame = field(default=None, init=False, repr=False)

    # ---------------------------------------------------------------- fitting
    def fit(self, x: pd.Series, y: pd.Series) -> "WoeBinner":
        x = pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce")
        y = pd.Series(y).reset_index(drop=True).astype(int)
        if not set(y.unique()) <= {0, 1}:
            raise ValueError("target must be 0/1")
        is_nan = x.isna().to_numpy()
        xv, yv = x.to_numpy(float)[~is_nan], y.to_numpy()[~is_nan]
        if len(xv) == 0:
            raise ValueError("feature has no non-missing values")
        n_total = len(x)
        min_n = int(np.ceil(self.min_bin_frac * n_total))

        cuts = self._initial_cuts(xv)
        good, bad = self._counts(xv, yv, cuts)
        nan_good = int(((y.to_numpy() == 0) & is_nan).sum())
        nan_bad = int(((y.to_numpy() == 1) & is_nan).sum())

        cuts, good, bad = self._merge_small(cuts, good, bad, min_n)
        if self.monotonic is not None and len(good) > 2:
            direction = self._direction(xv, yv)
            cuts, good, bad = self._enforce_monotonic(cuts, good, bad, direction)
        cuts, good, bad = self._limit_bins(cuts, good, bad)

        # Missing values: own bin if big enough, otherwise folded into the closest bin.
        nan_total = nan_good + nan_bad
        nan_target_bin = None
        if 0 < nan_total < min_n or (nan_total > 0 and nan_bad < self.min_bad_count):
            rate = bad / (good + bad)
            nan_rate = nan_bad / nan_total
            nan_target_bin = int(np.argmin(np.abs(rate - nan_rate)))
            good = good.copy()
            bad = bad.copy()
            good[nan_target_bin] += nan_good
            bad[nan_target_bin] += nan_bad
            nan_good = nan_bad = 0
            nan_total = 0

        all_good = np.append(good, nan_good) if nan_total else good
        all_bad = np.append(bad, nan_bad) if nan_total else bad
        woe_all, iv = _woe_iv(all_good.astype(float), all_bad.astype(float), self.smoothing)

        self.cuts_ = cuts
        self.woe_ = woe_all[: len(good)]
        if nan_total:
            self.nan_woe_ = float(woe_all[-1])
        elif nan_target_bin is not None:
            self.nan_woe_ = float(self.woe_[nan_target_bin])
        else:
            self.nan_woe_ = 0.0  # no missing values seen in training: neutral
        self.iv_ = iv
        self.bin_table_ = self._make_table(all_good, all_bad, woe_all, has_nan_bin=bool(nan_total))
        return self

    def _initial_cuts(self, xv: np.ndarray) -> np.ndarray:
        uniq = np.unique(xv)
        if len(uniq) <= self.max_unique:
            return uniq[1:]  # one bin per distinct value
        qs = np.unique(np.quantile(xv, np.linspace(0, 1, self.n_quantiles + 1)[1:-1]))
        qs = qs[qs > uniq[0]]
        if np.mean(xv == uniq[0]) >= self.min_bin_frac and not np.any(qs <= uniq[1]):
            # A big pile of identical minimum values (e.g. "paid nothing") gets its own bin,
            # unless a quantile cut already separates it from the next value.
            qs = np.append([uniq[1]], qs)
        return qs

    @staticmethod
    def _counts(xv: np.ndarray, yv: np.ndarray, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.searchsorted(cuts, xv, side="right")  # bin = number of cuts <= x
        n_bins = len(cuts) + 1
        good = np.bincount(idx[yv == 0], minlength=n_bins)
        bad = np.bincount(idx[yv == 1], minlength=n_bins)
        return good, bad

    @staticmethod
    def _merge(cuts: np.ndarray, good: np.ndarray, bad: np.ndarray, j: int):
        """Merge bin j with bin j+1."""
        new_good = np.concatenate([good[:j], [good[j] + good[j + 1]], good[j + 2 :]])
        new_bad = np.concatenate([bad[:j], [bad[j] + bad[j + 1]], bad[j + 2 :]])
        return np.delete(cuts, j), new_good, new_bad

    def _merge_small(self, cuts, good, bad, min_n):
        """Merge any bin that is too small (rows or defaults) into its closest neighbour."""
        while len(good) > 1:
            n = good + bad
            small = np.where((n < min_n) | (bad < self.min_bad_count))[0]
            if len(small) == 0:
                break
            j = int(small[np.argmin(n[small])])
            rate = bad / np.maximum(n, 1)
            options = [k for k in (j - 1, j) if 0 <= k < len(good) - 1]
            k = min(options, key=lambda m: abs(rate[m] - rate[m + 1]))
            cuts, good, bad = self._merge(cuts, good, bad, k)
        return cuts, good, bad

    def _direction(self, xv: np.ndarray, yv: np.ndarray) -> str:
        if self.monotonic in ("ascending", "descending"):
            return self.monotonic
        corr = pd.Series(xv).corr(pd.Series(yv), method="spearman")
        return "descending" if (not np.isnan(corr) and corr < 0) else "ascending"

    def _enforce_monotonic(self, cuts, good, bad, direction):
        """Merge neighbouring bins until the default rate only moves one way."""
        while len(good) > 2:
            rate = bad / np.maximum(good + bad, 1)
            diff = np.diff(rate)
            violating = np.where(diff < 0)[0] if direction == "ascending" else np.where(diff > 0)[0]
            if len(violating) == 0:
                break
            cuts, good, bad = self._merge(cuts, good, bad, int(violating[0]))
        return cuts, good, bad

    def _limit_bins(self, cuts, good, bad):
        """While there are too many bins, merge the pair that loses the least IV."""
        while len(good) > self.max_bins:
            losses = []
            _, current = _woe_iv(good.astype(float), bad.astype(float), self.smoothing)
            for j in range(len(good) - 1):
                _, g, b = self._merge(cuts, good, bad, j)
                _, merged_iv = _woe_iv(g.astype(float), b.astype(float), self.smoothing)
                losses.append(current - merged_iv)
            cuts, good, bad = self._merge(cuts, good, bad, int(np.argmin(losses)))
        return cuts, good, bad

    def _make_table(self, good, bad, woe, has_nan_bin: bool) -> pd.DataFrame:
        n_numeric = len(self.cuts_) + 1
        labels = []
        for i in range(n_numeric):
            if n_numeric == 1:
                labels.append("all values")
            elif i == 0:
                labels.append(f"< {self.cuts_[0]:.4g}")
            elif i == n_numeric - 1:
                labels.append(f">= {self.cuts_[-1]:.4g}")
            else:
                labels.append(f"[{self.cuts_[i - 1]:.4g}, {self.cuts_[i]:.4g})")
        if has_nan_bin:
            labels.append("missing")
        count = good + bad
        k = len(good)
        dist_good = (good + self.smoothing) / (good.sum() + self.smoothing * k)
        dist_bad = (bad + self.smoothing) / (bad.sum() + self.smoothing * k)
        return pd.DataFrame(
            {
                "bin": labels,
                "count": count,
                "count_share": count / count.sum(),
                "good": good,
                "bad": bad,
                "default_rate": bad / np.maximum(count, 1),
                "woe": woe,
                "iv_part": (dist_good - dist_bad) * woe,
            }
        )

    def iv_on(self, x: pd.Series, y: pd.Series) -> float:
        """IV of the SAME learned bins measured on other rows (e.g. validation).

        If this is much lower than the training IV, the bins were fitted to noise.
        """
        woe = pd.Series(self.transform(x)).round(12).to_numpy()
        counts = pd.crosstab(woe, np.asarray(y).astype(int)).reindex(columns=[0, 1], fill_value=0)
        k = len(counts)
        dist_good = (counts[0] + self.smoothing) / (counts[0].sum() + self.smoothing * k)
        dist_bad = (counts[1] + self.smoothing) / (counts[1].sum() + self.smoothing * k)
        return float(((dist_good - dist_bad) * counts.index.to_numpy(float)).sum())

    # ------------------------------------------------------------ transforming
    def transform(self, x: pd.Series) -> np.ndarray:
        """Map values to their bin's WoE. Missing values get the missing-bin WoE."""
        if self.woe_ is None:
            raise RuntimeError("call fit() first")
        values = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
        out = np.full(len(values), self.nan_woe_, dtype=float)
        ok = ~np.isnan(values)
        out[ok] = self.woe_[np.searchsorted(self.cuts_, values[ok], side="right")]
        return out


def fit_all(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    overrides: dict[str, dict] | None = None,
    **default_kwargs,
) -> tuple[dict[str, WoeBinner], pd.DataFrame]:
    """Fit one binner per feature. Returns the binners and an IV summary table."""
    overrides = overrides or {}
    binners, rows = {}, []
    for col in features:
        kwargs = {**default_kwargs, **overrides.get(col, {})}
        binner = WoeBinner(**kwargs).fit(df[col], df[target])
        binners[col] = binner
        rows.append(
            {
                "feature": col,
                "iv": binner.iv_,
                "strength": iv_strength(binner.iv_),
                "n_bins": len(binner.bin_table_),
            }
        )
    summary = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return binners, summary


def transform_all(
    df: pd.DataFrame, binners: dict[str, WoeBinner], suffix: str = "_woe"
) -> pd.DataFrame:
    """WoE-transform every fitted feature; returns a frame with one column per feature."""
    return pd.DataFrame(
        {f"{col}{suffix}": b.transform(df[col]) for col, b in binners.items()},
        index=df.index,
    )
