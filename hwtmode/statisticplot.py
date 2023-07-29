import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn import metrics
from scikitplot.metrics import plot_roc
from tqdm import tqdm

# Use scikitplot.metrics.plot_roc - nice because has adds other blended ROC curves in it.
def plot_roc(y_true, y_probas, **kwargs):
    # Reorder y_probas columns to match order of categories in y_true
    # plot_roc() expects alphabetical, but labels are probably ordered categories in a different order
    new_column_order = np.argsort(y_true.cat.categories)
    y_probas_new = y_probas[:, new_column_order]
    return plot_roc(y_true, y_probas_new, **kwargs)

def bss(obs, fcst):
    bs = np.mean((fcst - obs) ** 2)
    climo = np.mean((obs - np.mean(obs)) ** 2)
    return 1.0 - bs / climo


def reliability_diagram(ax, obs, fcst, thresh, n_bins=10, **kwargs):
    for o_thresh in thresh:
        label = f"{o_thresh}+ events"
        # calibration curve
        true_prob, fcst_prob = calibration_curve(obs>=o_thresh, fcst, n_bins=n_bins)
        bss_val = bss(obs>=o_thresh, fcst)
        base_rate = (obs>=o_thresh).mean()  # base rate
        s, = ax.plot(fcst_prob, true_prob, "s-", label="%s  bss:%1.4f" % (label, bss_val), **kwargs)
        for x, f in zip(fcst_prob, true_prob):
            if np.isnan(f): continue  # avoid TypeError: ufunc 'isnan' not supported...
            # label reliability points
            ax.annotate("%1.3f" % f, xy=(x, f), xytext=(0, 1), textcoords='offset points', 
                    va='bottom', ha='center', fontsize='xx-small')

        noskill_line = ax.plot([0, 1], [base_rate / 2, (1 + base_rate) / 2], linewidth=0.3, alpha=0.7,
                label="no skill", color=s.get_color())
        baserateline = ax.axhline(y=base_rate, label=f"base rate {base_rate:.4f}", linewidth=0.5,
                                  linestyle="dashed", dashes=(9, 9), color=s.get_color())
        baserateline_vertical = ax.axvline(x=base_rate, linewidth=0.5, linestyle="dashed", 
                dashes=(9, 9), color=s.get_color())

    # If it is not a child already add perfectly calibrated line
    perfect_label = "perfect"
    has_perfect = perfect_label in [x.get_label() for x in ax.get_lines()]
    if not has_perfect:
        ax.plot([0, 1], [0, 1], "k:", alpha=0.7, label=perfect_label)
    ax.set_ylabel("observed fraction of positives")
    ax.set_title("reliability diagram")
    ax.legend(loc="upper left", fontsize="x-small")
    ax.set_xlim((0, 1))

    return ax



def ROC_curve(ax, obs, fcst, label="", sep=0.1, plabel=True, fill=False):
    """
    Generate a ROC curve from the contingency table by calculating the probability of detection (TP/(TP+FN)) and the
    probability of false detection (FP/(FP+TN)).
    """
    
    auc = None
    if obs.all() or (obs == False).all():
        logging.info("obs are all True or all False. ROC AUC score not defined")
        r = ax.plot([0], [0], marker="+", linestyle="solid", label=label)
    elif obs is None or fcst is None:
        # placeholders
        r = ax.plot([0], [0], marker="+", linestyle="solid", label=label)
    else:
        # ROC auc with threshold labels separated by sep
        auc = metrics.roc_auc_score(obs, fcst)
        logging.debug(f"auc {auc}")
        pofd, pody, thresholds = metrics.roc_curve(obs, fcst)
        r = ax.plot(pofd, pody, marker="+", markersize=1 / np.log10(len(pofd)), linestyle="solid",
                    label="%s  auc:%1.4f" % (label, auc))
        if fill:
            auc = ax.fill_between(pofd, pody, alpha=0.2)
        if plabel:
            old_x, old_y = 0., 0.
            for x, y, s in zip(pofd, pody, thresholds):
                if ((x - old_x) ** 2 + (y - old_y) ** 2.) ** 0.5 > sep:
                    # label thresholds on ROC curve
                    ax.annotate("%1.3f" % s, xy=(x, y), xytext=(0, 1), textcoords='offset points', 
                            va='baseline', ha='left', fontsize='xx-small')
                    old_x, old_y = x, y
                else:
                    logging.debug(
                        f"statisticplot.ROC_curve: toss {x},{y},{s} annotation. Too close to last label.")
    ax.set_title("receiver operating characteristic curve")
    no_skill_label = "no skill:0.5"
    # If it is not a child already add perfectly calibrated line
    has_no_skill_line = no_skill_label in [x.get_label() for x in ax.get_lines()]
    if not has_no_skill_line:
        no_skill_line = ax.plot([0, 1], [0, 1], "k:", alpha=0.7, label=no_skill_label)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel("prob of false detection")
    ax.set_ylabel("prob of detection")
    ax.legend(loc="lower right", fontsize="x-small")
    return r, auc


map_crs = ccrs.LambertConformal(central_longitude=-95, standard_parallels=(25,25))
def make_map(bbox = [-121, -72, 22, 50], projection=map_crs, draw_labels=True, scale=1):
    fig, ax = plt.subplots(figsize=(10, 7),
            subplot_kw=dict(projection=projection))
    ax.set_extent(bbox)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.25*scale)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.25*scale)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.05*scale)
    ax.add_feature(cfeature.LAKES.with_scale('50m'), edgecolor='k', linewidth=0.25*scale, facecolor='k', alpha=0.05)
    gl = ax.gridlines(draw_labels=draw_labels, alpha=0.5, linewidth=0.25*scale)
    gl.top_labels = gl.right_labels = False
    return fig, ax

def far(obs, fcst):
    """
    false alarm ratio
    fp / (tp+fp)
    """
    tp =  (obs & fcst).sum()
    fp = (~obs & fcst).sum()
    return fp / (tp+fp) if tp+fp else np.nan

def pod(obs, fcst):
    """
    probability of detection
    tp / (tp+fn)
    """
    tp = (obs &  fcst).sum()
    fn = (obs & ~fcst).sum()
    return tp / (tp+fn) if tp+fn else np.nan

def count_histogram(ax, fcst, label=None, n_bins=10, count_label=True):
    """
    histogram of forecast probability
    """
    ax.set_xlabel("forecasted probability")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(label)
    if fcst is None: return None
    # Histogram of counts
    counts, bins, patches = ax.hist(fcst, bins=n_bins, histtype='step', lw=2)
    logging.debug(f"counts={counts}")
    logging.debug(bins)
    logging.debug(patches)
    if count_label:
        for count, x in zip(counts, bins):
            # label counts
            ax.annotate(str(int(count)), xy=(x, count),
                        xytext=(0, -1), textcoords='offset points', va='top', ha='left', fontsize='xx-small')
    ax.set_xlim((0, 1))
    return ax

def performance_diagram(ax, obs, fcst, thresh, pthresh):
    """
    performace diagram
    xaxis = 1-far 
    yaxis = prob of detection
    where far = fp / (tp+fp)
    """
    bias_lines = [0.2,0.5,0.8,1,1.3,2,5]
    csi_lines = np.arange(0.1,1.0,0.1)
    bias_pts =[ [ sr*b for sr in [0,1.0] ] for b in bias_lines ] # compute pod values for each bias line
    csi_pts = np.array([ [ csi/(csi-(csi/sr)+1) for sr in np.arange(0.011,1.01,0.005) ] for csi in csi_lines ]) # compute pod values for each csi line
    csi_pts = np.ma.masked_array(csi_pts, mask=(csi_pts<0.05))
        
    # add bias and CSI lines to performance diagram 
    for r in bias_pts: ax.plot([0,1], r, color='0.5', linestyle='dashed', lw=0.8, alpha=0.6)
    for r in csi_pts: ax.plot(np.arange(0.01,1.01,0.005), r, color='0.5', linestyle='solid', linewidth=0.5)
    for x in [b for b in bias_lines if b<=1]: ax.text(1.002, x, x, va='center', ha='left', fontsize='x-small', color='0.5')
    for x in [b for b in bias_lines if b>1]: ax.text(1/x, 1, x, va='bottom', ha='center', fontsize='x-small', color='0.5')

    # axes limits, labels
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_title('performance diagram')
    ax.set_xlabel('1 - false alarm ratio')
    ax.set_ylabel('probability of detection')
    for o in thresh:
        x = [1-far(obs >=o, fcst >= p) for p in pthresh]
        y = [pod(obs >= o, fcst >= p) for p in pthresh]
        ax.plot(x,y, label=o)
        for x, y, p in zip(x,y,pthresh):
            if ~np.isnan(x) and ~np.isnan(y):
                ax.text(x, y, p)
    
    ax.legend(title=thresh.name)
    return ax

def stat_plots(obs, fcst, thresh=pd.Series(np.arange(1,10), name=f"obs threshold"), 
               pthresh=pd.Series(np.arange(0,1.1,.2), name=f"prob threshold"), 
               o_thresh=10, sep=0.01, suptitle=None, fig=None):
    if fig is None:
        ncols, nrows = 2,2
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*4,nrows*4))
        axes = iter(axes.flatten())
    else:
        logging.info(f"use old figure {fig} with {fig.get_axes()}")
        axes = iter(fig.get_axes())
    

    label = f"{o_thresh}+ events"
    reliability_diagram(next(axes), obs, fcst, thresh)
    
    performance_diagram(next(axes), obs, fcst, thresh, pthresh=pthresh)

    """
    logging.info("pod")
    ax = next(axes)
    df=pd.DataFrame([[pod(obs>=o,fcst>=p) for p in pthresh] for o in tqdm(thresh)],index=thresh, columns=pthresh )
    df.plot(ax=ax, title="probability of detection")
    ax.set_xscale("log")
    ax.set_ylim((0,1))
    """

    logging.info("count histogram")
    count_histogram(next(axes), fcst, label=label, count_label=False)

    logging.info("ROC curve")
    ROC_curve(next(axes), obs>=o_thresh, fcst, label=label, sep=sep)

    """
    logging.info("brier skill score")
    ax = next(axes)
    df = pd.Series([bss(obs > o, fcst) for o in tqdm(thresh)], index=thresh)
    df.plot(ax=ax, title="brier skill score", label=label)
    ax.set_xscale("log")
    ax.set_ylim((-0.3, 0.6))
    """
    
    [a.grid(visible=True, lw=0.5, linestyle="dotted") for a in fig.get_axes()]
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    return fig

