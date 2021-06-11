import matplotlib.pyplot as plt
import wes
import numpy as np

#%% sample fit
from brainsmash.mapgen.eval import sampled_fit

dist_mat_mmap = "/home/daniel/Schreibtisch/bla/distmat.npy"
index_mmap = "/home/daniel/Schreibtisch/bla/index.npy"

kwargs = {'ns': np.round(0.1 * 21030).astype(int), # subsample of rows
          'pv': 75, # percentile of the pariwise distance ditribution at which to truncate during variogram fitting
          'nh': 10, # number of uniformly spaced distances at which to compute variogram
          'knn': 1000, # number of nearest regions to keep in the neighborhood of each region
          'b': None, # gaussian kernel bandwidth
          'deltas': np.arange(0.1,1.0,0.1), # propertions of neighbors to include for smoothing (0,1]
          'kernel': 'exp', # gaussian, exp, invdist, uniform
          'resample': True, # resample to match values from the target brain map          
          }

sampled_fit(contrast, 
            dist_mat_mmap, 
            index_mmap, 
            nsurr=10, 
            **kwargs)

#%%

from brainsmash.mapgen.sampled import Sampled
import matplotlib.pyplot as plt
import numpy as np


def sampled_fit(x, D, index, nsurr=10, **params):
    """
    Evaluate variogram fits for Sampled class.
    Parameters
    ----------
    x : (N,) np.ndarray
        Target brain map
    D : (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of `x`
    index : (N,N) np.ndarray or np.memmap
        See :class:`brainsmash.mapgen.sampled.Sampled`
    nsurr : int, default 10
        Number of simulated surrogate maps from which to compute variograms
    params
        Keyword arguments for :class:`brainsmash.mapgen.sampled.Sampled`
    Returns
    -------
    None
    Notes
    -----
    Generates and shows a matplotlib plot instance illustrating the fit of
    the surrogates' variograms to the target map's variogram.
    """

    # Instantiate surrogate map generator
    generator = Sampled(x=x, D=D, index=index, **params)

    # Simulate surrogate maps
    surrogate_maps = generator(n=nsurr)

    # Compute target & surrogate map variograms
    surr_var = np.empty((nsurr, generator.nh))
    emp_var_samples = np.empty((nsurr, generator.nh))
    u0_samples = np.empty((nsurr, generator.nh))
    for i in range(nsurr):
        idx = generator.sample()  # Randomly sample a subset of brain areas
        v = generator.compute_variogram(generator.x, idx)
        u = generator.D[idx, :]
        umax = np.percentile(u, generator.pv)
        uidx = np.where(u < umax)
        emp_var_i, u0i = generator.smooth_variogram(
            u=u[uidx], v=v[uidx], return_h=True)
        emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
        # Surrogate
        v_null = generator.compute_variogram(surrogate_maps[i], idx)
        surr_var[i] = generator.smooth_variogram(
            u=u[uidx], v=v_null[uidx], return_h=False)

    # # Create plot for visual comparison
    u0 = u0_samples.mean(axis=0)
    emp_var = emp_var_samples.mean(axis=0)

    # Plot target variogram
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.12, 0.15, 0.8, 0.77])
    ax.scatter(u0, emp_var, s=20, facecolor='none', edgecolor='k',
               marker='o', lw=1, label='Empirical')

    # Plot surrogate maps' variograms
    mu = surr_var.mean(axis=0)
    sigma = surr_var.std(axis=0)
    ax.fill_between(u0, mu-sigma, mu+sigma, facecolor='#377eb8',
                    edgecolor='none', alpha=0.3)
    ax.plot(u0, mu, color='#377eb8', label='SA-preserving', lw=1)

    # Make plot nice
    leg = ax.legend(loc=0)
    leg.get_frame().set_linewidth(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_xticklines(), visible=False)
    ax.set_xlabel("Spatial separation\ndistance")
    ax.set_ylabel("Variance")
    plt.show()
