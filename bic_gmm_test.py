'''
This script is modified from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
Some conditional statements are added for drawing different type of covariance.
'''

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn import mixture

print(__doc__)

# Number of samples per component
n_samples = 500

# Generate sample from standard normal distribution with transformation, two components
np.random.seed(0)

TEST='full'

if TEST=='full':
    W1 = np.array([[3., 1.2], [.6, 2.]])
    W2 = np.array([[5., .2], [.2, .5]])
elif TEST=='diag':
    W1 = np.array([[5., 0.], [0., 2.]])
    W2 = np.array([[3., 0.], [0., 6.]]) 
elif TEST=='tied':
    W1 = np.array([[3., 1.2], [.6, 2.]])
    W2 = W1
elif TEST=='spherical':
    W1 = np.array([[3., 0.], [0., 3.]])
    W2 = np.array([[1., 0.], [0., 1.]])

T1 = np.array([0, 0])
T2 = np.array([5., -5])

X = np.r_[np.dot(np.random.randn(n_samples, 2), W1) + T1,
np.dot(np.random.randn(n_samples, 2), W2)+ T2]

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['full', 'diag', 'tied', 'spherical']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
print(clf)
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(3, 1, 1)

for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)

# Plot the winner
splot = plt.subplot(3, 1, 2)
splot.axis('equal')

Y_ = clf.predict(X)
colors = cm.rainbow(np.linspace(0, 1, clf.n_components))

# Select two dimensions to draw a scatter plot.
maximum_values = np.amax(clf.means_,0)
minimum_values = np.amin(clf.means_,0)
mean_diff = maximum_values - minimum_values
sort_idx = np.argsort(mean_diff)
idx1=sort_idx[-1]
idx2=sort_idx[-2]
colors = cm.rainbow(np.linspace(0, 1, clf.n_components))
for i, mean in enumerate(clf.means_):
    subcov = np.zeros((2, 2))
    if clf.covariance_type == 'full':
        subcov[0,0] = clf.covariances_[i,idx1,idx1]
        subcov[0,1] = clf.covariances_[i,idx1,idx2]
        subcov[1,0] = clf.covariances_[i,idx2,idx1]
        subcov[1,1] = clf.covariances_[i,idx2,idx2]
    elif clf.covariance_type == 'tied':
        subcov[0,0] = clf.covariances_[idx1,idx1]
        subcov[0,1] = clf.covariances_[idx1,idx2]
        subcov[1,0] = clf.covariances_[idx2,idx1]
        subcov[1,1] = clf.covariances_[idx2,idx2]
    elif clf.covariance_type == 'diag':
        subcov[0,0] = clf.covariances_[i,idx1]
        subcov[1,1] = clf.covariances_[i,idx2]
    elif clf.covariance_type == 'spherical':
        subcov[0,0] = clf.covariances_[i]
        subcov[1,1] = clf.covariances_[i]
    v, w = linalg.eigh(subcov)
    if not np.any(Y_ == i):
        continue
    plt.scatter(X[Y_ == i, idx1], X[Y_ == i, idx2], .8, color=colors[i])
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse((mean[idx1],mean[idx2]), v[0], v[1], 180. + angle, color=colors[i])
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(.5)
    splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: {} model, {} components, x:idx {}, y:idx {}'.format(clf.covariance_type, clf.n_components,idx1,idx2))
    plt.subplots_adjust(hspace=.35, bottom=.02)
spl3 = plt.subplot(3, 1, 3)
plt.title('Posterior probability prediction of each component for the given data samples')
PY = clf.predict_proba(X)
for i in range(clf.n_components):
    spl3.plot(PY[:,i],color=colors[i])
plt.show()
