import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Replace 'TkAgg' with 'Qt5Agg' or 'Qt4Agg' if preferred
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
import matplotlib.cm as cm

from config import *

def fitting_fn(x, a, b, c, d):
    return d*np.power(x, -1 - 1/a) * np.exp(-c*x **(-1/b))

def normalpdf(x, mu, s2):
    return np.exp(-x**2 / (2*s2)) / np.sqrt(2*np.pi * s2)


combineFolder = "combined_n{}/".format(n)

figsize = (8, 6)
dpi = 300

# Choose which value of theta to proceed
i = 1 # theta100[i]
combineFile = "combinedOutput_n{}_point{}.npy".format(n, theta100[i])
allresults = np.load(combineFolder+combineFile)
#allresults = allresults[0:50000]
print(allresults.shape)
results = allresults[:, 1]
nResults = len(results)
print(nResults)
mu = np.mean(np.real(results))
s2 = np.mean(np.real(results)**2)
print("mu = {}, s2 = {}".format(mu, s2))


plt.figure("Re(Partial sum)", figsize=figsize, dpi=dpi)
plt.hist(np.real(results), bins=500, density=True)

x = np.linspace(-6*np.sqrt(s2), 6*np.sqrt(s2), 100)
plt.plot(x, normalpdf(x, mu, s2), color="black")
plt.xlim([-6*np.sqrt(s2), 6*np.sqrt(s2)])
#plt.ylim([0, 0.8])
plt.tight_layout()

nCopies = 1000
benchmarkData = np.zeros(nResults * nCopies)
for j in range(nCopies):
    benchmarkData[(j*nResults):((j+1)*nResults)] = np.sqrt(0.5*allresults[:, 0])*np.random.normal(0, 1, nResults)
    
hist_truth, bin_edges = np.histogram(benchmarkData, bins=500, density=True)

# 'hist' contains the frequency count (y-values)
# 'bin_edges' contains the edges of the bins (x-values)

# Calculate bin centers for x-values
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(bin_centers, hist_truth, color="red")
#plt.savefig("images/point{}/partialSum1d_120k.png".format(theta100[i]), dpi=dpi)


plt.figure("Vn", figsize=figsize, dpi=dpi)
plt.hist(allresults[:, 0], bins=500, density=True)
plt.xlim([0, 6*np.sqrt(s2)])
#plt.ylim([0, 0.8])
plt.tight_layout()
#plt.savefig("images/point{}/Vn_120k.png".format(theta100[i]), dpi=dpi)




# 3d plot of 2d histogram
x = np.real(results)
y = np.imag(results)

# Only keep the ones that are within 6*np.sqrt(s2)
keep_index = np.logical_and(np.abs(x) <= 6*np.sqrt(s2), np.abs(y) <= 6*np.sqrt(s2))
x = x[keep_index]
y = y[keep_index]
# Compute the 2D histogram
hist, xedges, yedges = np.histogram2d(x, y, bins=160)

# Construct arrays for the bar positions
xpos, ypos = np.meshgrid(xedges[:-1] + np.diff(xedges)/2, yedges[:-1] + np.diff(yedges)/2)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

# Construct arrays with the dimensions for the bars
dx = dy = np.ones_like(zpos) * np.diff(xedges)[0]
dz = hist.flatten()

# Normalize the data for the colormap
norm = colors.Normalize(dz.min(), dz.max())
color_values = cm.viridis(norm(dz))

# Create the 3D plot
fig = plt.figure("Partial sum", figsize=figsize, dpi=dpi)
ax = fig.add_subplot(111, projection='3d')
# Plot the bars
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=color_values)
fig.tight_layout()
fig.subplots_adjust(left=-0.2, right=1.2)
#plt.savefig("images/point{}/partialSum2d_120k.png".format(theta100[i]), dpi=dpi)

# Show the plot
plt.show()
