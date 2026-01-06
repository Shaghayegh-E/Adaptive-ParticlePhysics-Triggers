import hdf5plugin
import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# --- Stile CMS-like ---
hep.style.use("CMS")

# --- File input ---
file_name = "Data/Trigger_food_Data.h5"

with h5py.File(file_name, "r") as h5file:
    npv_data = h5file["bkg_Npv"][:]

npv_data = npv_data[npv_data > 0]

# --- Parametri di binning temporale ---
chunk_size = 100000
num_chunks = len(npv_data) // chunk_size
time_fraction = np.linspace(0, 1, num_chunks, endpoint=True)

# --- Calcolo media e incertezza statistica ---
avg_npv, std_npv = [], []
for i in range(num_chunks):
    start, end = i * chunk_size, (i + 1) * chunk_size
    chunk = npv_data[start:end]
    avg_npv.append(np.mean(chunk))
    std_npv.append(np.std(chunk))

# --- Plot ---
fig, ax = plt.subplots(figsize=(7,6))  # più rettangolare e compatto

ax.errorbar(
    time_fraction,
    avg_npv,
    yerr=std_npv,
    fmt='o',
    color='royalblue',
    markersize=7,        # marker più grandi
    markeredgecolor='black',
    capsize=1,
    elinewidth=1,
    label="Average Num Primary Vertices per chunk"
)

# --- Etichette e stile ---
ax.set_xlabel("Time (Fraction of Run)", labelpad=8, loc='center')
ax.set_ylabel("Average Num Primary Vertices",  labelpad=8, loc='center')

ax.tick_params(axis='both',  direction='in', top=True, right=True)
ax.grid(alpha=0.3, linestyle='--')

# --- Layout CMS-like ---
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(nbins=8))


ax.legend(
    frameon=True,
    loc='upper right',
    fontsize=18)


plt.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.15)
fig.savefig("outputs/NPV_vs_time.pdf", dpi=300)
plt.close(fig)
