import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from scipy.signal import find_peaks, peak_widths
import astropy.units as u
import numpy as np
import pandas as pd
from collections import defaultdict
from astropy.table import QTable, Table, Column
import math
from scipy.io import readsav

# Read the data 
input_filename = 'HD-95735-E140M-odn905010_1.xl_stis'
sav = readsav(input_filename)
wave_to_fit = sav['wave']
flux_to_fit = sav['flux']
plt.plot(wave_to_fit, flux_to_fit)

plt.xlabel('Wavelength (\AA)')
plt.ylabel('Flux (erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)')
plt.show()