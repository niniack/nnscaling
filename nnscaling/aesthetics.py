# aesthetics.py

from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic parameters in one step using Seaborn
sns.set_theme(style="white", context="paper")
sns.axes_style("darkgrid")

# Additional settings for Matplotlib
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.figsize"] = [6, 6]


# Get Seaborn's tab colors
@dataclass
class SeabornColors:
    palette = sns.color_palette("tab20c")
    blue = palette[1]
    orange = palette[6]
    green = palette[9]
    black = palette[16]
    white = palette[19]


# Function to set equal aspect ratio for all plots
def set_equal_aspect(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal", adjustable="datalim")


# # Apply equal aspect ratio to all plots by default
# plt.axis("equal")

# Color palettes
# sns.set_palette('pastel')  # Use pastel colors
# sns.set_palette('dark')    # Use dark colors

# Font settings
plt.rcParams["font.size"] = 10  # Set default font size
plt.rcParams["font.family"] = "sans-serif"  # Set font family

# Grid settings
plt.rcParams["grid.color"] = "gray"  # Customize grid color
# plt.rcParams['grid.linestyle'] = '--'  # Customize grid linestyle

# Save figures with a specific resolution
plt.rcParams["savefig.dpi"] = 300  # High resolution for saved figures
