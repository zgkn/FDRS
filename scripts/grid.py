"""The study box and the 0.25 degree ERA5 grid that hotspots and fire weather are both put on."""
import numpy as np

# Southern ASEAN: Sumatra, peninsular Malaysia, Singapore, Borneo, Java, Brunei
SOUTH, NORTH = -11.0, 8.0
WEST, EAST = 94.0, 120.0
STEP = 0.25                      # ERA5 grid spacing; ERA5 grid points sit on multiples of 0.25 degrees


def lat_axis():
    """Cell-centre latitudes, north to south like ERA5 (so that slice(NORTH, SOUTH) selects them)."""
    return np.arange(NORTH, SOUTH - STEP / 2, -STEP)


def lon_axis():
    return np.arange(WEST, EAST + STEP / 2, STEP)


def cell_index(lat, lon):
    """(row, col) of the grid cell whose centre is nearest to each point; -1 outside the box."""
    lat, lon = np.asarray(lat, "float64"), np.asarray(lon, "float64")
    row = np.rint((NORTH - lat) / STEP).astype(int)
    col = np.rint((lon - WEST) / STEP).astype(int)
    inside = (row >= 0) & (row < len(lat_axis())) & (col >= 0) & (col < len(lon_axis()))
    return np.where(inside, row, -1), np.where(inside, col, -1)


def noon_utc_hour(lon):
    """UTC hour (an integer, as ERA5 is hourly) of solar noon at each longitude: local noon is the FWI observation time."""
    return np.rint(12.0 - np.asarray(lon, "float64") / 15.0).astype(int)
