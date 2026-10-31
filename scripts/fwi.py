"""Canadian Forest Fire Weather Index (FWI) System moisture codes and indices, vectorised with numpy.

A transcription of the C version (Table 2) of Wang, Anderson and Suddaby (2015), "Updated source code for calculating fire
danger indices in the Canadian Forest Fire Weather Index System", Information Report NOR-X-424. The equation numbers in the
comments are those of Van Wagner and Pickett (1985) as cited there. tests/test_fwi.py reproduces the 49-day test data of the
report's Table 7.

Every function takes numpy arrays (or floats) of any shape, so one call advances a whole grid by one day.

Inputs are the noon local standard time values of the day:
    T   temperature, deg C
    H   relative humidity, %
    W   wind speed, km/h
    Ro  rain in the open over the previous 24 h, mm

The fire weather indices FFMC, DMC and DC are carried from day to day. ISI and BUI follow from them (and the wind). The final
FWI is NOT used in this project, but is kept so the module can be checked against the report.

Day length: the monthly tables of the report (EL_MONTHLY, LF_MONTHLY) are for temperate latitudes. For the equatorial region
de Groot et al. (2006) replaced them by constants, 9.0 drying hours for the DMC (`le`) and a day-length factor of 1.39 for the DC
(`lf`). Both are arguments here and may be arrays, so a grid can mix settings.
"""
import numpy as np

# Report tables 1-3, month 1..12 (temperate latitudes)
EL_MONTHLY = (6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0)
LF_MONTHLY = (-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6)

# Equatorial constants of de Groot et al. (2006), Mitig. Adapt. Strat. Glob. Change 12:165-180
LE_EQUATORIAL = 9.0
LF_EQUATORIAL = 1.39

# initial values of the report's test run; a real run should spin up long enough for the starting values to be forgotten
FFMC_START, DMC_START, DC_START = 85.0, 6.0, 15.0


def ffmc(T, H, W, Ro, Fo):
    """Fine Fuel Moisture Code of today from today's weather and yesterday's code `Fo`."""
    T, H, W, Ro, Fo = (np.asarray(x, dtype="float64") for x in (T, H, W, Ro, Fo))
    mo = 147.2 * (101.0 - Fo) / (59.5 + Fo)                                              # Eq. 1
    rain = Ro > 0.5
    rf = np.where(rain, Ro - 0.5, 1.0)                                                   # Eq. 2
    mr = mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))     # Eq. 3a
    mr = np.where(mo > 150.0, mr + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), mr)         # Eq. 3b
    mo = np.where(rain, np.minimum(mr, 250.0), mo)
    ed = 0.942 * H ** 0.679 + 11.0 * np.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))   # Eq. 4
    ew = 0.618 * H ** 0.753 + 10.0 * np.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))   # Eq. 5
    thermal = 0.581 * np.exp(0.0365 * T)
    ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (1.0 - (H / 100.0) ** 8)                       # Eq. 6a
    kl = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8)   # Eq. 7a
    m_dry = ed + (mo - ed) * 10.0 ** (-ko * thermal)                                     # Eq. 6b, 8
    m_wet = ew - (ew - mo) * 10.0 ** (-kl * thermal)                                     # Eq. 7b, 9
    m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))
    f = 59.5 * (250.0 - m) / (147.2 + m)                                                 # Eq. 10
    return np.clip(f, 0.0, 101.0)


def dmc(T, H, Ro, Po, le):
    """Duff Moisture Code. `le` is the effective day length (hours of drying) in Eq. 16."""
    T, H, Ro, Po, le = (np.asarray(x, dtype="float64") for x in (T, H, Ro, Po, le))
    k = np.where(T >= -1.1, 1.894 * (T + 1.1) * (100.0 - H) * le * 1e-4, 0.0)            # Eq. 16, 17
    rain = Ro > 1.5
    re = np.where(rain, 0.92 * Ro - 1.27, 0.0)                                           # Eq. 11
    mo = 20.0 + 280.0 / np.exp(0.023 * Po)                                               # Eq. 12
    p_safe = np.maximum(Po, 1e-6)
    b = np.where(Po <= 33.0, 100.0 / (0.5 + 0.3 * Po),                                   # Eq. 13a
                 np.where(Po <= 65.0, 14.0 - 1.3 * np.log(p_safe), 6.2 * np.log(p_safe) - 17.2))   # Eq. 13b, 13c
    mr = mo + 1000.0 * re / (48.77 + b * re)                                             # Eq. 14
    pr_rain = 43.43 * (5.6348 - np.log(np.maximum(mr - 20.0, 1e-6)))                     # Eq. 15
    pr = np.where(rain, pr_rain, Po)
    return np.maximum(np.maximum(pr, 0.0) + k, 0.0)


def dc(T, Ro, Do, lf):
    """Drought Code. `lf` is the day-length factor in Eq. 22."""
    T, Ro, Do, lf = (np.asarray(x, dtype="float64") for x in (T, Ro, Do, lf))
    rain = Ro > 2.8
    rd = np.where(rain, 0.83 * Ro - 1.27, 0.0)                                           # Eq. 18
    qo = 800.0 * np.exp(-Do / 400.0)                                                     # Eq. 19
    qr = qo + 3.937 * rd                                                                 # Eq. 20
    dr = 400.0 * np.log(800.0 / qr)                                                      # Eq. 21
    d_after_rain = np.where(rain, np.maximum(dr, 0.0), Do)
    v = np.where(T > -2.8, 0.36 * (T + 2.8) + lf, lf)                                    # Eq. 22
    return d_after_rain + 0.5 * np.maximum(v, 0.0)                                       # Eq. 23


def isi(F, W):
    """Initial Spread Index from today's FFMC and wind (km/h): the Van Wagner (1987) form of the report."""
    F, W = np.asarray(F, dtype="float64"), np.asarray(W, dtype="float64")
    m = 147.2 * (101.0 - F) / (59.5 + F)                                                 # Eq. 1
    fw = np.exp(0.05039 * W)                                                             # Eq. 24
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)                         # Eq. 25
    return 0.208 * fw * ff                                                               # Eq. 26


def bui(P, D):
    """Buildup Index from today's DMC (P) and DC (D)."""
    P, D = np.asarray(P, dtype="float64"), np.asarray(D, dtype="float64")
    denom = np.maximum(P + 0.4 * D, 1e-9)
    low = 0.8 * P * D / denom                                                            # Eq. 27a
    high = P - (1.0 - 0.8 * D / denom) * (0.92 + (0.0114 * P) ** 1.7)                    # Eq. 27b
    return np.maximum(np.where(P <= 0.4 * D, low, high), 0.0)


def fwi(R, U):
    """Final Fire Weather Index. Not used by this project; kept for checking against the report."""
    R, U = np.asarray(R, dtype="float64"), np.asarray(U, dtype="float64")
    fd = np.where(U <= 80.0, 0.626 * np.maximum(U, 0.0) ** 0.809 + 2.0,                  # Eq. 28a
                  1000.0 / (25.0 + 108.64 * np.exp(-0.023 * U)))                         # Eq. 28b
    b = 0.1 * R * fd                                                                     # Eq. 29
    safe_b = np.maximum(b, 1.0)
    return np.where(b > 1.0, np.exp(2.72 * (0.434 * np.log(safe_b)) ** 0.647), b)        # Eq. 30a, 30b


def step(state, T, H, W, Ro, le=LE_EQUATORIAL, lf=LF_EQUATORIAL):
    """Advance (FFMC, DMC, DC) by one day. Returns the new state and the dict of today's indices."""
    f0, p0, d0 = state
    H = np.minimum(np.asarray(H, dtype="float64"), 100.0)
    f = ffmc(T, H, W, Ro, f0)
    p = dmc(T, H, Ro, p0, le)
    d = dc(T, Ro, d0, lf)
    i = isi(f, W)
    b = bui(p, d)
    return (f, p, d), {"ffmc": f, "dmc": p, "dc": d, "isi": i, "bui": b}
