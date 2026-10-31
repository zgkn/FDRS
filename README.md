# FDRS: fire risk for southern ASEAN from fire-weather indices and hotspots

Fire danger for Sumatra, peninsular Malaysia, Singapore, Borneo, Java and Brunei (94-120E, 11S-8N), built in the way of the
Indonesian and Malaysian Fire Danger Rating Systems (de Groot et al. 2006, 2005) but calibrated by regression:

1. compute the Canadian FWI-System codes **FFMC, DMC, DC** and the indices **ISI, BUI** (not the final FWI) from ERA5;
2. count the daily VIIRS active fires (FIRMS: S-NPP, NOAA-20, NOAA-21) per 0.25 degree cell;
3. fit a logistic regression of "a fire was detected in the cell that day" on the indices; its probability is the risk.

Steps 1 and 2 are built; step 3 (the regression) is next.

## Pieces

| file | what |
|---|---|
| `scripts/fwi.py` | FWI codes, vectorised; a transcription of the C code of Wang, Anderson & Suddaby (2015), NOR-X-424. Day length is a parameter; the equatorial constants (DMC 9.0 h, DC 1.39) of de Groot et al. are the default. |
| `scripts/era5_fwi.py` | hourly ARCO-ERA5 -> noon T, RH, wind and 24 h rain -> daily FFMC, DMC, DC, ISI, BUI per cell. Solar-noon hour per column; 184-day spin-up. |
| `scripts/download_firms.py` | FIRMS area API -> daytime, nominal/high-confidence vegetation-fire pixels -> de-duplicated across the three sensors (500 m, NOAA-21 > NOAA-20 > S-NPP) -> counts per cell-day, and a per-day table of which sensors were active. |
| `scripts/grid.py` | the box and the ERA5 0.25 degree grid both are put on |
| `tests/` | the 49-day test of the report's Table 7 reproduced to 0.1; vectorised = scalar; FIRMS filtering, de-duplication, aggregation |
| `.github/workflows/pipeline.yml` | tests, then one hotspot job and one FWI job per year |

## Running

Add the FIRMS key as a repository secret named `FIRMS_MAP_KEY` (Settings > Secrets and variables > Actions); get a key at
https://firms.modaps.eosdis.nasa.gov/api/map_key/. Then Actions > *FDRS data pipeline* > Run workflow. Leave **smoke** ticked
first (10 days of 2019, one job per step); untick it for the full years. Outputs are workflow artifacts (30 days).

Locally: `pip install -r requirements.txt`, `python -m pytest tests`, and
`python scripts/era5_fwi.py --start 2019-09-01 --end 2019-09-10 --spinup-days 60 --out out/fwi.nc`.

## Design decisions

* **Response**: binary, at least one unique daytime detection in the cell that day.
* **Sensors**: all three, de-duplicated at pixel level; the number of sensors active that day is kept as a covariate because
  more sensors means more overpasses, which raises detection on its own (one sensor 2012-17, two from 2018, three from 2023).
* **Years**: pilot 2019-2023 (dry 2019 and 2023, wet La Nina 2020-2022); 2015, the extreme haze year, is held out to test the
  model at DC well above 400.
* **Predictors**: the indices plus a per-region intercept to start; no static covariates yet.

## Caveats

* ERA5 rain drizzles too often in the tropics and its 10 m wind is a 28 km average, so the paper's thresholds (FFMC 78/83,
  DC 400, ISI 6) do not transfer; the regression recalibrates.
* The paper's ISI classes came from the 1992 fire behaviour prediction ISI, this code has the 1987 form (the report notes the difference).
* ERA5 runs about 6 days behind real time, so this is a hindcast; a near-real-time map needs forecast weather.
