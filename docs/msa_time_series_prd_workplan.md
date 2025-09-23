# PRD and Workplan: MSA Time-Series Prep for Shapefile Plotting

## 1) Overview
Prepare a clean, analysis-ready time-series dataset for Metropolitan Statistical Areas (MSAs) suitable for geospatial plotting (choropleths) using an MSA shapefile. The pipeline will standardize identifiers, align time frequencies, compute derived metrics, validate joins to geometry, and save final tabular and geospatial outputs for straightforward plotting in the notebook.

Primary consumers: this Jupyter Notebook and downstream visualization/reporting workflows.


## 2) Goals and Non-Goals
- Goals
  - Ingest and clean MSA HPI time-series data from hpi_master.csv.
  - Normalize MSA identifiers (CBSA codes and/or MSA names) for consistent joins to the MSA shapefile.
  - Prepare long-form and wide-form time-series with clean period index, optional resampling, and derived rates (e.g., YoY, MoM).
  - Join to MSA geometries and output a GeoDataFrame for plotting.
  - Produce outputs that can drive interactive and static geospatial plots for all listed MSAs.

- Non-Goals
  - Building a full dashboard application (only notebook-level plotting).
  - Complex statistical modeling beyond basic derived metrics.
  - Curating external data beyond what is provided or clearly referenced.


## 3) Scope
- In-scope
  - hpi_master.csv cleaning and standardization.
  - MSA shapefile loading, reprojection, and ID standardization.
  - Identifier reconciliation and optional crosswalk construction.
  - Time normalization (monthly/quarterly), gap handling, and derived indicators.
  - QA checks with explicit acceptance thresholds.
  - Persisting clean tables and GeoJSON/Parquet for fast reloads.

- Out-of-scope
  - County- or tract-level analysis.
  - Custom tile services or web server deployment.


## 4) Inputs and Assumptions
- Inputs
  - hpi_master.csv (present in repo root)
    - Expected columns (flexible; will profile):
      - date or period (parsable to pandas datetime)
      - msa identifier: cbsa code (preferred) and/or msa name
      - metric column(s): e.g., HPI, HPI_SA, etc.
  - MSA Shapefile (placed in shapefiles/ directory)
    - Typical sources: Census CBSA boundaries. Common ID fields: CBSAFP (5-digit CBSA code), NAME.
    - CRS: may vary; will reproject to EPSG:4326 for web mapping.
  - “Listed MSAs” definition
    - Default: All MSAs present in hpi_master.csv.
    - Optional: Provide a whitelist as a CSV (e.g., data/input/listed_msas.csv with cbsa_code column) or a list variable in the notebook.

- Assumptions
  - CBSA code is the canonical join key. If only names exist, a name-to-CBSA crosswalk will be built.
  - Date frequency is monthly or quarterly; any other cadence will be coerced to monthly or quarterly as configured.
  - hpi_master.csv fits in memory. If not, chunking will be added.


## 5) Outputs
- Tabular
  - data/processed/msa_timeseries_long.parquet: columns [cbsa_code, msa_name, period, metric, value, freq, source]
  - data/processed/msa_timeseries_wide.parquet: index [cbsa_code, period] with metric columns (e.g., HPI, HPI_yoy, HPI_mom, …)
  - data/processed/msa_latest.csv: latest period snapshot for quick maps
  - data/processed/crosswalk_msa.csv: any generated crosswalks (name->cbsa)

- Geospatial
  - data/geo/msa_geometries_clean.geojson: standardized MSA geometries with cbsa_code
  - data/geo/msa_timeseries_latest.geojson: latest-period map-ready GeoJSON

- Notebook
  - notebooks/MSA_Time_Series_Prep.ipynb (this notebook can evolve from Untitled-1.ipynb)

- Visuals (optional but recommended)
  - figures/maps/msa_choropleth_latest.png
  - figures/maps/msa_choropleth_latest.html (interactive)


## 6) Functional Requirements
1) Ingestion & Profiling
- Read hpi_master.csv with explicit dtype handling for codes (string, left-padded to 5).
- Profile column names, nulls, unique counts, date range, and frequency.

2) Identifier Standardization
- cbsa_code: 5-character string, zero-padded if numeric.
- msa_name: title-cased, trimmed, canonical punctuation.
- If CBSA missing but names present, generate crosswalk via a known reference (if available) or fuzzy matching with manual review.

3) Time Processing
- Uniform datetime column period (first-of-month or quarter end configured).
- Frequency normalization:
  - If monthly: ensure complete monthly index across observed range; optionally forward-fill gaps if appropriate.
  - If quarterly: align to quarter end; optionally upsample to monthly with rules or keep quarterly.
- Derived Metrics (configurable menu):
  - YoY: pct_change(periods=12) for monthly, 4 for quarterly.
  - MoM/QoQ: pct_change(1).
  - Indexed series (base = 100 at first available or custom base date).

4) Shapefile Join
- Load MSA shapefile from shapefiles/; reproject to EPSG:4326.
- Identify shapefile CBSA field (e.g., CBSAFP) and standardize to cbsa_code.
- Validate uniqueness: one geometry per cbsa_code.
- Left join timeseries to geometry (or vice versa) depending on output target; report unmatched IDs both sides.

5) Filtering & Slicing
- All MSAs by default; optionally filter to “listed” MSAs from provided list.
- Allow date range filters for generating specific outputs.

6) Outputs & Persistence
- Save standardized tabular Parquet/CSV and GeoJSON artifacts.
- Ensure reproducible reload with minimal computation.

7) Plotting Readiness
- Confirm outputs can be plotted via:
  - GeoPandas + matplotlib for static PNGs.
  - Plotly/Folium for interactive HTML maps.


## 7) Non-Functional Requirements
- Performance: Processing should complete in <2 minutes on typical laptop-scale datasets; if larger, add chunking.
- Reproducibility: Capture package versions; recommend conda or pip requirements file.
- Data Quality: Implement validation checks with explicit thresholds.
- Documentation: Code cells and markdown explaining each step.


## 8) Data Model and Schemas
- Long format (preferred for joins and time ops)
  - cbsa_code: str (5 chars)
  - msa_name: str
  - period: datetime64[ns]
  - metric: str (e.g., HPI)
  - value: float
  - freq: {"M","Q"}
  - source: str (e.g., "FHFA HPI")
  - Unique key: [cbsa_code, period, metric]

- Wide format (for fast plotting)
  - Index: [cbsa_code, period]
  - Columns: value metrics (HPI, HPI_yoy, HPI_mom, HPI_indexed, …)

- Geometry
  - cbsa_code: str (5)
  - geometry: polygon/multipolygon
  - Optional: NAME, LSAD, etc.


## 9) Validation and Acceptance Criteria
- Identifier Integrity
  - 100% of cbsa_code values are 5-char strings; no non-numeric characters unless required by source.
  - One geometry per cbsa_code (no duplicates) after standardization.

- Join Quality
  - >= 98% of cbsa_codes in timeseries successfully match geometries.
  - List of unmatched codes on both sides saved to data/quality/.

- Time Series Quality
  - No duplicate rows for [cbsa_code, period, metric].
  - Date range and frequency correctly detected; gaps documented.
  - Derived metrics non-null where previous period exists.

- Output Completeness
  - All specified outputs saved with expected row counts and columns.


## 10) Risks and Mitigations
- Risk: MSA names differ between CSV and shapefile.
  - Mitigation: Prefer CBSA code; build crosswalk if names differ.
- Risk: Mixed frequencies or irregular periods.
  - Mitigation: Explicit resampling rules, clear documentation.
- Risk: Large shapefile or dataset slows joins.
  - Mitigation: Use Parquet, GeoJSON; index on cbsa_code; avoid repeated heavy operations.
- Risk: Missing CBSA codes in input data.
  - Mitigation: Attempt name-based mapping; flag for manual resolution.


## 11) Tooling and Environment
- Python 3.10+
- pandas, numpy, pyarrow
- geopandas, shapely, pyproj, fiona
- plotly or folium (optional), matplotlib

Example install:
- pip install pandas numpy pyarrow geopandas shapely pyproj fiona plotly folium matplotlib


## 12) Directory and File Conventions
- data/
  - input/ (optional if adding more inputs)
  - processed/
  - geo/
  - quality/
- figures/
  - maps/
- notebooks/
- shapefiles/ (existing; contains MSA geometries)


## 13) Workplan (Step-by-Step)

Phase 0: Setup
- [ ] Create/activate environment; install dependencies.
- [ ] Create directories: data/processed, data/geo, data/quality, figures/maps, notebooks.

Phase 1: Ingest & Profile
- [ ] Load hpi_master.csv with dtype for codes as string.
- [ ] Standardize column names (snake_case), inspect head/info/describe.
- [ ] Profile: date range, frequency, unique MSAs, missing values.

Phase 2: Identifier Standardization
- [ ] Create cbsa_code column: zero-pad to 5 if numeric.
- [ ] Clean msa_name (title, strip, normalize whitespace/punctuation).
- [ ] If cbsa_code missing, build crosswalk (from external reference or name patterns); save to data/processed/crosswalk_msa.csv.

Phase 3: Time Normalization
- [ ] Parse date to period column (datetime64[ns]).
- [ ] Detect frequency (monthly, quarterly). Normalize to chosen target (e.g., monthly).
- [ ] Optionally fill small gaps; document any imputations.
- [ ] Compute derived metrics: mom/qoq, yoy, indexed base=100.

Phase 4: Shapefile Processing
- [ ] Load shapefile from shapefiles/; inspect available ID fields (e.g., CBSAFP, NAME).
- [ ] Reproject to EPSG:4326 and standardize cbsa_code as 5-char string.
- [ ] Validate one geometry per cbsa_code; resolve duplicates (keep primary or dissolve by cbsa_code if necessary).
- [ ] Save cleaned geometries to data/geo/msa_geometries_clean.geojson.

Phase 5: Join & Outputs
- [ ] Create long and wide tables as specified; enforce unique keys.
- [ ] Join latest period wide data to geometries; report unmatched to data/quality/.
- [ ] Save outputs:
  - data/processed/msa_timeseries_long.parquet
  - data/processed/msa_timeseries_wide.parquet
  - data/processed/msa_latest.csv
  - data/geo/msa_timeseries_latest.geojson

Phase 6: Plotting Prototypes
- [ ] Static PNG: choropleth of latest HPI (or chosen metric).
- [ ] Interactive HTML: choropleth (plotly or folium). Optionally add time slider (plotly animation frames or folium time slider plugin).

Phase 7: QA & Documentation
- [ ] Run validation checks; export QA summary to data/quality/qa_report.md.
- [ ] Document steps and decisions in notebook markdown.

Phase 8: Finalization
- [ ] Save notebook as notebooks/MSA_Time_Series_Prep.ipynb.
- [ ] Ensure clean reload path using saved Parquet/GeoJSON artifacts.


## 14) Implementation Notes (Notebook Outline)
1. Setup
   - Imports, paths, config flags (target frequency, listed MSAs path, base date for index).
2. Ingest hpi_master.csv
   - Read, profile, standardize columns.
3. Identifier handling
   - Ensure cbsa_code and msa_name standardized; generate crosswalks if needed.
4. Time processing
   - Parse dates, normalize frequency, derive metrics.
5. Shapefile handling
   - Load, reproject, standardize cbsa_code, save cleaned geometries.
6. Joins and outputs
   - Build long/wide tables, joins, save outputs.
7. Plotting
   - Static and interactive examples.
8. QA & export
   - Validation checks and QA report.


## 15) Acceptance Checklist
- [ ] >=98% join rate between timeseries and geometries.
- [ ] No duplicate keys [cbsa_code, period, metric].
- [ ] Output files exist with correct columns and reasonable sizes.
- [ ] Plots render without errors using the saved artifacts.


## 16) Open Questions
- What is the authoritative list of “listed MSAs”? If separate from hpi_master.csv, provide path or list.
- Which HPI metric variant should be primary (seasonally adjusted/non-SA)?
- Preferred target frequency (monthly vs quarterly)? If quarterly, align to quarter end.
- Any specific date range focus for plotting?
