# Multi-Indexed Lalinet GeneRAlized and Unified algorithm (MILGRAU)

MILGRAU is a set of Python scripts used to process Licel lidar measurements. The workflow is split into numbered steps and each script performs a specific part of the processing chain.

## Available scripts

1. **libids.py** – Organize raw binary data and remove spurious files.
2. **libids_scc2netcdf.py** – Convert organized files to NetCDF for the EARLINET Single Calculus Chain.
3. **lipancora.py** – Apply pre-processing corrections and generate level‑0 and level‑1 products.
4. **liracos.py** – Create range corrected signal (RCS) quick‑look plots.
5. **radiodata.py** – Download radiosonde profiles from the University of Wyoming.
6. **lebear.py** – Retrieve elastic backscatter and extinction profiles.
7. **lirabear.py** – Retrieve Raman backscatter and extinction profiles.
8. **08-TROPOPAUSE.py** – Compute tropopause height statistics using the downloaded radiosondes.

## Requirements

Install the scientific Python stack before running the scripts:

```bash
pip install -r requirements.txt
```

## Folder hierarchy

Each step assumes the following directories under the working folder:

```
01-data/               # raw lidar data
02-data_raw_organized/
03-rcsignal/
04-data_level0/
05-data_level1/
07-rawinsonde/         # radiosonde downloads
```

Example data with this layout can be found in `tests/data`.

## Running the scripts

### Organize raw data

```bash
python scripts/libids.py /path/to/workdir
```

### Convert to NetCDF

```bash
python scripts/libids_scc2netcdf.py /path/to/workdir
```

### Apply pre-processing corrections

```bash
python -c "from scripts import lipancora; lipancora.lipancora('/path/to/workdir')"
```

### Produce RCS quick‑looks

```bash
python -c "from scripts import liracos; liracos.liracos('/path/to/workdir')"
```

### Download radiosonde data

Edit `initial_date`, `final_date` and `station` in `scripts/radiodata.py` and run:

```bash
python scripts/radiodata.py
```

### Calculate the tropopause

Set the date range and station in `scripts/08-TROPOPAUSE.py` then execute:

```bash
python scripts/08-TROPOPAUSE.py
```

### Retrieval algorithms

Edit the user parameters inside `scripts/06-LEBEAR.py` or `scripts/07-LIRABEAR.py` and run the desired script:

```bash
python scripts/06-LEBEAR.py
python scripts/07-LIRABEAR.py
```

## Running the test suite

Execute the unit tests from the repository root:

```bash
PYTHONPATH=. pytest
```
