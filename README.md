# SUMO Travel Performance Analysis

This project uses SUMO and Python to simulate vehicle behavior and evaluate travel performance for decision-making research.

## Project contents

- `run_simulation.py` - main SUMO simulation script using the LSTM model and `map.sumocfg`
- `hybrid_simulation.py` - alternate simulation workflow
- `hybrid_sim_changed.py` - modified hybrid simulation logic
- `urban_simulation.py` - urban grid simulation
- `model.py` - model definition or helper script
- `*.sumocfg`, `*.net.xml`, `*.rou.xml`, `*.nod.xml`, `*.edg.xml` - SUMO network, route, and configuration files
- `Levin_allcars_lstm.h5`, `NGSIM_lstm.h5` - trained TensorFlow/Keras model files
- `simulation_results.csv` - output data from simulations

## Prerequisites

- Windows
- Python 3.11+ (3.13 works for this project)
- SUMO installed and `SUMO_HOME` environment variable set
- A Python virtual environment is strongly recommended

## Setup

1. Create and activate a virtual environment in the project folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Confirm SUMO is configured:

```powershell
echo $env:SUMO_HOME
```

`SUMO_HOME` should point to your SUMO installation directory.

## Running the simulation

Run the main simulation script from the project root:

```powershell
python run_simulation.py
```

This script starts `sumo-gui` with `map.sumocfg` and uses the `Levin_allcars_lstm.h5` model.

## Other scripts

- `python hybrid_simulation.py`
- `python hybrid_sim_changed.py`
- `python urban_simulation.py`

Each script uses its own SUMO configuration file, so open the script if you need to adjust the route or network file.

## Notes

- If `pip` fails, use the venv Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

- If you do not want to track large model files in Git, consider using Git LFS for `*.h5` files.

## GitHub recommendations

Add these files to Git:

- `*.py`
- `*.sumocfg`
- `*.xml`
- `README.md`
- `requirements.txt`

Ignore these in `.gitignore`:

- `venv/`
- `__pycache__/`
- `*.pyc`
- `*.h5` (if large)
- `.vscode/`
- `.idea/`
- `Thumbs.db`
- `.DS_Store`

## License

Add a license if you want to share this project publicly.
