# Short-horizon room headcount forecasting (MMA3001)

Individual project for MMA3001 Numerical Methods and Machine Learning.
Dataset: Monash Smart Infrastructure occupancy and environmental sensors.

> **Status:** in progress. Sections marked TODO are filled in as the project develops.

## Engineering problem
Forecast the headcount in a room 30 minutes ahead (with 15 and 60 minutes as comparison horizons),
so building systems (e.g. HVAC) can pre-condition spaces before occupants arrive. TODO: expand.

## Inputs and outputs
- **Input:** occupancy sensor readings (`headcount`, `occupancystatus`, timestamps with UTC offset, per `floorspaceid`).
- **Output:** predicted headcount (people) at time t + h, per room.
- TODO: units, domains, invalid-input handling.

## Repository layout
```
src/occupancy_forecast/   library code (data loading, features, models, validation)
tests/                    pytest tests
notebooks/                exploratory analysis
docs/                     generated HTML documentation (pdoc)
data/                     data instructions only; raw data is git-ignored
AI_USE_LOG.md             running AI-use record
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the data files in data/raw/ (see data/README.md)
pytest
```

## Known data issues (audit findings)
- Timestamps carry a UTC offset; hour-of-day features must use Australia/Melbourne local time.
- The file name says "MayToDec2024" but timestamps span Nov 2023 to Apr 2026.
- Occupancy data comes from a single hub device covering 5 floor spaces.
- Only 2 of 5 occupancy floor spaces and 1 of 5 environmental sensors appear in the location lookup, and the two sets share no room.

## Licence
MIT (see LICENSE). Data is not covered by this licence.
