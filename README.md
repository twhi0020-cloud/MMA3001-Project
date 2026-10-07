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

## Data cleaning decisions
Raw occupancy covers only 36-58% of the 5-minute grid per room, and gaps cluster overnight and at weekends, so the hub appears to go quiet when a room is empty. `src/occupancy_forecast/cleaning.py` therefore:
- caps spikes more than 5 people above a rolling median (0.19% of observed bins), keeping the raw value;
- interpolates gaps up to 15 minutes;
- fills gaps up to 1 day with 0 people when the preceding headcount was 0.5 or less (`assumed_empty`, 27-43% of bins per room);
- leaves all other gaps missing (about 10% of bins) and excludes them from training.

The `assumed_empty` rule is an inference, not a measurement: there is no ground truth for why the hub is quiet.

## Planned sensitivity tests
- `quiet_threshold` set to 0.5, 1.0 and 2.0 people.
- Results with and without `assumed_empty` bins.
- Bin width of 1, 5 and 15 minutes.

## Licence
MIT (see LICENSE). Data is not covered by this licence.
