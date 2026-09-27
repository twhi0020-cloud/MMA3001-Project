# AI use log

Running record for the report's AI reflection (brief section 7). Add an entry whenever AI materially contributes.

| Date | Tool | Task | How I checked it | Errors / unhelpful output | My decision |
|---|---|---|---|---|---|
| 2026-09-28 | Claude | Explained brief; profiled datasets; proposed scope | Re-ran queries myself; questioned the "occupancy prediction" framing | Initial framing was circular (predicting a value the sensor already reports) | Chose 30-min headcount forecasting; audit as a finding |
| 2026-09-28 | Claude | Repo skeleton + `data_loading.py` | Read code; ran pytest | Caught UTC vs local-hour bug in first exploratory query | Adopted explicit Melbourne timezone conversion |
