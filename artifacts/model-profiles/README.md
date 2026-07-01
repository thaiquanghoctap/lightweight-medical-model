Model profiling outputs belong here.

Suggested command:

```bash
uv run python -m scripts.analysis.profile_models \
  --models mednet mk_mnet r_cbam_mnet \
  --width-mults 0.25 0.5 1.0 \
  --output artifacts/model-profiles/profile.csv
```
