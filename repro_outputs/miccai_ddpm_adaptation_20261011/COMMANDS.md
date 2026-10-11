# Commands

Executed from the V12 repository unless another path is shown.

```bash
git clone --depth 1 https://github.com/miguel-isidro05/pddm_code /tmp/pddm-code.bvr7fd/repo
git -C /tmp/pddm-code.bvr7fd/repo rev-parse HEAD
shasum -a 256 \
  /tmp/pddm-code.bvr7fd/repo/pipeline.py \
  /tmp/pddm-code.bvr7fd/repo/pipeline2.py \
  /tmp/pddm-code.bvr7fd/repo/SGDM.py

# Original PDDM CPU smoke: synthetic 2-condition/1-target EEG,
# four diffusion steps, one epoch, finite (2, 1, 64) output.

# Final V12 MPS smoke; launcher used PETERSON_SEARCH_SMOKE=1 and both shards.
python scripts/check_peterson_search_v12.py \
  --output-dir /tmp/peterson-v12-final-smoke-20261011 --smoke

python -m pytest tests/peterson_search_v12 -q
python -m ruff check experiments/peterson_search_v12 tests/peterson_search_v12
python -m pytest -q
git diff --check
```
