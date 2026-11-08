# functional_flow_climate

Neural-operator dynamics in the physical function space of ERA5, built on pg-flow
([physical_guide_flowmatching](https://github.com/nayehyeon61-glitch/physical_guide_flowmatching)):

    ERA5 -> flow matching (pg-flow, W2 / Sobolev structure) -> ensemble scenarios
         -> neural operator on the physical dynamics function space (this repo) -> prediction
         -> WeatherBench2 scores against a climate-ODE baseline

**Step 1 (this version): the dynamics field.** `ffc.field.DynamicsField` is a spherical neural operator
`G_theta : x -> dx/dt` on the 66 dynamic KAI channels, integrated by torchdiffeq, trained on 6 h rollouts with
an L2 + Sobolev `H^s` loss in the spherical-harmonic basis and a PINN term (strong-form primitive-equation
residual with the field's own tendency). It reads and writes pg-flow ensemble files and is scored with
WeatherBench2. Design and results: [docs/step1_design.md](docs/step1_design.md), [docs/step1_results.md](docs/step1_results.md).

| Path | What |
| --- | --- |
| `ffc/field.py` | `DynamicsField`: SFNO / FNO (neuraloperator) or ClimODE ResNet (baseline) tendency, torchdiffeq step, member-wise rollout |
| `ffc/sobolev.py` | `SphericalSobolev`: `H^s` norms through torch-harmonics SHT (`s = 1` gradients, `s = -1` the W2 tangent norm) |
| `ffc/physics.py` | PINN term: pg-flow `pgcfm.physics` strong residuals, ERA5-level scales, hinge loss |
| `ffc/ensemble.py` | pg-flow `forecast.nc` (member dim) in / out; perturbed-IC stand-in members |
| `ffc/evaluate.py` | WeatherBench2 datasets, climatology, RMSE / ACC / CRPS / spread-skill tables (pg-flow `pgcfm.eval`) |
| `ffc/data.py` | KAI 6 h ERA5 loaders (pg-flow `pgcfm.runs`), windows, static features |
| `scripts/fetch_era5.sh` | WeatherBench2 ERA5 -> KAI layout (pg-flow `wb2_to_kai.py`) |
| `scripts/train_field.py` | train one arm (field or baseline), resumable |
| `scripts/predict.py` | rollouts from ERA5 ICs or from pg-flow members -> `forecast.nc` |
| `scripts/evaluate.py` | WB2 scores of several forecast files + persistence |
| `configs/step1_64x32.yaml` | the Step 1 arms and commands |
| `third_party/pgflow`, `third_party/neuraloperator` | pinned submodules (pg-flow branch `claude/blend-sr-s1`, neuraloperator) |

```bash
git clone --recurse-submodules https://github.com/nayehyeon61-glitch/functional_flow_climate.git
pip install -r requirements.txt            # see the file for the torch CPU wheel and torchcfm --no-deps
python -m pytest tests -q
bash scripts/fetch_era5.sh                 # ERA5 64x32, 2010-2017, ~5 GB
python scripts/train_field.py --kai data/era5_5p625/kai6h --out runs/step1 --backbone sfno --lambda-sob 1 --lambda-p 0.05 --tag sfno_h1_pinn
python scripts/predict.py  --kai data/era5_5p625/kai6h --run runs/step1 --tag sfno_h1_pinn --year 2017 --members 8
python scripts/evaluate.py --kai data/era5_5p625/kai6h --year 2017 --forecast SFNO=runs/step1/sfno_h1_pinn_2017_m8.nc --clim-years 2010-2015 --out runs/step1/eval
```
