# Fair and SLO-Aware Pricing for GPU-Enabled Serverless AI Inference

Simulation code for the paper

> E. Safari and M. Ramezanpour, "Fair and SLO-Aware Pricing for GPU-Enabled Serverless AI Inference: A Stackelberg Game with Closed-Form Shapley Cost Sharing" (under review).

The code implements a discrete-event simulator of multi-tenant GPU serverless endpoints (SimPy). It also implements the three layers of the proposed framework:

1. **Stackelberg pricing.** The provider (leader) sets a unit price for each model in every billing round. The users (followers) respond with quasi-linear demand that accounts for their overhead share and for the lower value of late or degraded responses.
2. **Cooperative cost sharing.** The boot and keep-alive overhead of each model and round is divided by the exact Shapley value of a keep-warm cost game. The Shapley value is computed in closed form in O(n²) time, and each share is billed as the fixed part of a two-part tariff.
3. **SLO-aware dispatching.** Requests are served with deadline-aware dynamic batching, predictive admission control, fallback to a smaller model of the same family and overflow-aware scaling.

## Repository layout

```
gpu_serverless_pricing_sim.py   simulator, mechanisms, experiments, Excel and figure output
scripts/download_data.py        downloads the public traces into data/
scripts/paper_figures.py        Figures 3-6 of the paper from the Excel outputs
scripts/paper_tables.py         Tables 7-9 of the paper (Markdown) from the Excel outputs
run_experiments.sh / .bat       runs every experiment of the paper
data/                           workload traces (not tracked, see below)
```

## Requirements

Python 3.9 or newer. The code was tested with Python 3.13, SimPy 4.1, NumPy 2.5, pandas 3.0, SciPy 1.18, Matplotlib 3.11 and openpyxl 3.1.

```bash
pip install -r requirements.txt
```

## Workload data

The experiments use two public traces from the [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset). They are not included in this repository because of their size (about 450 MB in total).

| File | Used for | Size |
|---|---|---|
| `invocations_per_function_md.anon.d01.csv` | arrival pattern of the users, Azure day 1 | 146 MB |
| `invocations_per_function_md.anon.d05.csv` | arrival pattern of the users, Azure day 5 | 147 MB |
| `invocations_per_function_md.anon.d10.csv` | arrival pattern of the users, Azure day 10 | 152 MB |
| `AzureLLMInferenceTrace_conv.csv` | prompt and output lengths of LLM requests | 0.7 MB |

### Option 1: automatic download

```bash
python scripts/download_data.py
```

The script downloads the Azure Functions 2019 archive (143 MB) and extracts the three days. It also downloads the LLM trace and deletes the archive afterwards; add `--keep-archive` to keep it.

### Option 2: manual download

1. **Azure Functions trace 2019** [1]. Download the archive

   https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz

   Extract it with `tar -xJf <file>` on Linux and macOS, or with 7-Zip on Windows. Then copy `invocations_per_function_md.anon.d01.csv`, `...d05.csv` and `...d10.csv` into `data/`. The other files of the archive are not needed.

2. **Azure LLM inference trace 2023, conversation** [2]. Download

   https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv

   and save it as `data/AzureLLMInferenceTrace_conv.csv`.

The expected layout is:

```
data/
  AzureLLMInferenceTrace_conv.csv
  invocations_per_function_md.anon.d01.csv
  invocations_per_function_md.anon.d05.csv
  invocations_per_function_md.anon.d10.csv
```

Please cite the original papers and follow the license of the Azure Public Dataset when you use these traces.

If the LLM trace is missing, request sizes are drawn from a lognormal distribution. If the Azure Functions trace is missing, run with `--synthetic`.

## Quick test

```bash
python gpu_serverless_pricing_sim.py --quick --only core --out results_test
```

This runs a shortened version of the main comparison and the ablation in a few minutes and writes `results_test/results.xlsx`.

## Reproducing the paper

Run everything with one command:

```bash
./run_experiments.sh          # Linux / macOS
run_experiments.bat           # Windows
```

or run the experiments one by one:

```bash
# Azure day 1: main comparison, ablation, sensitivity analysis, Shapley benchmark
python gpu_serverless_pricing_sim.py --only all  --out results/results_d01 \
       --azure-functions data/invocations_per_function_md.anon.d01.csv
# Azure days 5 and 10 and the synthetic workload: main comparison and ablation
python gpu_serverless_pricing_sim.py --only core --out results/results_d05 \
       --azure-functions data/invocations_per_function_md.anon.d05.csv
python gpu_serverless_pricing_sim.py --only core --out results/results_d10 \
       --azure-functions data/invocations_per_function_md.anon.d10.csv
python gpu_serverless_pricing_sim.py --only core --out results/results_synthetic --synthetic

# figures and tables of the paper
python scripts/paper_figures.py results paper_figures
python scripts/paper_tables.py results > paper_figures/tables.md
```

Each run uses 10 seeds for the main comparison and the ablation, and 10 seeds per point of the sensitivity analysis. On a two-core machine with `--workers 2`, the day-1 run takes about 25 minutes and each of the other runs about 5 minutes. On a single core expect several hours for day 1.

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--only` | `all` | `all`, `core` (main + ablation), `main`, `ablation` or `sensitivity` |
| `--seeds` | 10 | seeds for the main comparison and the ablation |
| `--sens-seeds` | 10 | seeds per point of the sensitivity analysis |
| `--horizon` | 6 | simulated hours (24 billing rounds of 15 min) |
| `--users` | 40 | number of users |
| `--workers` | all cores | parallel processes |
| `--azure-functions` | day 1 | path of an Azure Functions trace file |
| `--azure-llm` | `data/AzureLLMInferenceTrace_conv.csv` | path of the LLM trace |
| `--synthetic` | off | Gamma renewal arrivals instead of the Azure trace |
| `--out` | `results` | output folder |
| `--quick` | off | short smoke test |
| `--replot` | off | redraw figures and Excel from `raw_results.pkl` |

The same defaults can also be changed in the `USER_SETTINGS` dictionary at the top of the simulator, which is convenient when the script is run from an IDE.

### Compared configurations

| Configuration | Pricing | Overhead sharing | Tariff | Batching | Capacity control |
|---|---|---|---|---|---|
| Industry | fixed cost-plus | trigger pays | linear | fixed (8, 0.25 SLO) | no |
| Industry + L3 | fixed cost-plus | trigger pays | linear | SLO-aware | yes |
| Stackelberg-Prop | Stackelberg | proportional | linear | fixed | no |
| Stackelberg-Prop + L3 | Stackelberg | proportional | linear | SLO-aware | yes |
| Proposed | Stackelberg | Shapley (closed form) | two-part | SLO-aware | yes |

The ablation replaces one component of the proposed framework at a time. The sensitivity analysis varies load, keep-alive window, SLO, cold-start time, consumer-surplus weight, price elasticity, and the value of fallback and late responses.

## Outputs

Each run writes the following to its output folder:

- `results.xlsx`: all results. The main sheets are `Main_Summary` (mean and 95% CI), `Main_StatTests` (paired t-tests), `Ablation_Summary`, `Sensitivity_Summary`, `PerUser_<method>`, `Prices_Proposed`, `Demand_Proposed`, `Shapley_Accuracy` and `Shapley_Scalability`. The `README` and `Metric_Definitions` sheets describe every sheet and metric.
- `figures/`: diagnostic figures (PNG and PDF).
- `summary.txt`: run settings and summary tables.
- `raw_results.pkl`: raw results, used by `--replot`.

Correspondence with the paper:

| Paper | Source |
|---|---|
| Table 7 | `Main_Summary` of the four runs |
| Table 8 | `Ablation_Summary` (allocation rules) |
| Table 9 | `Ablation_Summary` (other components) |
| Fig. 3 | `Prices_Proposed`, `Demand_Proposed` (day 1) |
| Fig. 4 | `PerUser_*` sheets (day 1) |
| Fig. 5 | `Sensitivity_Summary` (day 1) |
| Fig. 6 | `Shapley_Accuracy`, `Shapley_Scalability` |

All random streams are seeded, so repeated runs on the same platform give identical results. Results on the synthetic workload can differ in the last digits across operating systems, and the running times in `Shapley_Scalability` depend on the machine.

## Model parameters

| Model | GPU | Price ($/h) | Cold start (s) | Isolated latency (s) | Max instances |
|---|---|---|---|---|---|
| S-Vision | 1× L4 | 0.80 | 8 | 0.011 | 4 |
| M-LLM-7B | 1× A100-80GB | 2.50 | 48 | 3.06 | 4 |
| L-LLM-70B | 8× H100 | 31.60 | 84 | 6.00 | 3 |

The prices are Modal list prices. The cold starts and latencies are taken from Torpor (USENIX ATC 2025), ServerlessLLM (OSDI 2024) and Splitwise (ISCA 2024). The per-request SLO is five times the isolated latency, with a 150 ms floor for the vision model. The full list of parameters and their sources is given in the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{safari2026pricing,
  author  = {Safari, Elaheh and Ramezanpour, Mohammadreza},
  title   = {Fair and {SLO}-Aware Pricing for {GPU}-Enabled Serverless {AI} Inference:
             A {Stackelberg} Game with Closed-Form {Shapley} Cost Sharing},
  year    = {2026},
  note    = {Under review}
}
```

## References

[1] M. Shahrad et al., "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider," USENIX ATC, 2020.

[2] P. Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA, 2024.

## License

The code is released under the MIT License (see `LICENSE`). The workload traces are distributed by Microsoft Azure under their own terms.
