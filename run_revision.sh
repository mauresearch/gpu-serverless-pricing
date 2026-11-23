#!/bin/sh
set -e
# Experiments of the revision (results in ./results_rev).
python scripts/download_data.py --out data
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev/results_d01 --azure-functions data/invocations_per_function_md.anon.d01.csv
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev/results_d05 --azure-functions data/invocations_per_function_md.anon.d05.csv
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev/results_d10 --azure-functions data/invocations_per_function_md.anon.d10.csv
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev/results_synthetic --synthetic
python scripts/revision_experiments.py workpred --out results_rev/workpred
python scripts/revision_experiments.py manipulation --out results_rev/manipulation
echo Finished.
