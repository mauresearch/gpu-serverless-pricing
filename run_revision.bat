@echo off
REM Experiments of the revision (results in .\results_rev).
python scripts\download_data.py --out data || exit /b 1
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev\results_d01 --azure-functions data\invocations_per_function_md.anon.d01.csv || exit /b 1
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev\results_d05 --azure-functions data\invocations_per_function_md.anon.d05.csv || exit /b 1
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev\results_d10 --azure-functions data\invocations_per_function_md.anon.d10.csv || exit /b 1
python gpu_serverless_pricing_sim.py --seeds 10 --only core --out results_rev\results_synthetic --synthetic || exit /b 1
python scripts\revision_experiments.py workpred --out results_rev\workpred || exit /b 1
python scripts\revision_experiments.py manipulation --out results_rev\manipulation || exit /b 1
echo Finished.
