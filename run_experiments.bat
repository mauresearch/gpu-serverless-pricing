@echo off
REM Reproduces all experiments of the paper (results in .\results).
python scripts\download_data.py --out data || exit /b 1
python gpu_serverless_pricing_sim.py --only all  --out results\results_d01 --azure-functions data\invocations_per_function_md.anon.d01.csv || exit /b 1
python gpu_serverless_pricing_sim.py --only core --out results\results_d05 --azure-functions data\invocations_per_function_md.anon.d05.csv || exit /b 1
python gpu_serverless_pricing_sim.py --only core --out results\results_d10 --azure-functions data\invocations_per_function_md.anon.d10.csv || exit /b 1
python gpu_serverless_pricing_sim.py --only core --out results\results_synthetic --synthetic || exit /b 1
python scripts\paper_figures.py results paper_figures || exit /b 1
python scripts\paper_tables.py results > paper_figures\tables.md
