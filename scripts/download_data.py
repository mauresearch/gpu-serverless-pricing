"""Download the public workload traces used in the experiments into ./data."""
import argparse
import os
import shutil
import tarfile
import urllib.request

FUNCTIONS_URL = ("https://github.com/Azure/AzurePublicDataset/releases/download/"
                 "dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz")
LLM_URL = ("https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/"
           "data/AzureLLMInferenceTrace_conv.csv")
DAYS = ("d01", "d05", "d10")


def fetch(url, path):
    if os.path.exists(path):
        print(f"[skip] {path} already exists")
        return
    print(f"[get ] {url}")
    tmp = path + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--days", nargs="+", default=list(DAYS), help="days of the Azure Functions trace")
    ap.add_argument("--keep-archive", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fetch(LLM_URL, os.path.join(a.out, "AzureLLMInferenceTrace_conv.csv"))

    wanted = {f"invocations_per_function_md.anon.{d}.csv" for d in a.days}
    missing = [w for w in wanted if not os.path.exists(os.path.join(a.out, w))]
    if missing:
        archive = os.path.join(a.out, "azurefunctions-dataset2019.tar.xz")
        fetch(FUNCTIONS_URL, archive)
        with tarfile.open(archive, "r:xz") as tar:
            for m in tar.getmembers():
                name = os.path.basename(m.name)
                if name in missing and m.isfile():
                    print(f"[extract] {name}")
                    with tar.extractfile(m) as src, open(os.path.join(a.out, name), "wb") as dst:
                        shutil.copyfileobj(src, dst, length=1 << 20)
        if not a.keep_archive:
            os.remove(archive)
    print("done:", ", ".join(sorted(os.listdir(a.out))))


if __name__ == "__main__":
    main()
