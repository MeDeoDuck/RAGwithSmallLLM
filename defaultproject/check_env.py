"""One-shot environment check for the CSE402 default project (run inside the Docker container).

    docker run --gpus all --rm --env-file .env -v "$PWD":/workspace -v cse402-cache:/root/.cache cse402 python check_env.py

FAIL = must be fixed before any training, WARN = only needed for a specific case (see the message).
"""
import os
import shutil
import subprocess
import sys

results = []  # (level, name, detail)


def check(name, fn, level_on_error="FAIL"):
    # run every check and report all problems at once instead of stopping at the first one
    print(f"[ .. ] {name} ...", flush=True)
    try:
        ok, detail = fn()
        level = "OK" if ok else level_on_error
    except Exception as e:
        level, detail = level_on_error, f"{type(e).__name__}: {e}"
    results.append((level, name, detail))
    print(f"[{level:4s}] {name}  {detail}", flush=True)


def python_version():
    version = sys.version.split()[0]
    return version.startswith("3.12."), version


def torch_gpu():
    import torch
    if not torch.cuda.is_available():
        return False, f"torch {torch.__version__}, CUDA not available (docker run --gpus all / NVIDIA Container Toolkit)"
    major, minor = torch.cuda.get_device_capability(0)
    # torch.cuda.is_bf16_supported() also returns True for emulated bf16 on pre-Ampere cards
    bf16 = major >= 8
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    detail = (f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, "
              f"sm_{major}{minor}, bf16={bf16}, {vram:.0f} GB VRAM")
    if vram < 12:
        detail += " (<12 GB: lower batch_size and raise gradient_accumulation_steps)"
    # the handout asks for 2.6.0; Blackwell cards (sm_120) have no 2.6.0 kernels at all,
    # so 2.7.x is accepted there and the deviation is recorded in the report
    version_ok = torch.__version__.startswith("2.6.0") or (major >= 12 and torch.__version__.startswith("2.7."))
    if not torch.__version__.startswith("2.6.0") and version_ok:
        detail += " (2.7.x required by sm_120: handout pins 2.6.0, note this in the report)"
    return version_ok and bf16, detail


def packages():
    import transformers, datasets
    pinned = {"transformers": "4.52.4", "datasets": "3.6.0"}
    import importlib.metadata as md
    seen = {}
    for name in ["transformers", "datasets", "pyserini", "faiss-cpu", "evaluate",
                 "nltk", "rouge_score", "wget", "numpy", "pandas", "accelerate"]:
        try:
            seen[name] = md.version(name)
        except md.PackageNotFoundError:
            seen[name] = "MISSING"
    bad = [f"{n}={v}" for n, v in seen.items()
           if v == "MISSING" or (n in pinned and v != pinned[n])]
    return not bad, (f"problems: {bad}" if bad else
                     ", ".join(f"{n} {v}" for n, v in seen.items()))


def java():
    first_line = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    return '"21' in first_line, f"{first_line} (JAVA_HOME={os.environ.get('JAVA_HOME')})"


def pyserini_jvm():
    from pyserini.search.lucene import LuceneSearcher  # noqa: F401  (starts the JVM through pyjnius)
    return True, "pyserini.search.lucene import OK"


def project_files():
    required = ["main.ipynb", "model.py", "model_rag.py", "run_case.py",
                "rag_parsing.py", "prompt_rag.py", "enhanced_rag.py",
                "rag_diagnostics.py", "make_random_init.py", "test_prompt_rag.py",
                "dataset/pretrain.py", "dataset/summary.py", "dataset/classification.py",
                "dataset/rag.py", "utils/etc.py", "utils/logger.py", "utils/metrics.py"]
    missing = [f for f in required if not os.path.exists(f)]
    return not missing, f"cwd={os.getcwd()}" + (f", missing: {missing} (-v \"$PWD\":/workspace ?)" if missing else "")


def disk():
    # the corpus goes to /root/.cache while the BM25 index goes to /workspace: these are
    # usually different partitions, so both have to be measured
    targets = [(".", 70), ("/root/.cache", 50), ("/dev/shm", 1)]
    parts, ok = [], True
    for path, need_gb in targets:
        if not os.path.exists(path):
            continue
        free_gb = shutil.disk_usage(path).free / 2**30
        parts.append(f"{path}: {free_gb:.0f} GB free (need {need_gb})")
        ok = ok and free_gb >= need_gb
    note = (" | corpus 35GB + BM25 index 8.55GB + DPR NQ + checkpoints"
            " | WARNING: a named docker volume reports the VM's virtual free space, not the"
            " host disk backing it. Check the host separately (df -h on the docker data root);"
            " if the host fills up the filesystem turns read-only mid-run and the JVM dies with"
            " SIGBUS. Bind-mounting a host directory instead makes this number truthful.")
    return ok, "; ".join(parts) + note


def host_ram():
    if not hasattr(os, "sysconf") or "SC_PHYS_PAGES" not in os.sysconf_names:
        return True, "skipped (not a POSIX host)"
    total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    # dataset/rag.py loads biencoder-nq-train.json with pandas.read_json in one shot
    return total_gb >= 64, f"{total_gb:.0f} GB (64 GB+ for the rag / zeroshot_rag cases)"


def hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, "HF_TOKEN is not set -> write it in .env and use --env-file .env (needed for zeroshot_rag)"
    from huggingface_hub import whoami
    info = whoami(token=token)
    return True, f"logged in as {info['name']}"


def llama_access():
    if not os.environ.get("HF_TOKEN"):
        return False, "skipped (no HF_TOKEN)"
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
    try:
        hf_hub_download("meta-llama/Llama-3.2-1B-Instruct", "config.json",
                        token=os.environ["HF_TOKEN"])
    except GatedRepoError:
        return False, ("403 gated: accept the license at "
                       "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct AND make sure the token "
                       "has read access to public gated repos (a classic Read token has it)")
    return True, "meta-llama/Llama-3.2-1B-Instruct is accessible"


def readme():
    return os.path.exists("README"), "README (no extension) is packed by the submission cell (needed for submission)"


check("Python 3.12", python_version)
check("PyTorch 2.6.0 + GPU (bf16)", torch_gpu)
check("pinned packages", packages)
check("Java 21", java)
check("pyserini (JVM)", pyserini_jvm)
check("project files mounted", project_files)
check("disk space", disk)
check("host RAM", host_ram, "WARN")
check("HF_TOKEN", hf_token, "WARN")
check("Llama-3.2 license / access", llama_access, "WARN")
check("README for submission", readme, "WARN")

print()
width = max(len(name) for _, name, _ in results)
for level, name, detail in results:
    print(f"[{level:4s}] {name:{width}s}  {detail}")

failed = [name for level, name, _ in results if level == "FAIL"]
warned = [name for level, name, _ in results if level == "WARN"]
print()
if failed:
    print(f"NOT READY: fix {failed} first (RUN_GUIDE.md section 0)")
    sys.exit(1)
print("READY: training cases can run" + (f" (warnings: {warned})" if warned else ""))
