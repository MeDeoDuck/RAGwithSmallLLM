"""Run one case of main.ipynb from the command line WITHOUT modifying main.ipynb.

The DO_* flags of the notebook's config cell are switched in memory, the code cells are written to
runs/main_<case>.ipy and executed with IPython, so `!`/`%` lines work and the tqdm bar / evaluation
logs stream to the terminal.

    python run_case.py pretrain         # 1) GPT-small pretraining       -> output/pretraining/best_model
    python run_case.py summary          # 2) summarization finetuning    (needs 1)
    python run_case.py classification   # 3) classification finetuning   (needs 1)
    python run_case.py rag              # 4) RAG finetuning on NQ        (needs 1)
    python run_case.py zeroshot_rag     # 5) naive RAG with Llama-3.2-1B-Instruct (needs HF_TOKEN)
    python run_case.py prompt_rag       # 6) baseline RAG (4.2.1) and extensions (4.3)  (needs HF_TOKEN)
    python run_case.py submission       # 7) build defaultproject_code.zip / defaultproject_supplementaries.zip

Add --from-scratch to 2) 3) 4) for the without-pretraining control the report needs:
results go to output/<case>_scratch/ and output/results/<case>_scratch_score.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys

CASES = {
    "pretrain": "DO_PRETRAIN",
    "summary": "DO_FINETUNE_SM",
    "classification": "DO_FINETUNE_CF",
    "rag": "DO_FINETUNE_RAG",
    "zeroshot_rag": "DO_ZEROSHOT_RAG",
    "prompt_rag": "DO_PROMPT_RAG",
    "submission": "DO_SUBMISSION",
}
NEEDS_PRETRAINED = {"summary", "classification", "rag"}
NEEDS_HF_TOKEN = {"zeroshot_rag", "prompt_rag"}
# directory name each finetuning case writes to, used by --from-scratch
CASE_DIRS = {"summary": "summary", "classification": "classification", "rag": "rag"}
RANDOM_INIT_DIR = os.path.join("output", "pretraining_random", "best_model")


# TrainingConfig values as written in main.ipynb, used to rescale for a smaller GPU
CASE_BATCH = {"pretrain": 16, "summary": 16, "classification": 32, "rag": 16}


def batch_replacements(case, batch):
    """Fit a smaller card without changing the training maths.

    global_steps in the notebook counts micro-batches, so shrinking batch_size alone
    would move every evaluation to a different token position and break the comparison
    with the reference log. gradient_accumulation_steps and both intervals are scaled by
    the same factor, which keeps the effective batch, the LR schedule and the token
    positions of each evaluation identical.
    """
    original = CASE_BATCH[case]
    if batch >= original:
        return []
    if original % batch:
        raise SystemExit(f"--batch {batch} must divide the notebook value {original} for {case}")
    factor = original // batch
    accum = {"pretrain": 2, "summary": 2, "classification": 1, "rag": 1}[case]
    return [
        # eval_batch_size first: "batch_size=16," is a substring of "eval_batch_size=16,"
        (f"eval_batch_size={original},", f"eval_batch_size={batch},", 1),
        (f"batch_size={original},", f"batch_size={batch},", 1),
        (f"gradient_accumulation_steps={accum},", f"gradient_accumulation_steps={accum * factor},", 1),
        ("eval_interval=2500,", f"eval_interval={2500 * factor},", 1),
        ("logging_interval=10,", f"logging_interval={10 * factor},", 1),
    ]


def scratch_replacements(case):
    """Redirect one finetuning case to the random checkpoint and to _scratch outputs.

    Only the log/output paths are renamed. The bare literal "rag" must NOT be
    rewritten: the rag cell also uses it for os.path.join(LOCAL_CACHE_PATH, "rag"),
    and renaming that re-downloads the NQ data and the 8.55 GB BM25 index.
    """
    directory = CASE_DIRS[case]
    return [
        ('"pretraining", "best_model"', '"pretraining_random", "best_model"', 1),
        (f'os.path.join(LOG_PATH, "{directory}")',
         f'os.path.join(LOG_PATH, "{directory}_scratch")', 1),
        (f'os.path.join(OUTPUT_PATH, "{directory}", "best_model")',
         f'os.path.join(OUTPUT_PATH, "{directory}_scratch", "best_model")', None),
        (f'os.path.join(OUTPUT_PATH, "{directory}")',
         f'os.path.join(OUTPUT_PATH, "{directory}_scratch")', 1),
        (f'"{directory}_score.json"', f'"{directory}_scratch_score.json"', 1),
        (f'"{directory}_output.txt"', f'"{directory}_scratch_output.txt"', 1),
    ]


def build_script(notebook, case, from_scratch=False, batch=None):
    with open(notebook, encoding="utf-8") as f:
        cells = [cell for cell in json.load(f)["cells"] if cell["cell_type"] == "code"]

    script = []
    patched = {flag: 0 for flag in CASES.values()}
    replacements = scratch_replacements(case) if from_scratch else []
    if batch is not None and case in CASE_BATCH:
        replacements = batch_replacements(case, batch) + replacements
    applied = {old: 0 for old, _, _ in replacements}
    for cell in cells:
        src = "".join(cell["source"])
        for name, flag in CASES.items():
            # \s*=(?!=) so that a comparison like `DO_X == True` is not rewritten
            src, n = re.subn(rf"^{flag}\s*=(?!=).*$", f"{flag} = {name == case}", src, flags=re.M)
            patched[flag] += n
        if src.lstrip().startswith(f"if {CASES[case]}:"):
            for old, new, _ in replacements:
                applied[old] += src.count(old)
                src = src.replace(old, new)
        if "notebook_login()" in src:
            # the login widget needs a browser; huggingface_hub reads the HF_TOKEN environment variable instead
            src = "# notebook_login() skipped by run_case.py (HF_TOKEN is used)\n"
        script.append(src)

    # re-count on the emitted script: the notebook_login() cell is wiped after patching,
    # so a flag assigned inside that cell would pass the first count and vanish here
    emitted = "\n".join(script)
    patched = {flag: len(re.findall(rf"^{flag}\s*=(?!=)", emitted, flags=re.M))
               for flag in CASES.values()}
    wrong = {flag: n for flag, n in patched.items() if n != 1}
    if wrong:
        sys.exit(f"[run_case] refusing to run: each DO_* flag must be assigned exactly once in "
                 f"{notebook}, got {wrong}")

    missed = {old: applied[old] for old, _, expected in replacements
              if applied[old] == 0 or (expected is not None and applied[old] != expected)}
    if missed:
        sys.exit(f"[run_case] refusing to run: unexpected replacement counts in "
                 f"{notebook}: {missed}")

    marker = f"{case}-scratch" if from_scratch else case
    script.append(f'print("[run_case] DONE: {marker}")\n')
    return "\n\n".join(script)


def main():
    parser = argparse.ArgumentParser(description="Run one case of main.ipynb headlessly.")
    parser.add_argument("case", choices=CASES)
    parser.add_argument("--notebook", default="main.ipynb")
    parser.add_argument("--dry-run", action="store_true", help="only write runs/main_<case>.ipy")
    parser.add_argument("--batch", type=int, default=None,
                        help="micro-batch size for a smaller GPU; gradient accumulation and the "
                             "log/eval intervals are scaled so the training maths and the token "
                             "positions of each evaluation stay identical")
    parser.add_argument("--from-scratch", action="store_true",
                        help="finetune from a random checkpoint instead of the pretrained one "
                             "(without-pretraining control; run make_random_init.py first)")
    args = parser.parse_args()

    if args.from_scratch and args.case not in CASE_DIRS:
        sys.exit(f"--from-scratch only applies to {sorted(CASE_DIRS)}")

    if args.case in NEEDS_PRETRAINED:
        best = RANDOM_INIT_DIR if args.from_scratch else os.path.join("output", "pretraining", "best_model")
        weights = ["model.safetensors", "pytorch_model.bin"]
        if not os.path.isdir(best):
            hint = "python make_random_init.py" if args.from_scratch else "python run_case.py pretrain"
            sys.exit(f"{best} not found: run `{hint}` first")
        if not os.path.exists(os.path.join(best, "config.json")) or \
                not any(os.path.exists(os.path.join(best, w)) for w in weights):
            sys.exit(f"{best} is incomplete (needs config.json and one of {weights}): rerun pretrain")
    if args.case in NEEDS_HF_TOKEN and not os.environ.get("HF_TOKEN"):
        print("[run_case] warning: HF_TOKEN is not set (meta-llama/Llama-3.2-1B-Instruct is a gated model)")

    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", f"main_{args.case}{'_scratch' if args.from_scratch else ''}.ipy")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_script(args.notebook, args.case, from_scratch=args.from_scratch,
                             batch=args.batch))
    print(f"[run_case] {args.case}: wrote {path}", flush=True)

    if not args.dry_run:
        sys.exit(subprocess.call([sys.executable, "-m", "IPython", path]))


if __name__ == "__main__":
    main()
