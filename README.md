# stml

Course project: fitting a meta-model on a label indicating whether the signal is worth trading, where underlying signal generation process is unknown. Meta-model can be found by exploring various ML models. Research on a panel of futures price data + primary-signal labels (11 instruments, 2020–2022 signal window inside 1990–2022 price history).

This README is the single source of truth for getting set up and for our collaboration workflow. Read sections **2** (setup), **4** (why we set things up this way), and **5** (workflow) at least once.

---

## 1. What's in this repo

```
stml/
├── data/                    raw CSVs — read-only inputs
│   ├── ohlcv_data.csv
│   └── primary_signals.csv
├── notebooks/               shared notebooks (edits via PR review)
│   ├── eda.ipynb            project-wide EDA
│   ├── exploration.ipynb    minimal smoke test for the venv/kernel
│   └── <initials>/          your personal notebooks (e.g. notebooks/jj/)
├── results/<initials>/      your figures / tables / intermediate outputs
├── reports/                 shared deliverables (markdown / PDF write-ups)
├── src/stml/                shared Python helpers (importable from any notebook)
├── pyproject.toml           project metadata + dependencies
├── uv.lock                  pinned dependency versions (committed)
├── .gitattributes           rule that strips notebook outputs at commit time
└── .gitignore               OS junk + verification artifacts
```

---

## 2. First-time setup

### 2.1 Install `uv`

`uv` is the Python package manager + virtual environment manager we use. Pick the install command for your OS:

| OS | Command |
| --- | --- |
| **Windows** (PowerShell) | `irm https://astral.sh/uv/install.ps1 \| iex` |
| **macOS / Linux** (Terminal) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Any OS via pipx** (if pipx already installed) | `pipx install uv` |

**After install**, **restart your terminal** so the PATH is reloaded, then verify:

```bash
uv --version       # should print 0.11+
```

### 2.2 Clone the repo, sync the environment, hook up nbstripout

```bash
git clone git@github.com:Ise-Q/stml.git     # or the HTTPS URL if you don't have SSH set up
cd stml
uv sync                                     # creates .venv/, installs every pinned dep (~1–2 min first time)
uv run nbstripout --install                 # one-time per clone: enables auto-strip of notebook outputs
```

After `uv sync`, a `.venv/` folder appears in the repo. That's your project-local Python environment — Python 3.12, with every dependency from `uv.lock`.

### 2.3 Use the project in VS Code

1. Install the **Python** and **Jupyter** extensions in VS Code (if not already).
2. Open the `stml` folder (File → Open Folder).
3. Open `notebooks/eda.ipynb`. Top-right of the notebook there's a kernel picker — pick **`.venv (Python 3.12)`**. VS Code remembers it for next time.
4. Click **Run All**. If every cell runs and you see plots, you're set up correctly.

### 2.4 Optional: JupyterLab in the browser

```bash
uv run jupyter lab
```

---

## 3. Day-to-day `uv` cheat-sheet

| Task | Command |
| --- | --- |
| Run a Python script with project deps | `uv run python path/to/script.py` |
| Open a Python REPL with project deps | `uv run python` |
| Open JupyterLab | `uv run jupyter lab` |
| Add a runtime dependency | `uv add <pkg>` |
| Add a dev-only tool | `uv add --dev <pkg>` |
| Remove a dependency | `uv remove <pkg>` |
| Re-sync after pulling teammates' deps | `uv sync` |

`uv run <cmd>` always uses the project's `.venv/` — you never need to manually `source .venv/bin/activate`.

---

## 4. Why we set things up this way

Skim this once so you understand the *why* behind the three opinionated choices in this repo.

### 4.1 Why `src/stml/`?

Notebooks tend to copy-paste boilerplate (loading CSVs, computing returns, defining plotting helpers). If three people each define `load_data()` slightly differently, fixing a bug means fixing it three times. `src/stml/` is where any function used in more than one notebook lives.

Use it from any notebook, regardless of depth:

```python
import sys, pathlib

# Add src/ to sys.path. Walks up from the notebook to find the repo root.
_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / 'src').is_dir())
sys.path.insert(0, str(_root / 'src'))

from stml.io import load_data
ohlcv, signals = load_data()   # auto-finds data/ — no path argument needed
```

We're keeping `src/stml/` as a plain folder (not a packaged install) on purpose — for a short-lived project, packaging is overhead with no payoff. If we outgrow this, switching is a 5-minute change.

### 4.2 Why `nbstripout`?

Jupyter notebooks store their outputs (execution counters, plot images encoded as base64, dataframe HTML) **inside the `.ipynb` file**. Two people running the same notebook produce different bytes — which means a merge conflict every single time. `nbstripout` strips outputs from `.ipynb` files **at commit time only**: your local view is unchanged (you still see plots when you run cells), but git only sees code and markdown. This is the single highest-leverage fix for notebook git pain.

You **must** run `uv run nbstripout --install` once after cloning — otherwise the filter isn't active locally and outputs will leak into your commits.

### 4.3 Why `.gitattributes`?

Without `.gitattributes`, every person would have to remember to *separately* register the nbstripout filter on every clone. With `.gitattributes` committed to the repo, the rule (`*.ipynb filter=nbstripout`) ships with the project, so every clone applies the same filter — provided `nbstripout --install` has been run locally to register the filter implementation. Belt-and-braces: declarative rule in repo + one-time local install = consistent behaviour across the whole team.

---

## 5. Collaboration workflow

This is the conflict-minimization core. The headline idea: **work on your own branch, merge to `main` only at group checkpoints.**

### 5.1 Golden rules

1. **Never commit directly to `main`.** Always work on your own branch.
2. **Use a long-lived personal branch** named `dev/<initials>` (e.g. `dev/jj`). Push to it as often as you like — treat it as your personal sandbox in the cloud.
3. **Pull `main` into your branch regularly** (at the start of each session). After pulling, run `uv sync` if `pyproject.toml` or `uv.lock` changed.
4. **Open a PR to `main` only at a group checkpoint** — when a chunk of your work is ready to be shared with the team. Don't drip-merge half-finished work into `main`; checkpoints keep `main` clean and reviewable.
5. **Don't edit other people's personal folders** (`notebooks/<their-initials>/`, `results/<their-initials>/`). If you need their helper function, ask them to move it to `src/stml/` so we both depend on the same canonical version.
6. **Notebook outputs are auto-stripped** (nbstripout). Commit freely — git sees only code.

### 5.2 The standard cycle (copy-paste-ready)

```bash
# --- one-time, on first clone (covered in §2) ---
uv sync
uv run nbstripout --install

# --- once per person, the first time you start work ---
git checkout main
git pull
git checkout -b dev/<initials>         # e.g. dev/jj
git push -u origin dev/<initials>

# --- start of every session after that ---
git checkout main && git pull          # grab the latest shared state
git checkout dev/<initials>
git merge main                         # bring your branch up to date
uv sync                                # re-sync deps in case main moved them

# --- do your work in notebooks/<initials>/ , results/<initials>/ , src/stml/ etc. ---
git add notebooks/<initials>/ results/<initials>/
git commit -m "<short message>"
git push                               # pushes to your dev/<initials> branch

# --- at a group checkpoint, open a PR on GitHub: dev/<initials> → main ---
```

### 5.3 Where to put things

| Path | Who owns it | Notes |
| --- | --- | --- |
| `data/*` | shared, read-only | Never overwrite. |
| `notebooks/eda.ipynb` (and other notebooks at root) | shared | Edits go through PR review. |
| `notebooks/<initials>/*.ipynb` | **you** | Your personal notebooks. Committed, but only on your branch until checkpoint. |
| `src/stml/*.py` | shared helpers | Adding a function here affects everyone — do it on a PR. |
| `results/<initials>/...` | **you** | Your figures, tables, intermediate CSVs. Per-person subfolder = no name collisions. |
| `reports/*` | shared deliverables | Group writes together (group write-ups, final markdown/PDF). |
| `pyproject.toml`, `uv.lock` | shared | Modified by `uv add` / `uv remove`. Commit both together. |

### 5.4 Adding a dependency without breaking everyone

```bash
# from your branch
git checkout main && git pull
git checkout dev/<initials>
git merge main

uv add <pkg>          # updates pyproject.toml + uv.lock
git add pyproject.toml uv.lock
git commit -m "deps: add <pkg> for <reason>"
git push
```

The new dep lands in `main` when your next PR is merged. Teammates run `uv sync` after pulling to pick it up.

If two PRs both touched `uv.lock` and they conflict, **never hand-edit `uv.lock`**. Resolve by:

```bash
# accept the merged pyproject.toml, then regenerate the lock
rm uv.lock
uv lock
git add uv.lock
git commit
```

### 5.5 Resolving a notebook merge conflict (if it happens)

Even with nbstripout, two people editing the **same** notebook can still conflict at the code/markdown level. Notebook JSON is a bad merge target.

The robust recipe:

```bash
# pick one side wholesale, then redo the other side manually
git checkout --ours notebooks/eda.ipynb       # keep your version
# or
git checkout --theirs notebooks/eda.ipynb     # take their version
# then open the notebook, re-apply the discarded changes by hand, save, commit.
```

The better approach is to **not get into this situation**: keep personal work in `notebooks/<initials>/`, and only touch shared notebooks (`notebooks/eda.ipynb`, etc.) through coordinated PRs.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `command not found: uv` after install | Restart the terminal so PATH is reloaded. |
| `ModuleNotFoundError: torch` (or any project dep) | Run `uv sync` again. |
| VS Code can't find the `.venv` kernel | Command Palette → "Python: Select Interpreter" → pick `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows). |
| Notebook output still appearing in `git diff` | Run `uv run nbstripout --install` again, then `git add` the notebook. |
| `Permission denied (publickey)` on `git clone` | Use the HTTPS URL instead, or add an SSH key to your GitHub account. |
| Windows long-path errors during `uv sync` | Enable Windows long paths (see [Microsoft docs](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation)). |
| `uv` install on macOS blocked by Gatekeeper | Run the curl command from Terminal exactly as shown; Gatekeeper won't intercept it. |
