# titanic-disaster

Predicting passenger survival on the Titanic (April 15, 1912) with logistic regression, packaged in **two Docker containers**: one in **Python** and one in **R**. Each container loads the data, cleans it and engineers features, fits a model, reports accuracy, and writes predictions for the test set. Every step is printed to the terminal.

The dataset is **not** included in this repository. Step 2 below explains how to download it.

---

## Repository structure

```
titanic-disaster/
├── .github/
│   └── CODEOWNERS            # only the repo owner can approve changes
├── .gitignore                # ignores src/data/, *.csv, venvs, .DS_Store, etc.
├── README.md
└── src/
    ├── data/                 # YOU create this and put the Kaggle CSVs here (git-ignored)
    ├── python_app/
    │   ├── Dockerfile
    │   ├── requirements.txt  # pinned package versions
    │   └── run_model.py
    └── r_app/
        ├── Dockerfile
        ├── install_packages.R  # pinned package versions
        └── run_model.R
```

---

## Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| Git | clone the repo | https://git-scm.com/downloads |
| Docker Desktop (or Docker Engine) | build and run the containers | https://docs.docker.com/get-docker/ |
| A free Kaggle account | download the data | https://www.kaggle.com |

You do **not** need Python or R installed locally; everything runs inside Docker. Make sure Docker Desktop is open and running before Step 3 (check with `docker --version`).

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/XinyanGe/titanic-disaster.git
cd titanic-disaster
```

All commands below are run from this repository root folder.

---

## Step 2 — Download the data

You need three files from the Kaggle Titanic competition: `train.csv`, `test.csv`, and `gender_submission.csv`.

### Option A: download in the browser (easiest)

1. Go to https://www.kaggle.com/competitions/titanic/data and sign in.
2. Click **Join Competition** / accept the rules if prompted (required before downloading).
3. Click **Download All**, which gives you `titanic.zip`.
4. Create the data folder and unzip into it:

```bash
mkdir -p src/data
unzip ~/Downloads/titanic.zip -d src/data
```

(On Windows, create `src\data` and extract the zip there with File Explorer.)

### Option B: Kaggle command-line tool

```bash
pip install kaggle
# place your API token at ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
mkdir -p src/data
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

### Check

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

---

## Step 3 — Run the Python container

```bash
docker build -t titanic-python src/python_app
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

**Windows PowerShell** users, replace the second line with:

```powershell
docker run --rm -v "${PWD}/src/data:/app/data" titanic-python
```

**Windows Command Prompt** users:

```cmd
docker run --rm -v "%cd%/src/data:/app/data" titanic-python
```

The `-v` flag mounts your local `src/data` folder into the container at `/app/data`, so the container can read the CSVs and write its predictions back to your machine. Predictions are saved to `src/data/predictions_python.csv`.

---

## Step 4 — Run the R container

```bash
docker build -t titanic-r src/r_app
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

(Use the same PowerShell / Command Prompt substitutions as in Step 3.)

The first build takes a few minutes while R packages install; later builds are cached. Predictions are saved to `src/data/predictions_r.csv`.

---

## What the scripts do (and what you will see printed)

Both scripts follow the same steps and print every change they make:

1. **Load `train.csv`** and print its shape and first rows.
2. **Explore**: missing values per column, overall survival rate, survival by sex and by passenger class.
3. **Clean and engineer features** (all fill values are learned from the training set only and reused on the test set):
   - `Title` extracted from `Name` (Mr, Mrs, Miss, Master, Rare)
   - `FamilySize` = `SibSp` + `Parch` + 1, and `IsAlone` flag
   - Missing `Age` filled with the median age for that passenger's title
   - Missing `Fare` filled with the training median; missing `Embarked` filled with the training mode
   - `Sex` encoded female = 1, male = 0; `Pclass`, `Embarked`, `Title` treated as categorical
   - `PassengerId`, `Name`, `Ticket`, `Cabin` dropped
4. **Fit a logistic regression** and print its coefficients. The Python model uses all features above (scikit-learn regularizes by default). The R model uses base R `glm`, which is not regularized, so it leaves out `Sex` (already captured by `Title`) and `SibSp`/`Parch` (already captured by `FamilySize`) to avoid collinearity.
5. **Training accuracy**, plus an 80/20 hold-out validation accuracy as a more honest estimate.
6. **Load `test.csv`, apply the same cleaning, predict**, and save predictions.
7. **Test accuracy**. Kaggle does not release the true labels for `test.csv`. The scripts compare predictions against `gender_submission.csv`, which is Kaggle's sample baseline (every female survives, every male dies), and say so in the output. For the true test accuracy, upload `predictions_python.csv` or `predictions_r.csv` at https://www.kaggle.com/competitions/titanic/submissions.

---

## Package versions

**Python** (`python:3.11-slim` image): numpy 1.26.4, pandas 2.2.2, scikit-learn 1.5.1, plus their sub-dependencies (scipy, joblib, threadpoolctl, cloudpickle, packaging, python-dateutil, pytz, tzdata), all pinned in `src/python_app/requirements.txt`

**R** (`rocker/r-ver:4.4.1` image): dplyr 1.1.4, readr 2.1.5 (logistic regression uses base R `glm`)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ERROR: /app/data/train.csv not found` | The CSVs are not in `src/data/`, or you ran `docker run` from a folder other than the repo root. Re-check Step 2 and `cd` into `titanic-disaster`. |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it is running. |
| Volume mount fails on Windows | Use the PowerShell / Command Prompt versions of the command in Step 3. In Docker Desktop settings, make sure file sharing is allowed for the drive. |
| Kaggle download returns 403 | You must click **Join Competition** / accept the rules on the Kaggle page first. |

---

## Running without Docker (optional)

```bash
# Python
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r src/python_app/requirements.txt
python src/python_app/run_model.py

# R (from the src/r_app folder)
cd src/r_app && Rscript install_packages.R && Rscript run_model.R
```

Without Docker, the scripts read from `src/data/` automatically.
