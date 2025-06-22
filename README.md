# 🧪 Open Data: Version Control and Repositories for Beginners

> Learn to structure, version, and reproduce data science projects using Git, Poetry, Docker, and best practices.

---

## 📁 Project Structure

```
open-data-version-control/
├── notebooks/              # Notebooks for data, EDA, training, inference
├── src/                    # Modular code
│   ├── model_code/         # train.py, inference.py
│   ├── utils/              # preprocessing, evaluation
│   └── tests/              # unit tests
├── data/                   # Datasets (ignored in Git)
├── models/                 # Saved models (ignored in Git)
├── .gitignore
├── pyproject.toml
├── poetry.lock
├── README.md
├── pytest.ini
└── .vscode/settings.json   # VSCode config (optional)
```

---

## ⚙️ Environment Setup

### 🧩 Option 1: Create Conda Environment

```bash
conda create -n python_310 python=3.10
conda activate python_310
pip install poetry
poetry install
```

---

### 📦 Option 2: Poetry Only

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

---

## 🧪 Add Jupyter Kernel

```bash
poetry run ipython kernel install --user --name=python_310
```

Select `python_310` kernel inside your notebooks.

---

## 🚀 Steps to Run the Project

1. **Generate dataset**
   ```bash
   jupyter notebook notebooks/data.ipynb
   ```

2. **Explore data**
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```

3. **Train the model**
   ```bash
   jupyter notebook notebooks/train.ipynb
   ```

4. **Run inference**
   ```bash
   jupyter notebook notebooks/inference.ipynb
   ```

5. **Check model + prediction outputs in `/models` and `data/iris_predictions.csv`**

---

## 🔁 Git & GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/open-data-version-control.git
git push -u origin main
```

✅ Create feature branches:
```bash
git checkout -b feature/add-logging
```

✅ Merge changes:
```bash
git checkout main
git merge feature/add-logging
```

---

## ✅ Run Tests

```bash
# Run unit tests
pytest

# OR run with unittest
python -m unittest discover -s src/tests
```

---

## 🖥️ VSCode Setup 

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false,
  "python.testing.pytestArgs": ["src/tests"],
  "jupyter.jupyterServerType": "local"
}
```

## 🔧 Git Push Troubleshooting (HTTPS + Token)

If you get this error:

```
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/...'
```

✅ Use your GitHub token directly:

```bash
git push -u https://<your_username>:<your_token>@github.com/<your_username>/<repo>.git <branch_name>
```

Then configure credential storage:

```bash
git config --global credential.helper osxkeychain
```

## 🧼 Best Practices

- Use `pyproject.toml` instead of `requirements.txt` for long-term projects
- Always use `.gitignore`
- Write `README.md` with clear run instructions
- Commit often with meaningful messages
- Use `tests/` to validate behavior
- Tag releases for reproducibility

---
