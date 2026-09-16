## 🚀 Quick Start / Reproduction Guide

Follow the steps below in your terminal to set up the environment and reproduce the results:

### 1. Create and Activate Conda Environment (Recommended)
```bash
conda create -n eunsu python=3.11
conda activate eunsu
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python src/main.py
```

### 4. Evaluation & Submission Generation
1. Once training on the 16 samples completes, two evaluation plot windows will appear.
2. **Close both plot windows** to proceed; the script will then finalize and generate the output.
3. Check and verify the generated `submission.json` file in the root or `outputs/` directory.
