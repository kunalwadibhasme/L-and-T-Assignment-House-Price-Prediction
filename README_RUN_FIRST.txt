START HERE

1. Open this project folder in VS Code.
2. Read README.md.
3. Use PowerShell in this folder:
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
4. Open notebooks/01_eda.ipynb and run all cells.
5. Open notebooks/02_model_training.ipynb and run all cells.
6. Or run from terminal: python train_model.py
7. Generate PDF: python generate_report.py
8. Launch app: python app/app.py

You have the real Kaggle files in data/. train.csv has SalePrice; test.csv does not.
Do not use sample_submission.csv as training labels.
