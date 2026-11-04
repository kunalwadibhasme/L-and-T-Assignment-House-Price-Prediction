"""Generate the PDF report using the latest real-data training summary."""
from pathlib import Path
import json
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "artifacts" / "summary.json"
if not SUMMARY_PATH.exists():
    raise FileNotFoundError("Training summary not found. Run `python train_model.py` first.")
S = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
PDF_PATH = ROOT / "House_Price_Project_Report.pdf"

BLUE = colors.HexColor("#173B5E")
TEAL = colors.HexColor("#087E82")
GRAY = colors.HexColor("#52616B")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="MainTitle2", parent=styles["Title"], fontSize=20, leading=25, textColor=BLUE, alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name="SubTitle2", parent=styles["Normal"], fontSize=9, leading=13, textColor=TEAL, alignment=TA_CENTER, spaceAfter=15))
styles.add(ParagraphStyle(name="Section2", parent=styles["Heading2"], fontSize=12, leading=15, textColor=TEAL, spaceBefore=10, spaceAfter=5))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#263640"), spaceAfter=6))
styles.add(ParagraphStyle(name="Small2", parent=styles["BodyText"], fontSize=7.2, leading=10, textColor=GRAY, spaceAfter=5))

story = []
def para(text, style="Body2"):
    story.append(Paragraph(text, styles[style]))
def heading(text):
    para(text, "Section2")
def table(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), BLUE),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 7.1),
        ("LEADING", (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF3F6")]),
        ("GRID", (0,0), (-1,-1), .25, colors.HexColor("#CFD9E0")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t); story.append(Spacer(1, 7))
def chart(filename, width=165*mm):
    p = ROOT / "outputs" / "charts" / filename
    if p.exists():
        from PIL import Image as PILImage
        with PILImage.open(p) as im:
            w,h = im.size
        story.append(Image(str(p), width=width, height=width*h/w))
        story.append(Spacer(1, 5))

para("End-to-End House Price Prediction", "MainTitle2")
para("Exploratory Data Analysis • Feature Engineering • Regression • Model Evaluation • Gradio", "SubTitle2")
para("<b>Dataset:</b> Kaggle House Prices: Advanced Regression Techniques (Ames, Iowa). This report uses the actual labelled `train.csv` supplied for the project. Prices are historical USD, not current market values.")

heading("1. Problem statement")
para("Build a supervised machine-learning regression system that estimates historical house sale prices from property characteristics such as size, rooms, quality, condition, construction age, garage, basement, and neighborhood. The final model is exposed through a Gradio interface.")

heading("2. Dataset understanding")
table([
    ["Dataset", "Rows", "Columns", "Purpose"],
    ["train.csv", f"{S['rows_raw']:,}", str(S["columns_raw"]), "Labelled training data; includes SalePrice"],
    ["test.csv", f"{S['test_rows']:,}", "80", "Unlabelled records for final prediction"],
    ["sample_submission.csv", f"{S['test_rows']:,}", "2", "Output format template; prices are placeholders"],
], [34*mm, 22*mm, 23*mm, 100*mm])
para(f"After feature engineering, the model uses <b>{S['feature_count_after_engineering']}</b> predictors. The labelled training data is split into {S['training_rows']:,} training rows and {S['validation_rows']:,} validation rows for model evaluation.")
para(f"`SalePrice` mean: <b>${S['target_mean']:,.0f}</b>; median: <b>${S['target_median']:,.0f}</b>; minimum: <b>${S['target_min']:,.0f}</b>; maximum: <b>${S['target_max']:,.0f}</b>.")

heading("3. Data cleaning and feature engineering")
para("The `Id` field is excluded because it is only an identifier. `MSSubClass` is treated as a categorical dwelling-type code. Missing values are interpreted using the data dictionary: missing basement, garage, fireplace, pool, fence, or alley categories can mean that the feature is absent. Relevant absent-feature measurements are set to zero; other numerical gaps are imputed with the training median, and remaining categorical gaps are filled within the training pipeline.")
para("Engineered features include HouseAge, RemodAge, GarageAge, TotalArea, TotalBathrooms, TotalPorchSF, and indicators such as HasGarage and HasBasement. `PricePerSqFt` is not used as a model input because it is calculated using `SalePrice` and would cause target leakage.")
para("Numeric values are median-imputed and standardized. Categorical values are imputed and one-hot encoded. Preprocessing is fitted inside the cross-validation pipeline to reduce data leakage. The target is log-transformed during fitting and predictions are converted back to USD.")

heading("4. Exploratory data analysis")
para("The charts below show target distribution, area and quality relationships, neighborhood price differences, missing values, and numeric correlations. Correlation and group averages describe associations; they do not prove causation.")
chart("01_saleprice_distribution.png", 165*mm)
chart("02_area_quality_relationships.png", 165*mm)
story.append(PageBreak())

heading("5. Neighborhood and correlation findings")
chart("03_neighborhood_average_price.png", 165*mm)
chart("04_missing_values.png", 145*mm)
chart("06_house_age_price_per_sqft.png", 155*mm)
top_corr = S.get("top_correlations", {})
if top_corr:
    corr_rows = [["Numeric feature", "Correlation with SalePrice"]]
    corr_rows += [[k, f"{v:+.3f}"] for k,v in list(top_corr.items())[:8]]
    table(corr_rows, [75*mm, 75*mm])
chart("05_correlation_heatmap.png", 145*mm)

heading("6. Model development and evaluation")
para("The labelled data is split using an 80/20 train-validation split (random_state=42). Linear Regression is used as a baseline. Random Forest and XGBoost are tuned using randomized hyperparameter search with 3-fold cross-validation. The selected model is chosen using cross-validation RMSE from the training portion; the validation set is retained for final evaluation.")
rows = [["Model", "CV RMSE", "Validation MAE", "Validation RMSE", "Validation R²"]]
for r in S["model_results"]:
    rows.append([
        r["Model"], f"${r['CV RMSE']:,.0f}",
        f"${r['Validation MAE']:,.0f}",
        f"${r['Validation RMSE']:,.0f}",
        f"{r['Validation R2']:.3f}",
    ])
table(rows, [32*mm, 30*mm, 31*mm, 34*mm, 26*mm])
para(f"Selected model by cross-validation: <b>{S['selected_model']}</b>. The validation metrics are reported separately and should be interpreted as an estimate on a held-out subset of this historical dataset.")
if S["selected_model"] == "XGBoost":
    para("On this particular holdout split, Linear Regression has a lower validation RMSE than XGBoost, even though XGBoost has the lowest average cross-validation RMSE. This difference illustrates that model rankings can vary across data splits; cross-validation is used for model selection and the holdout metrics are reported transparently as a separate final check.")
chart("06_actual_vs_predicted.png", 125*mm)
chart("07_residual_distribution.png", 125*mm)

heading("7. Kaggle test prediction")
para(f"The selected model was refitted using all {S['rows_raw']:,} labelled training rows and used to predict the {S['test_rows']:,} records in `test.csv`. The generated `submission.csv` contains `Id` and predicted `SalePrice`, matching the structure of `sample_submission.csv`. Since test.csv has no actual SalePrice values, its accuracy cannot be calculated from the supplied files.")

heading("8. Business recommendations")
para("Historical data can help a real-estate team understand how living area, overall quality, neighborhood, basement, garage, and age relate to sale prices. Use estimates as decision support, not as a replacement for professional appraisal. A real deployment should use recent verified transactions from the target city, show uncertainty intervals, check errors by neighborhood and price range, and retrain as market conditions change.")

heading("9. Limitations and responsible interpretation")
para("The Ames dataset represents historical sales in Ames, Iowa, primarily 2006–2010. Prices are in historical USD. The model is not trained on Nagpur or Indian real-estate data and must not be interpreted as a current property valuation in India. The Gradio form exposes key fields and fills other fields with representative training-data values, so its predictions are illustrative. This is an educational project, not a certified valuation.")

heading("10. Deliverables and reproducibility")
para("Project deliverables include two Jupyter notebooks, reusable Python modules, a saved preprocessing/model pipeline (`artifacts/best_house_price_model.pkl`), model comparison (`artifacts/model_comparison.csv`), EDA/evaluation charts, a Gradio interface (`app/app.py`), and `submission.csv`.")
para("From the project root: install packages with `pip install -r requirements.txt`; run `python train_model.py`; generate this report with `python generate_report.py`; launch the interface with `python app/app.py`.")

heading("11. References")
para("Kaggle — House Prices: Advanced Regression Techniques: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data", "Small2")
para("L&T Fake News Classification reference repository (project organization only): https://github.com/akashkeote/L-and-T-Assignment-Fake-News-Classification", "Small2")

def page_decor(canvas, doc):
    canvas.saveState()
    w,h = A4
    canvas.setStrokeColor(colors.HexColor("#D5DFE6"))
    canvas.line(17*mm, h-14*mm, w-17*mm, h-14*mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GRAY)
    canvas.drawString(17*mm, h-11*mm, "HOUSE PRICE PREDICTION | KAGGLE AMES DATASET")
    canvas.drawRightString(w-17*mm, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

pdf = SimpleDocTemplate(
    str(PDF_PATH), pagesize=A4,
    rightMargin=17*mm, leftMargin=17*mm,
    topMargin=19*mm, bottomMargin=17*mm,
    title="House Price Prediction — Project Report",
    author="Machine Learning Project",
)
pdf.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
print(f"Created {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
