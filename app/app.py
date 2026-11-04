"""Gradio frontend. From the project root run: python app/app.py"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from src.predictor import load_model_bundle, predict_from_inputs

bundle = load_model_bundle()
NEIGHBORHOODS = bundle["neighborhoods"]
YEAR_MIN = bundle["sale_year_min"]
YEAR_MAX = bundle["sale_year_max"]
YEAR_BUILT_MIN = bundle["year_built_min"]

def estimate(living_area, basement_area, bedrooms, total_rooms, quality,
             condition, garage_cars, year_built, year_sold, neighborhood):
    try:
        price = predict_from_inputs(
            living_area=float(living_area),
            basement_area=float(basement_area),
            bedrooms=int(bedrooms),
            total_rooms=int(total_rooms),
            quality=int(quality),
            condition=int(condition),
            garage_cars=int(garage_cars),
            year_built=int(year_built),
            year_sold=int(year_sold),
            neighborhood=neighborhood,
            bundle=bundle,
        )
    except (TypeError, ValueError) as exc:
        raise gr.Error(str(exc))

    return (
        f"## Estimated historical sale price: ${price:,.0f} USD\n\n"
        f"**Model:** {bundle['selected_model']}\n\n"
        f"**Dataset:** Historical Ames, Iowa sales ({YEAR_MIN}–{YEAR_MAX}).\n\n"
        "The model fills property characteristics not shown in this simple form "
        "with representative training-data values. This is an educational estimate, "
        "not a current appraisal or a Nagpur/India property valuation."
    )

with gr.Blocks(title="Ames House Price Prediction") as demo:
    gr.Markdown(
        "# House Price Prediction\n"
        "Enter key property details to estimate a historical sale price using a model "
        "trained on the Kaggle Ames dataset."
    )
    gr.Markdown(
        f"> **Dataset scope:** Ames, Iowa, historical sales from {YEAR_MIN} to {YEAR_MAX}. "
        "Prices are in USD. Other property characteristics use typical training-data values."
    )
    with gr.Row():
        with gr.Column():
            living_area = gr.Number(value=1500, label="Above-ground living area (sq ft)", minimum=1)
            basement_area = gr.Number(value=700, label="Total basement area (sq ft)", minimum=0)
            bedrooms = gr.Number(value=3, label="Bedrooms above ground", precision=0, minimum=0)
            total_rooms = gr.Number(value=7, label="Total rooms above ground", precision=0, minimum=1)
            quality = gr.Slider(1, 10, value=6, step=1, label="Overall quality (1 low – 10 excellent)")
        with gr.Column():
            condition = gr.Slider(1, 10, value=5, step=1, label="Overall condition (1 poor – 10 excellent)")
            garage_cars = gr.Number(value=2, label="Garage capacity (cars)", precision=0, minimum=0)
            year_built = gr.Number(value=2000, label="Year built", precision=0, minimum=YEAR_BUILT_MIN, maximum=YEAR_MAX)
            year_sold = gr.Number(value=YEAR_MAX, label="Historical sale year", precision=0, minimum=YEAR_MIN, maximum=YEAR_MAX)
            neighborhood = gr.Dropdown(choices=NEIGHBORHOODS, value="NAmes" if "NAmes" in NEIGHBORHOODS else NEIGHBORHOODS[0], label="Neighborhood")
    predict_button = gr.Button("Estimate House Price", variant="primary")
    output = gr.Markdown()
    predict_button.click(
        estimate,
        inputs=[living_area, basement_area, bedrooms, total_rooms, quality,
                condition, garage_cars, year_built, year_sold, neighborhood],
        outputs=output,
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
