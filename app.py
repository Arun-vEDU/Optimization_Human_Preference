import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dash import Dash, dcc, html, Input, Output

# Load the trained model and tokenizer
model_name = "dpo-finetuned-model"  # Replace with the path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("DPO Fine-Tuned Model Demo"),
    html.Div([
        html.Label("Enter your input text:"),
        dcc.Input(id="input-text", type="text", placeholder="Type something...", style={"width": "100%"}),
    ]),
    html.Br(),
    html.Div(id="output-text", style={"whiteSpace": "pre-line"})
])

# Define the callback to generate responses
@app.callback(
    Output("output-text", "children"),
    Input("input-text", "value")
)
def generate_response(input_text):
    if not input_text:
        return "Please enter some text to get a response."

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate a response using the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly pass the attention mask
            max_length=50,  # Adjust max_length as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"Model Response:\n{response}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)