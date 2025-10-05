from flask import Flask, render_template, request, jsonify
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import whisper
def get_model():
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class RobertaHateModel(pl.LightningModule):
        def __init__(self, model, lr=2e-5):
            super().__init__()
            self.save_hyperparameters(ignore=["model"])
            self.roberta = model
            self.lr = lr
            self.hate_loss = MSELoss()
            self.binary_loss = BCEWithLogitsLoss()
            hidden_size = self.roberta.config.hidden_size
            self.hate_head = torch.nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]  # [CLS] embedding

        def training_step(self, batch, _):
          device = self.device  # Lightning handles device placement
          input_ids = batch["input_ids"].to(device)
          attention_mask = batch["attention_mask"].to(device)
          hate_score = batch["hate_speech_score"].to(device)

          hidden = self(input_ids, attention_mask)
          hate_pred = self.hate_head(hidden).squeeze(-1)

          loss = self.hate_loss(hate_pred, hate_score)
          self.log("train_loss", loss, prog_bar=True)
          return loss

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        def on_fit_start(self):
            pass

    model = RobertaHateModel(AutoModel.from_pretrained(model_name))
    state_dict = torch.load("roberta-base-targeted-hate.pt", map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("Missing:", missing)
    print("Unexpected:", unexpected)

    return (model, tokenizer)

def eval_text(model, tokenizer, texts):
    cres = []
    for i in range(len(texts) - 2):
        cres.append(' '.join(texts[i:i+2]))
    eval_inputs = tokenizer(
        cres,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    model.eval()
    model.to("cpu")

    inputs = {k: v.to('cpu') for k, v in eval_inputs.items()}

    with torch.no_grad():
        hidden = model(inputs["input_ids"], inputs["attention_mask"])
        hate_pred = model.hate_head(hidden).squeeze(-1)

    return hate_pred.cpu().tolist()

def parse_audio(whisp, audio_path):
    print(audio_path)
    result = whisp.transcribe(audio_path, word_timestamps=True)
    return result['segments']

app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model, tokenizer = get_model()
whisp = whisper.load_model('base')

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith(".mp3"):
        return jsonify({"error": "Only .mp3 files are allowed"}), 400

    # Save the file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filepath)
        parsed = parse_audio(whisp, filepath)

        texts = [x['text'] for x in parsed]
        scores = eval_text(model, tokenizer, texts)
        for i in range(len(scores)):
            if scores[i]>-0.04:
                print(parsed[i]['text'], parsed[i+1]['text'], parsed[i+2]['text'], parsed[i]['start'], parsed[i]['end'], scores[i])

        # Example: return JSON array of timestamps
        results = []
        for i in range(len(scores)):
            if scores[i]>0.01:
                results.append(
                    {'text': f"{parsed[i]['text']}{parsed[i + 1]['text']}{parsed[i + 2]['text']}",
                    'start': parsed[i]['start'], 'end': parsed[i]['end'], 'scores': scores[i]})
    except Exception as e:
        print(e)
        results = []
    try:
        os.remove(filepath)
    except Exception as e:
        print(e)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)