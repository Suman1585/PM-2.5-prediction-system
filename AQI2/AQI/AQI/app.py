import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import csv
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Dense, Layer
from tensorflow.keras import Sequential

app = Flask(__name__)

# Register Mean Squared Error Loss Function
@register_keras_serializable()
class MeanSquaredErrorKeras(MeanSquaredError):
    def __init__(self, reduction="sum_over_batch_size", name="mean_squared_error"):
        super().__init__(reduction=reduction, name=name)

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

lstm_model = load_model('lstm_model.h5', custom_objects={'MeanSquaredError': MeanSquaredErrorKeras, 'TransformerBlock': TransformerBlock})
gru_model = load_model('gru_model.h5', custom_objects={'MeanSquaredError': MeanSquaredErrorKeras, 'TransformerBlock': TransformerBlock})
transformer_model = load_model('transformer_model.h5', custom_objects={'MeanSquaredError': MeanSquaredErrorKeras, 'TransformerBlock': TransformerBlock})
scaler = joblib.load('scaler.pkl')

FEATURES = ['co', 'no', 'no2', 'o3', 'so2', 'pm10', 'nh3']
LOOK_BACK = 24
FORECAST_HORIZON = 3

history_data = np.zeros((LOOK_BACK, len(FEATURES)))

def load_suggestions():
    suggestions = []
    with open('suggestions.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:

            pm_range = row['PM2.5 Range'].split('-')
            pm_range = [int(value) for value in pm_range]
            suggestions.append({
                'range': pm_range,
                'suggestion': row['Suggestion']
            })
    return suggestions

def get_suggestion(pm25_value):
    suggestions = load_suggestions()
    for suggestion in suggestions:
        if suggestion['range'][0] <= pm25_value <= suggestion['range'][1]:
            return suggestion['suggestion']
    return "No suggestion available."

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    suggestion = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form[feat]) for feat in FEATURES]

            dummy_targets = [0]  # target
            data_row = np.array(inputs + dummy_targets).reshape(1, -1)
            scaled_row = scaler.transform(data_row)[0][:-1]  # drop target after scaling

            input_seq = history_data.copy()
            input_seq[-1] = scaled_row
            input_seq = input_seq.reshape(1, LOOK_BACK, len(FEATURES))

            lstm_pred = lstm_model.predict(input_seq)[0]
            gru_pred = gru_model.predict(input_seq)[0]
            transformer_pred = transformer_model.predict(input_seq)[0]
            pm25_value = lstm_pred[0]

            suggestion = get_suggestion(pm25_value)

            predictions = {
                'LSTM': [round(x, 3) for x in lstm_pred],
                'GRU': [round(x, 3) for x in gru_pred],
                'Transformer': [round(x, 3) for x in transformer_pred],
            }

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html', features=FEATURES, predictions=predictions, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)