from pathlib import Path

# run as soon as imported
home = Path(__file__).resolve().parents[1]

log_path = home / 'logs'
data_path = home / 'data'
model_path = home / 'models'
src = home / 'src'
data_src = home / 'data'
model_src = home / 'model'