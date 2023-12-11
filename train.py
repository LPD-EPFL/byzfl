import json

with open('config.json') as config_file:
    config = json.load(config_file)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

