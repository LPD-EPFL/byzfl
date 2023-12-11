import json

with open('config.json') as config_file:
    config = json.load(config_file)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_loader_dict, test_loader = dataset.make_train_test_datasets(config)
	
server = 

