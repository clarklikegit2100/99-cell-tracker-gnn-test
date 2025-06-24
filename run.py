import dotenv
import hydra
import torch_geometric
from omegaconf import DictConfig

# Somewhere early in your code (e.g. run.py or train.py)
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#torch.serialization.add_safe_globals([EarlyStopping])
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#torch.serialization.add_safe_globals([ModelCheckpoint, EarlyStopping])

# only CPU
#torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

# GPU
# remove above torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


#  config_multiLabel config
@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
