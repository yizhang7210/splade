import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import expit

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def index(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    d_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["COLLECTION_PATH"], id_style="row_id")
    d_loader = CollectionDataLoader(dataset=d_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                    max_length=model_training_config["max_length"],
                                    batch_size=config["index_retrieve_batch_size"],
                                    shuffle=False, num_workers=10, prefetch_factor=4)
    evaluator = SparseIndexing(model=model, config=config, compute_stats=True)
    # evaluator.index(d_loader)
    
    passages = []
    for i in range(len(d_collection)):
        passages.append(d_collection[i][1])
    
    MODEL = f"cardiffnlp/tweet-topic-21-multi"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # PT
    batch_size = 100
    num_passages = len(passages)
    result = np.empty(shape=(0, 19))
    
    num_batches = num_passages // batch_size + 1 if num_passages % batch_size > 0 else num_passages // batch_size
    
    
    for i in range(num_batches):
        print(f"indexing batch {i}")
        start = i * batch_size
        end = (i +1) * batch_size

        tokens = tokenizer(passages[start:end], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        output = model(**tokens)
        batch_result = output[0].detach().numpy()
        result = np.vstack([result, batch_result])
        print(result.shape)
        
    print("DONE")


if __name__ == "__main__":
    index()
