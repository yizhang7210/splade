import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import h5py
import pandas as pd
from scipy.special import expit

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseIndexing
from .utils.utils import get_initialize_config
from tqdm.auto import tqdm

INDEX_PASSAGES = True
INDEX_QUERIES = False

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
    
    # Topic indexing of all the passages too
    if INDEX_PASSAGES:
        passages = []
        for i in range(len(d_collection)):
            passages.append(d_collection[i][1])

        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # PT
        batch_size = 1
        num_passages = len(passages)
        result = np.empty(shape=(0, 19))

        num_batches = num_passages // batch_size + 1 if num_passages % batch_size > 0 else num_passages // batch_size

        classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = (i +1) * batch_size

            tokens = tokenizer(passages[start:end], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
            output = classification_model(**tokens)
            batch_result = output[0].detach().numpy()
            result = np.vstack([result, batch_result])

        classifications = pd.DataFrame(result)

        transformed = expit(classifications)

        h5f = h5py.File('full_collection_classifications.h5', 'w')
        h5f.create_dataset('classifications', data=transformed)
        h5f.close()
        transformed.to_csv("full_collection_classifications.csv", index=False)

        print("DONE")
 

    if INDEX_QUERIES:
        q_collection = CollectionDatasetPreLoad(data_dir=exp_dict["data"]["Q_COLLECTION_PATH"][0], id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=1,
                                        shuffle=False, num_workers=1)

        queries = []
        for i in range(len(q_collection)):
            queries.append(q_collection[i][1])

        MODEL = f"cardiffnlp/tweet-topic-21-multi"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # PT
        batch_size = 10
        num_queries = len(queries)
        result = np.empty(shape=(0, 19))

        num_batches = num_queries // batch_size + 1 if num_queries % batch_size > 0 else num_queries // batch_size

        classification_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        for i in range(num_batches):
            print(f"indexing batch {i} of query")
            start = i * batch_size
            end = (i +1) * batch_size

            tokens = tokenizer(queries[start:end], truncation=True, max_length=512, padding='max_length', return_tensors='pt')
            output = classification_model(**tokens)
            batch_result = output[0].detach().numpy()
            result = np.vstack([result, batch_result])
            print(result.shape)

        classifications = pd.DataFrame(result)
        classifications.to_csv("query_classifications.csv")
        print("DONE")

if __name__ == "__main__":
    index()
