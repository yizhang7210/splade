import hydra
from omegaconf import DictConfig

from conf.CONFIG_CHOICE import CONFIG_NAME, CONFIG_PATH
from .datasets.dataloaders import CollectionDataLoader
from .datasets.datasets import CollectionDatasetPreLoad
from .evaluate import evaluate
from .models.models_utils import get_model
from .tasks.transformer_evaluator import SparseRetrieval
from .utils.utils import get_dataset_name, get_initialize_config
import h5py
import numpy as np
import pandas as pd
import time
from scipy.sparse import csr_matrix

FILTER_BY_TOPIC = True
PASSAGE_TOPIC_THRESHOLD = 0.25
QUERY_TOPIC_THRESHOLD = 0.25

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def retrieve_evaluate(exp_dict: DictConfig):
    exp_dict, config, init_dict, model_training_config = get_initialize_config(exp_dict)

    model = get_model(config, init_dict)

    passage_topic_scores = []
    query_topic_scores = []
    if FILTER_BY_TOPIC:
        # Construct sparse map of passage scores
        topic_index_file = h5py.File('data/toy_data/full_collection/full_collection_classifications.h5', 'r')
        topic_index = topic_index_file['classifications'][()]
        topic_index_file.close()

        for passage_i in range(topic_index.shape[0]):
            topic_map = set()
            for topic_i in range(topic_index.shape[1]):
                topic_score = topic_index[passage_i, topic_i]
                if topic_score > PASSAGE_TOPIC_THRESHOLD:
                    topic_map.add(topic_i)

            passage_topic_scores.append(topic_map)

        # Get the topic of the query
        q_topic_index_file = h5py.File('data/toy_data/dev_queries/query_classifications.h5', 'r')
        q_topic_index = q_topic_index_file['classifications'][()]
        q_topic_index_file.close()

        for query_i in range(q_topic_index.shape[0]):
            topic_map = set()
            for topic_i in range(q_topic_index.shape[1]):
                topic_score = q_topic_index[query_i, topic_i]
                if topic_score > QUERY_TOPIC_THRESHOLD:
                    topic_map.add(topic_i)

            query_topic_scores.append(topic_map)

    batch_size = 1
    # NOTE: batch_size is set to 1, currently no batched implem for retrieval (TODO)
    for data_dir in set(exp_dict["data"]["Q_COLLECTION_PATH"]):
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = CollectionDataLoader(dataset=q_collection, tokenizer_type=model_training_config["tokenizer_type"],
                                        max_length=model_training_config["max_length"], batch_size=batch_size,
                                        shuffle=False, num_workers=1)
        evaluator = SparseRetrieval(config=config, model=model, dataset_name=get_dataset_name(data_dir),
                                    compute_stats=True, dim_voc=model.output_dim)

        start = time.time()
        evaluator.retrieve(q_loader,
                           top_k=exp_dict["config"]["top_k"],
                           threshold=exp_dict["config"]["threshold"],
                           passage_topic_scores=passage_topic_scores,
                           query_topic_scores=query_topic_scores,
                           filter_by_topic=FILTER_BY_TOPIC)
        finish = time.time()

        print(f"Retrieval took {finish-start} seconds")

    evaluate(exp_dict)



if __name__ == "__main__":
    retrieve_evaluate()
