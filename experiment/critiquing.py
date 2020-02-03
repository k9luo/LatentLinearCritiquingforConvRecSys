from utils.critique import sample_users
from utils.modelnames import critiquing_models

import numpy as np
import pandas as pd


def critiquing(matrix_Train, matrix_Test, keyphrase_freq, dataset_name, model,
               parameters_row, critiquing_model_name, item_keyphrase_freq=None, num_users_sampled=10,
               num_items_sampled=5, max_iteration_threshold=20):

    num_users = matrix_Train.shape[0]
    num_keyphrases = keyphrase_freq.shape[1]

    keyphrase_popularity = np.sum(item_keyphrase_freq, axis=1)

    columns = ['user_id', 'item_id', 'target_rank', 'iteration', 'critiqued_keyphrase', 'item_rank', 'item_score', 'num_existing_keyphrases', 'result', 'lambda']
    df = pd.DataFrame(columns=columns)

    row = {}

    target_ranks = [1, 5, 10]

    # Randomly select test users
    test_users = sample_users(num_users, num_users_sampled)

    critiquing_model = critiquing_models[critiquing_model_name](keyphrase_freq=keyphrase_freq,
                                                                item_keyphrase_freq=item_keyphrase_freq,
                                                                row=row,
                                                                matrix_Train=matrix_Train,
                                                                matrix_Test=matrix_Test,
                                                                test_users=test_users,
                                                                target_ranks=target_ranks,
                                                                num_items_sampled=num_items_sampled,
                                                                num_keyphrases=num_keyphrases,
                                                                df=df,
                                                                max_iteration_threshold=max_iteration_threshold,
                                                                keyphrase_popularity=keyphrase_popularity,
                                                                dataset_name=dataset_name,
                                                                model=model,
                                                                parameters_row=parameters_row)

    df = critiquing_model.start_critiquing()

    return df

