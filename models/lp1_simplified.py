from prediction.predictor import predict_scores, predict_vector
from sklearn.linear_model import LinearRegression
from utils.critique import LP1SimplifiedOptimize

import numpy as np


class LP1Simplified(object):
    def __init__(self, keyphrase_freq, item_keyphrase_freq, row, matrix_Train, matrix_Test, test_users,
                 target_ranks, num_items_sampled, num_keyphrases, df,
                 max_iteration_threshold, keyphrase_popularity, dataset_name,
                 model, parameters_row, **unused):
        self.keyphrase_freq = keyphrase_freq
        self.item_keyphrase_freq = item_keyphrase_freq.T
        self.row = row
        self.matrix_Train = matrix_Train
        self.num_users, self.num_items = matrix_Train.shape
        self.matrix_Test = matrix_Test
        self.test_users = test_users
        self.target_ranks = target_ranks
        self.num_items_sampled = num_items_sampled
        self.num_keyphrases = num_keyphrases
        self.df = df
        self.max_iteration_threshold = max_iteration_threshold
        self.keyphrase_popularity = keyphrase_popularity
        self.dataset_name = dataset_name
        self.model = model
        self.parameters_row = parameters_row

    def start_critiquing(self):
        self.get_initial_predictions()

        for user in self.test_users:
            # User id starts from 0
            self.row['user_id'] = user

            # The iteration will stop if the wanted item is in top n
            for target_rank in self.target_ranks:
                self.row['target_rank'] = target_rank
                # Pick wanted items in test items
                candidate_items = self.matrix_Test[user].nonzero()[1]
                train_items = self.matrix_Train[user].nonzero()[1]
                wanted_items = np.setdiff1d(candidate_items, train_items)

                for item in wanted_items:
                    # Item id starts from 0
                    self.row['item_id'] = item
                    # Set the wanted item's initial rank as None
                    self.row['item_rank'] = None
                    # Set the wanted item's initial prediction score as None
                    self.row['item_score'] = None

                    # Get the item's existing keyphrases
                    item_keyphrases = self.item_keyphrase_freq[item].nonzero()[0]
                    # Get keyphrases that don't belong to the item (we can critique)
                    remaining_keyphrases = np.setdiff1d(range(self.num_keyphrases), item_keyphrases)
#                    print("The number of remaining_keyphrases is {}. remaining_keyphrases are: {}".format(len(remaining_keyphrases), remaining_keyphrases))
                    self.row['num_existing_keyphrases'] = len(remaining_keyphrases)
                    if len(remaining_keyphrases) == 0:
                        break
                    self.row['iteration'] = 0
                    self.row['critiqued_keyphrase'] = None
                    self.row['result'] = None
                    self.df = self.df.append(self.row, ignore_index=True)

                    query = []
                    affected_items = np.array([])

                    for iteration in range(self.max_iteration_threshold):
                        self.row['iteration'] = iteration + 1
                        # Always critique the most popular keyphrase
                        critiqued_keyphrase = remaining_keyphrases[np.argmax(self.keyphrase_popularity[remaining_keyphrases])]
#                        print("remaining keyphrases popularity: {}".format(self.keyphrase_popularity[remaining_keyphrases]))
                        self.row['critiqued_keyphrase'] = critiqued_keyphrase
                        query.append(critiqued_keyphrase)

                        # Get affected items (items have critiqued keyphrase)
                        current_affected_items = self.item_keyphrase_freq[:, critiqued_keyphrase].nonzero()[0]
                        affected_items = np.unique(np.concatenate((affected_items, current_affected_items))).astype(int)
                        unaffected_items = np.setdiff1d(range(self.num_items), affected_items)

                        if iteration == 0:
                            prediction_items = predict_vector(rating_vector=self.prediction_scores[user],
                                                              train_vector=self.matrix_Train[user],
                                                              remove_train=True)

                        affected_items_mask = np.in1d(prediction_items, affected_items)
                        affected_items_index_rank = np.where(affected_items_mask == True)
                        unaffected_items_index_rank = np.where(affected_items_mask == False)

                        import copy
                        prediction_scores_u, lambdas = LP1SimplifiedOptimize(initial_prediction_u=self.prediction_scores[user],
                                                                             keyphrase_freq=copy.deepcopy(self.keyphrase_freq),
                                                                             affected_items=np.intersect1d(affected_items, prediction_items[affected_items_index_rank[0][:100]]),
                                                                             unaffected_items=np.intersect1d(unaffected_items, prediction_items[unaffected_items_index_rank[0][:100]]),
                                                                             num_keyphrases=self.num_keyphrases,
                                                                             query=query,
                                                                             test_user=user,
                                                                             item_latent=self.Y,
                                                                             reg=self.reg)

                        self.row['lambda'] = lambdas
                        prediction_items = predict_vector(rating_vector=prediction_scores_u,
                                                          train_vector=self.matrix_Train[user],
                                                          remove_train=True)
                        recommended_items = prediction_items

                        # Current item rank
                        item_rank = np.where(recommended_items == item)[0][0]

                        self.row['item_rank'] = item_rank
                        self.row['item_score'] = prediction_scores_u[item]

                        if item_rank + 1 <= target_rank:
                            # Items is ranked within target rank
                            self.row['result'] = 'successful'
                            self.df = self.df.append(self.row, ignore_index=True)
                            break
                        else:
                            remaining_keyphrases = np.setdiff1d(remaining_keyphrases, critiqued_keyphrase)
                            # Continue if more keyphrases and iterations remained
                            if len(remaining_keyphrases) > 0 and self.row['iteration'] < self.max_iteration_threshold:
                                self.row['result'] = None
                                self.df = self.df.append(self.row, ignore_index=True)
                            else:
                                # Otherwise, mark fail
                                self.row['result'] = 'fail'
                                self.df = self.df.append(self.row, ignore_index=True)
                                break
        return self.df


    def get_initial_predictions(self):
        self.RQ, Yt, Bias = self.model(self.matrix_Train,
                                       iteration=self.parameters_row['iter'].values[0],
                                       lamb=self.parameters_row['lambda'].values[0],
                                       rank=self.parameters_row['rank'].values[0])
        self.Y = Yt.T

        self.reg = LinearRegression().fit(self.keyphrase_freq, self.RQ)

        self.prediction_scores = predict_scores(matrix_U=self.RQ,
                                                matrix_V=self.Y,
                                                bias=Bias)

