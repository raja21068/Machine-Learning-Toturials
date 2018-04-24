import sys

# To show some messages:
import recsys.algorithm
# recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
from recsys.datamodel.data import Data
from recsys.evaluation.prediction import RMSE, MAE
from recsys.evaluation.decision import PrecisionRecallF1
from recsys.evaluation.ranking import SpearmanRho, KendallTau


# Dataset
PERCENT_TRAIN = 70
data = Data()
data.load('./data/dataset-recsys.csv', sep=',', format={'col': 0, 'row': 1, 'value': 2, 'ids': int})

# Train & Test data
train, test = data.split_train_test(percent=PERCENT_TRAIN)

# Create SVD
K = 100
svd = SVD()
svd.set_data(train)

svd.compute(k=K, min_values=1, pre_normalize=None, mean_center=True, post_normalize=True)
# svd.compute(k=K, min_values=5, pre_normalize=None, mean_center=True, post_normalize=True)
# svd.compute(k=K, pre_normalize=None, mean_center=True, post_normalize=True)

print
('')
print('COMPUTING SIMILARITY')
print(svd.similarity(1, 2))  # similarity between items
print(svd.similar(1, 5) ) # show 5 similar items

print('GENERATING PREDICTION')
MIN_RATING = 0.0
MAX_RATING = 5.0
ITEMID = 1
USERID = 1
print(svd.predict(ITEMID, USERID, MIN_RATING, MAX_RATING))  # predicted rating value
print(svd.get_matrix().value(ITEMID, USERID))  # real rating value

print('')
print('GENERATING RECOMMENDATION')
print(svd.recommend(USERID, n=5, only_unknowns=True, is_row=False))

# Evaluation using prediction-based metrics
rmse = RMSE()
mae = MAE()
spearman = SpearmanRho()
kendall = KendallTau()
# decision = PrecisionRecallF1()
for rating, item_id, user_id in test.get():
    try:
        pred_rating = svd.predict(item_id, user_id)
        rmse.add(rating, pred_rating)
        mae.add(rating, pred_rating)
        spearman.add(rating, pred_rating)
        kendall.add(rating, pred_rating)
    except KeyError:
        continue

print
('')
print('EVALUATION RESULT')
print('RMSE=%s' % rmse.compute())
print('MAE=%s' % mae.compute())
print('Spearman\'s rho=%s' % spearman.compute())
print('Kendall-tau=%s' % kendall.compute())
