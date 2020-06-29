"""
Contains all string constants relevant to the project
"""

# String concerning logs
original_consumer = 'original_consumer_id'
consumer = 'consumer_id'
timestamp = 'timestamp'
original_invoice = 'original_invoice_id'
invoice = 'invoice_id'
sku = 'sku'
item_count = 'item_count'
sku_count = 'sku_count'
value = 'value'
avg_value = 'avg_value'
label = 'label'
sku_des = 'sku_des'
cat = 'category'
enc = 'encoding'
id_features = [consumer, timestamp, invoice]
supercat = f'super-{cat}'
human_names_dict = {sku_count: 'Qty', sku_des: 'Product', value: 'Price', item_count: '#'}
unknown = 'unknown'

# General ML / Classification
fm_str = 'feature_mode'
lab_str = 'labelled'
bal_str = 'balanced'
clf_str = 'classifier'
acc_str = 'acc'
f1_str = 'f1'
f1_class_base_str = 'f1_{}'
IEM = 'iem'
IEM_tex = r'$\Phi$'
ARI = 'ARI'
ARI_tex = '$ARI$'

# SSL Specific
ssl_str = 'ssl'
k_str = 'k'
alpha_str = 'alpha'
n_lab_str = 'n_lab'
n_unlab_str = 'n_unlab'
comment_str = 'comment'

# Durations
train_time_str = 'train_time'
test_time_str = 'test_time'

# Tool
RECEIPT_SHOWN = 'receipt_shown'
END_TIME = 'end_time'
START_TIME = 'start_time'
GIVEN_ANSWER = 'given_answer'
NOTES = 'notes'
USER = 'user'
WALVIS_MODE = 'mode'
ASSIGNMENT_SET = 'assignment_set'


def hl_name(i):
    """
    Returns the correct naming of a level in the hierarchy

    Parameters
    ----------
    i : int
        Hierarchy level required. Must be positive

    Returns
    -------
    hierarchy_level_name : String
        Name of the level in the hierarchy at level ``i``
    """
    assert isinstance(i, int)
    assert i > 0
    return f'{cat}_{i}'


def hl_encoding(i):
    """
    Returns the correct encoding of a level in the hierarchy

    Parameters
    ----------
    i : int
        Hierarchy level required. Must be non-negative

    Returns
    -------
    hierarchy_level_encoding : String
        Encoding name of the level in the hierarchy at level ``i``
    """
    assert isinstance(i, int)
    assert i >= 0
    if i == 0:
        return f'{cat}_{enc}'
    else:
        return f'{cat}_{enc}_{i}'


def is_hierarchy_level_name(hln):
    """
    Verifies whether a given string is a hierarchy level name

    Parameters
    ----------
    hln : String
        The text to be checked

    Returns
    -------
    is_hln : Boolean
        True if ``hln`` is a correctly formatted hierarchy level, False otherwise
    """
    return hln.count('_') == 1 and hln.split('_')[0] == cat and hln.split('_')[1].isdigit()


# Bootstrapping
OPTIMIZATION_TIME = 'optimization'
CLUSTERING_TIME = 'clustering'

SUPPORT = "support"
cluster_size = 'cluster_size'
medoid = 'medoid'
receipt = 'receipt'

REPETITION = 'repetition'
ITERATION = 'iteration'
ISC_WEIGHT = 'w_s'
ISC_WEIGHT_LATEX = '$w_s$'
fp_id = 'fixed point'
fpw_str = 'fixed point weight string'
multiplicity = 'multiplicity'

CLUSTER = 'cluster'
OTHER_CLUSTER = 'other_cluster'
OTHER_CLUSTER_SIZE = 'other_cluster_size'
DATASET_NAME = 'dataset_name'
DISTANCE = 'distance'
DISTANCE_TO_OTHER = 'distance_to_other'
MEDOID_OTHER = 'medoid_other'
RP = 'rp'
EXPERIMENT_NAME = 'experiment_name'

OTHER_LABEL = 'other_label'

super_medoid = 'super_medoid'

ALGORITHM = 'algorithm'
OUR_FRAMEWORK = 'Our framework'
