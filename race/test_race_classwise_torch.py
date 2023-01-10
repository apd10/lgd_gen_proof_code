import pytest
from race.race_classwise_torch import Race_Classwise_torch
import numpy as np 
import torch
from functools import reduce

# RACE Unit tests
# Properties to be assessed:
# - Independence of each row
# - Score is higher when there are more similar elements
# - Test weights are incorporated in score
# - Scores are as expected? But what is the expectation?
# - Repeat the above tests for classwise inserts

INPUT_DIMENSION = 8

RACE_PARAMS = {
    "num_classes": 2,
    "num_bucket": 4,  
    "device_id" : -1, # to do on cpu. test device as well later
    "recovery" : "mean", # mean recovery
    "lsh_function": {
        "name": "srp_torch",
        "srp_torch": {
            "dimension": INPUT_DIMENSION,
            "num_bits": 2,
            "num_hash" : 5,
            "seed": 101,
            "device_id": -1,
        }
    }
}


def test_rows_independent():
    """Checks that each row in the RACE sketch uses an independent hash 
    function. We do this by asserting that none of the samples we pass into 
    race gets the same count from every row of the sketch.
    """
    race = Race_Classwise_torch(RACE_PARAMS)
    
    num_samples = 1000
    
    # Normally distributed random dataset
    dataset = np.random.normal(
        loc=0.0, 
        scale=1.0, 
        size=(num_samples, INPUT_DIMENSION))
    dataset = torch.from_numpy(dataset)
    dataset = dataset.type(torch.FloatTensor)
    
    # Alpha is 1.0 for all datasets
    alpha = torch.ones(size=(num_samples,)).type(torch.FloatTensor)
    
    race.add(dataset, alpha)

    # Returns a num_hashes x num_samples matrix;
    # Each row contains the counts for all samples from a row of the RACE sketch.
    rowwise_counts = race.query_all_rows(dataset)
    
    # The following reduction is equivalent to applying an elementwise logical 
    # "and" operator over [scores == rowwise_scores[0] for scores in rowwise_scores].
    #
    # scores and rowwise_scores[0] are both vectors with num_samples elements.
    # scores == rowwise_scores[0] creates a binary vector with num_samples 
    # elements, representing element-wise equality of scores and rowwise_scores[0]
    equal_counts_across_race_rows = reduce(
        lambda accumulator, counts: accumulator * (counts == rowwise_counts[0]), 
        rowwise_counts,
        # Accumulator is initialized as a vector of ones
        torch.ones(num_samples))
    
    # Assert that no sample gets equal counts from every row of RACE.
    assert not any(equal_counts_across_race_rows)




# - Score is higher when there are more similar elements
# - Test weights are incorporated in score
# - Scores are as expected? But what is the expectation?
# - Repeat the above tests for classwise inserts