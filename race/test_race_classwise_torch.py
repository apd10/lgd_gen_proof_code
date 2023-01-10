import pytest
from race.race_classwise_torch import Race_Classwise_torch
import numpy as np 
import torch
from functools import reduce
import random

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
            "num_hash" : 10,
            "seed": 101,
            "device_id": -1,
        }
    }
}

def random_dataset(num_samples, mean=0.0, scale=1.0):
    """Creates a set of normally distributed vectors around a mean
    and converts it to a torch float tensor.
    """
    dataset = np.random.normal(
        loc=mean, 
        scale=scale, 
        size=(num_samples, INPUT_DIMENSION))
    return torch.from_numpy(dataset).type(torch.FloatTensor)

def test_rows_independent():
    """Checks that each row in the RACE sketch uses an independent hash 
    function. We do this by asserting that none of the samples we pass into 
    race gets the same count from every row of the sketch.
    """
    
    num_samples = 1000
    dataset = random_dataset(num_samples)
    alpha = torch.ones(size=(num_samples,))
    
    race = Race_Classwise_torch(RACE_PARAMS)
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


def test_more_populous_neighborhood_gets_higher_score():
    """Checks that RACE gives a higher score to elements that have more similar
    elements than elements with fewer similar elements.
    """
    # Neighborhood A has a much larger population than neighborhood B
    A_population = 1000
    B_population = 100
    
    # Means are 180˚ apart; their hashes will never collide.
    # Their neighbors may collide though.
    A_mean = np.random.normal(0.0, scale=1.0, size=(INPUT_DIMENSION,))
    B_mean = -A_mean
    
    # Use smaller scale to generate neighborhood so that neighbors are much
    # closer to their respective centers than elements of the other neighborhood.
    neighborhood_A = random_dataset(A_population, A_mean, scale=0.5)
    neighborhood_B = random_dataset(B_population, B_mean, scale=0.5)

    race = Race_Classwise_torch(RACE_PARAMS)
    race.add(neighborhood_A, alpha=torch.ones(size=(A_population,)))
    race.add(neighborhood_B, alpha=torch.ones(size=(B_population,)))

    A_average_score = torch.mean(race.query(neighborhood_A))
    B_average_score = torch.mean(race.query(neighborhood_B))
    # Factor of 5 instead of 10 because there may be hash collisions between
    # elements of the different neighborhoods.
    assert A_average_score > 5 * B_average_score


def test_scores_account_for_alpha():
    """Checks RACE counts are correctly weighted by alpha for each sample. 
    To test this, we pass pass in a dataset consisting of two neighborhoods,
    one with 10 times the alpha of the other. We then check that the scores 
    reflect this difference in alpha.
    """
    race = Race_Classwise_torch(RACE_PARAMS)
    neighborhood_population = 100
    
    # Means are 180˚ apart; their hashes will never collide. 
    # Their neighbors may collide though.
    A_mean = np.random.normal(0.0, scale=1.0, size=(INPUT_DIMENSION,))
    B_mean = -A_mean
    
    # Use smaller scale to generate neighborhood so that neighbors are much
    # closer to their respective centers than elements of the other neighborhood.
    neighborhood_A = random_dataset(neighborhood_population, A_mean, scale=0.5)
    neighborhood_B = random_dataset(neighborhood_population, B_mean, scale=0.5)
    combined_neighborhoods = torch.concat([neighborhood_A, neighborhood_B])

    # Neighborhood A elements have 10 times the alpha.
    A_alphas = torch.ones(size=(neighborhood_population,)) * 10
    B_alphas = torch.ones(size=(neighborhood_population,))
    combined_alphas = torch.concat([A_alphas, B_alphas])

    race = Race_Classwise_torch(RACE_PARAMS)
    race.add(combined_neighborhoods, combined_alphas)

    A_average_score = torch.mean(race.query(neighborhood_A))
    B_average_score = torch.mean(race.query(neighborhood_B))
    
    # Factor of 5 instead of 10 because there may be hash collisions between
    # elements of the different neighborhoods.
    assert A_average_score > 5 * B_average_score


# TODO(Geordie): Repeat the above tests for classwise inserts