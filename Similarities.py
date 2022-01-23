import math
from icecream import ic

def sim_pearson(ratings1, ratings2):
    #ic("sm.sim_pearson()")
    
    # Get co-rated
    co_rated = []
    
    for id in ratings1: 
        
        if id in ratings2: 
            co_rated.append(id)
            
    n = len(co_rated)
    
    # If no co-rated, return 0
    if n == 0: 
        return 0.0
    
    # Means over co-rated
    mean1 = sum([ratings1[id] for id in co_rated]) * 1.0 / n
    mean2 = sum([ratings2[id] for id in co_rated]) * 1.0 / n
    
    # Sums of products of differences and sums of squared differences
    sum_prods = 0.0
    sum_squares1 = 0.0
    sum_squares2 = 0.0
    
    for id in co_rated:
        difference1 = ratings1[id] - mean1
        difference2 = ratings2[id] - mean2
        sum_prods = sum_prods + difference1 * difference2
        sum_squares1 = sum_squares1 + difference1 * difference1
        sum_squares2 = sum_squares2 + difference2 * difference2
        
    # Handle zero variances
    if sum_squares1 == 0 or sum_squares2 == 0:
        return 0.0
    
    # Calculate Pearson correlation
    return sum_prods / (math.sqrt(sum_squares1) * math.sqrt(sum_squares2))


def sim_cosine(ratings1, ratings2):
    #ic("sm.sim_cosine()")
    
    if len(ratings1) == 0 or len(ratings2) == 0:
        return 0.0
    
    return dot_product(ratings1, ratings2) / (magnitude(ratings1) * magnitude(ratings2))


def dot_product(ratings1, ratings2):
    #ic("sm.dot_product()")

    dot_product = 0.0
    
    for id in ratings1:
        
        if id in ratings2:
            dot_product = dot_product + ratings1[id] * ratings2[id]
            
    return dot_product
    
    
def magnitude(ratings):
    #ic("sm.magnitude()")

    sum_squares = 0.0
    
    for rating in ratings.values():
        sum_squares = sum_squares + rating * rating
        
    return math.sqrt(sum_squares)
    