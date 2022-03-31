"""

"""


import math


class Similarities:
    def __init__(self) -> None:
        pass
        
        
    @staticmethod
    def sim_mod(ratings1: dict, ratings2: dict) -> float:
        
        co_rated = []
        
        for id in ratings1: 
            
            if id in ratings2: 
                co_rated.append(id)
                
        n = len(co_rated)
        
        if n == 0: 
            return 0.0
        
        # means over co-rated
        mean1 = sum([ratings1[id] for id in co_rated]) * 1.0 / n
        mean2 = sum([ratings2[id] for id in co_rated]) * 1.0 / n
        
        # sums of products of differences and sums of squared differences
        sum_prods = 0.0
        sum_squares1 = 0.0
        sum_squares2 = 0.0
        
        for id in co_rated:
            difference1 = ratings1[id] - mean1
            difference2 = ratings2[id] - mean2
            sum_prods = sum_prods + difference1 * difference2
            sum_squares1 = sum_squares1 + difference1 * difference1
            sum_squares2 = sum_squares2 + difference2 * difference2
            
        # handle zero variances
        if sum_squares1 == 0 or sum_squares2 == 0:
            return 0.0
        
        # calculate Pearson correlation
        pear =  sum_prods / (math.sqrt(sum_squares1) * math.sqrt(sum_squares2))
        lambda_param = 100
        
        return (n/(n + lambda_param))*pear
        
        
    @staticmethod
    def sim_pearson(ratings1: dict, ratings2: dict) -> float:
        # {item_id/user_id, rating}
        
        # get co-rated
        co_rated = []
        
        for id in ratings1: 
            
            if id in ratings2: 
                co_rated.append(id)
                
        n = len(co_rated)
        
        # if no co-rated, return 0
        if n == 0: 
            return 0.0
        
        # means over co-rated
        mean1 = sum([ratings1[id] for id in co_rated]) * 1.0 / n
        mean2 = sum([ratings2[id] for id in co_rated]) * 1.0 / n
        
        # sums of products of differences and sums of squared differences
        sum_prods = 0.0
        sum_squares1 = 0.0
        sum_squares2 = 0.0
        
        for id in co_rated:
            difference1 = ratings1[id] - mean1
            difference2 = ratings2[id] - mean2
            sum_prods = sum_prods + difference1 * difference2
            sum_squares1 = sum_squares1 + difference1 * difference1
            sum_squares2 = sum_squares2 + difference2 * difference2
            
        # handle zero variances
        if sum_squares1 == 0 or sum_squares2 == 0:
            return 0.0
        
        return sum_prods / (math.sqrt(sum_squares1) * math.sqrt(sum_squares2))


    @staticmethod
    def sim_cosine(ratings1: dict, ratings2: dict) -> float:
        # {item_id/user_id, rating}

        if len(ratings1) == 0 or len(ratings2) == 0:
            return 0.0
        
        return Similarities.dot_product(ratings1, ratings2) / (Similarities.magnitude(ratings1) * Similarities.magnitude(ratings2))


    @staticmethod
    def dot_product(ratings1: dict, ratings2: dict) -> float:
        # {item_id/user_id, rating}

        dot_product = 0.0
        
        for id in ratings1:
            
            if id in ratings2:
                dot_product = dot_product + ratings1[id] * ratings2[id]
                
        return dot_product
        
        
    @staticmethod  
    def magnitude(ratings: dict) -> float:
        # {item_id/user_id, rating}

        sum_squares = 0.0
        
        for rating in ratings.values():
            sum_squares = sum_squares + rating * rating
            
        return math.sqrt(sum_squares)
        