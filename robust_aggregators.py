import aggregators
import preaggregators

class RobustAggregator(object):

    """
    Description
    -----------
    Class to apply all the pre-aggregations and the aggregation
    to the vectors.

    Parameters
    ----------
    aggregator-info : dict 
        Dictionary with the keys "name" and "parameters" 
        of the aggregation defined.
    pre-agg-info : list
        List of dictionaries (one for every pre_agg function)
        where every dictionary have the keys "name" and "parameters" defined.

    Methods
    -------
    """

    def __init__(self, aggregator_info, pre_agg_list=[]):
        self.aggregator = getattr(
            aggregators, 
            aggregator_info["name"]
        )(**aggregator_info["parameters"])
        

        self.pre_agg_list = []

        for pre_agg_info in pre_agg_list:
            pre_agg = getattr(
                preaggregators, 
                pre_agg_info["name"]
            )(**pre_agg_info["parameters"])

            self.pre_agg_list.append(pre_agg)

    def aggregate(self, vectors):
        """
        Description
        -----------
        Apply pre-aggregations and aggregations to the vectors

        Parameters
        ----------
        vectors : (list or np.ndarray or torch.Tensor)
            A list of vectors or a matrix (2D array/tensor) where each
            row represents a vector.

        Returns
        -------
        ndarray or torch.Tensor
            Returns a vector with the Meamed applied to the input vectors
        """
        for pre_agg in self.pre_agg_list:
            vectors = pre_agg.pre_aggregate_vectors(vectors)

        aggregate_vector = self.aggregator.aggregate_vectors(vectors)  

        return aggregate_vector