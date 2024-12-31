import inspect
from byzfl.aggregators import aggregators
from byzfl.aggregators import preaggregators
from byzfl.utils import misc

class RobustAggregator:

    """
    Description
    -----------
    Class to apply all the pre-aggregations and the aggregation to the vectors.

    Initialization parameters
    -------------------------
    aggregator-info : dict 
        Dictionary with the keys "name" and "parameters" of the aggregation defined.
    pre-agg-info : list
        List of dictionaries (one for every pre_agg function) where every dictionary has the keys "name" and "parameters" defined.
    
    Calling the instance
    --------------------

    Input parameters
    ----------------

    vectors: numpy.ndarray, torch.Tensor, list of numpy.ndarray or list of torch.Tensor
        A set of vectors, matrix or tensors.

    Returns
    -------
    :numpy.ndarray or torch.Tensor
        The data type of the output will be the same as the input.

    """

    def __init__(self, aggregator_info, pre_agg_list=[]):

        self.aggregator = getattr(aggregators, aggregator_info["name"])
        signature_agg = inspect.signature(self.aggregator.__init__)
        agg_parameters = {}
        for parameter in signature_agg.parameters.values():
            param_name = parameter.name
            if param_name in aggregator_info["parameters"]:
                agg_parameters[param_name] = aggregator_info["parameters"][param_name]
        self.aggregator = self.aggregator(**agg_parameters)

        self.pre_agg_list = []
        for pre_agg_info in pre_agg_list:
            pre_agg = getattr(preaggregators, pre_agg_info["name"])
            signature_pre_agg = inspect.signature(pre_agg.__init__)
            pre_agg_parameters = {}
            for parameter in signature_pre_agg.parameters.values():
                param_name = parameter.name
                if param_name in pre_agg_info["parameters"]:
                    pre_agg_parameters[param_name] = pre_agg_info["parameters"][param_name]
            pre_agg = pre_agg(**pre_agg_parameters)
            self.pre_agg_list.append(pre_agg)

    def __call__(self, vectors):
        """
        Description
        -----------
        Apply pre-aggregations and aggregations to the vectors
        """

        for pre_agg in self.pre_agg_list:
            vectors = pre_agg(vectors)
        return self.aggregator(vectors)