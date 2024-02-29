import torch
import agg

class RobustAggregator(object):

    def __init__(self, 
                 aggregator, 
                 nb_byz = 0, 
                 pre_aggregator = None,
                 bucket_size = None):
        self.aggregator = aggregator
        self.pre_aggregator = pre_aggregator
        self.nb_byz = nb_byz
        self.bucket_size = bucket_size
        self.prev_momentum = 0
        self.robust_aggregators = {"average": self.average,
                                   "trmean": self.trmean, 
                                   "median": self.median,
                                   "geometric_median":self.geometric_median, 
                                   "krum": self.krum, 
                                   "multi_krum": self.multi_krum,
                                   "nnm": self.nnm, 
                                   "bucketing": self.bucketing, 
                                   "cc": self.cc, 
                                   "mda": self.mda,
                                   "mva": self.mva, 
                                   "monna": self.monna, 
                                   "meamed": self.meamed, 
                                   "identity": self.identity
                                   }

    def aggregate(self, vectors):
        if self.pre_aggregator is not None:
            mixed_vectors = self.robust_aggregators[self.pre_aggregator](vectors)
            aggregate_vector = self.robust_aggregators[self.aggregator](mixed_vectors)
        else:
            aggregate_vector = self.robust_aggregators[self.aggregator](vectors)
        #JS: Update the value of the previous momentum (e.g., for Centered Clipping aggregator)
        self.prev_momentum = aggregate_vector
        return aggregate_vector

    def average(self, vectors):
        return agg.average(vectors)

    def trmean(self, vectors):
        return agg.trmean(vectors, self.nb_byz)

    def median(self, vectors):
        return agg.median(vectors)

    def geometric_median(self, vectors):
        return agg.geometric_median(vectors)

    def krum(self, vectors):
        return agg.krum(vectors, self.nb_byz)

    def multi_krum(self, vectors):
        return agg.multi_krum(vectors, self.nb_byz)
    
    def meamed(self, vectors):
        return agg.meamed(vectors, self.nb_byz)
    
    def mda(self, vectors):
        return agg.mda(vectors, self.nb_byz)

    def mva(self, vectors):
        return agg.mva(vectors, self.nb_byz)

    def cc(self, vectors):
        return agg.centered_clipping(vectors, self.prev_momentum)

    def monna(self, vectors):
        return agg.monna(vectors, self.nb_byz)

    def nnm(self, vectors):
        return agg.nnm(vectors, self.nb_byz)

    def bucketing(self, vectors):
        return agg.bucketing(vectors, self.bucket_size)

    def identity(self, vectors):
        return agg.identity(vectors)