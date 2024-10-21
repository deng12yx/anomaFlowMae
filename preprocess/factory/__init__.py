class FactoryMeta(type):
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        else:
            cls.registry[name.lower()] = cls


from preprocess.factory.aggregator import BaseAggregator, AggregatorFactory, aggregator_factory
from preprocess.factory.transform_pcap import BaseAdaptor, AdaptorFactory, transform_pcap_factory

__all__ = [
    "FactoryMeta",
    "BaseAggregator",
    "AggregatorFactory",
    "aggregator_factory",
    "BaseAdaptor",
    "AdaptorFactory",
    "transform_pcap_factory",
]
