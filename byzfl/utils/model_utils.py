"""Model lookup and explicit SNN identification shared by the framework."""


def get_model_class(model_name):
    """Return a model class from the existing model module."""
    if not isinstance(model_name, str):
        raise TypeError("Model name must be a string; expand configuration sweeps first.")
    # Defer the model-module import until a model is requested.
    import byzfl.fed_framework.models as models
    return getattr(models, model_name)


def is_snn_model(model_class):
    """Read the class's boolean is_snn declaration, defaulting to False."""
    is_snn = getattr(model_class, "is_snn", False)
    if not isinstance(is_snn, bool):
        raise TypeError("The model class attribute 'is_snn' must be a bool.")
    return is_snn
