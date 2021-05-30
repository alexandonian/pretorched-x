import numpy as np


class Label2OneHot(object):
    """Convert integer labels to one-hot vectors for arbitrary dimensional data."""

    def __init__(self, num_classes):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        dtype : str
            Datatype of the output.
        super_kwargs : dict
            Keyword arguments to the superclass.
        """
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, x):
        reshaped_arange = np.arange(self.num_classes).reshape(-1, *(1,) * x.ndim)
        output = np.equal(reshaped_arange, x)
        return output
