"""
Notes
-----
This module contains the underground structures classes.
"""


class UndergroundStructure:
    """
    Base class for underground structures.
    """
    def __init__(self, label, **kwargs):
        """
        Initializes the underground structure class.

        Parameters
        ----------
        label : str or int
            The label of the underground structure.
        kwargs : dict
            Additional keyword arguments.
        """
        self.label = label
        self.fracs = None

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class Tunnel(UndergroundStructure):
    """
    Class for tunnels.
    """
    def __init__(self, label, radius, start, end, n_sides=-1, **kwargs):
        """
        Initializes the tunnel class.

        Parameters
        ----------
        label : str or int
            The label of the tunnel.
        radius : float
            The radius of the tunnel.
        start : np.ndarray
            The start point of the tunnel.
        end : np.ndarray
            The end point of the tunnel.
        n_sides : int, optional
            The number of sides of the tunnel. Default is -1 (circular tunnel).
        """
        super().__init__(label, **kwargs)
        self.radius = radius
        self.start = start
        self.end = end
        self.n_sides = n_sides
