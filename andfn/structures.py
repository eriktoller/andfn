"""
Notes
-----
This module contains the underground structures classes.
"""

import numpy as np
import pyvista as pv
import andfn.geometry_functions as gf
from andfn.const_head import ConstantHeadLine
from . import BoundingCircle, Intersection
from .impermeable_object import ImpermeableLine

STRUCTURES_COLOR = {0: "FF0000", 1: "0000FF"}


class Structure:
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

    def __str__(self):
        """
        Returns the string representation of the element.

        Returns
        -------
        str
            The string representation of the element.
        """
        return f"{self.__class__.__name__}: {self.label}"


class RegularPolygonPrism(Structure):
    """
    Base class for polygonal underground structures.
    """

    def __init__(self, label, radius, start, end, n_sides, _structure_type, **kwargs):
        """
        Initializes the polygonal underground structures class.

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
        self.start = start.astype(np.float64)
        self.end = end.astype(np.float64)
        if n_sides < 3:
            raise ValueError("n_sides must be at least 3 for a tunnel.")
        self.n_sides = n_sides
        self._structure_type = _structure_type
        self.head = 0

        # Calculate the vertices of the tunnel
        self.vertices = None
        self.faces = None
        self.get_vertices_and_faces()

        # Make lists for the ConstantHeadLine elements and the fractures
        self.fracs = []
        self.elements = []

    def get_lvc(self):
        """
        Returns the length, directional vector, and center of the tunnel.

        Returns
        -------
        length : float
            The length of the tunnel.
        direction : np.ndarray
            The directional vector of the tunnel, normalized to unit length.
        center : np.ndarray
            The center point of the tunnel, calculated as the midpoint between start and end points.
        """
        length = np.linalg.norm(self.end - self.start)
        direction = (self.end - self.start) / length
        center = (self.start + self.end) / 2
        return length, direction, center

    def get_vertices_and_faces(self):
        """
        Calculates the vertices of the tunnel.

        Returns
        -------
        vertices : np.ndarray
            The vertices of the tunnel.
        """
        length, direction, center = self.get_lvc()
        angle = np.linspace(0, 2 * np.pi, self.n_sides, endpoint=False)
        # rotate it pi/4
        angle += np.pi / 4
        x0 = self.radius * np.cos(angle)
        x1 = self.radius * np.sin(angle)

        # Create the vertices in the local coordinate system
        z = x0 + 1j * x1  # Complex representation of the circle in the xy-plane
        x2_vec = np.array([0, 0, 1])  # z-axis vector
        x0_vec = np.cross(direction, x2_vec)  # Perpendicular vector in the xy-plane
        if np.linalg.norm(x0_vec) < 1e-6:  # If the direction is aligned with z-axis
            x0_vec = np.array([1, 0, 0])
        x0_vec /= np.linalg.norm(x0_vec)  # Normalize the vector

        # Map the xy coordinates to the 3D space
        vertices_start = (
            np.real(z)[:, np.newaxis] * x2_vec
            + np.imag(z)[:, np.newaxis] * x0_vec
            + self.start
        )
        vertices_end = (
            np.real(z)[:, np.newaxis] * x2_vec
            + np.imag(z)[:, np.newaxis] * x0_vec
            + self.end
        )
        # Combine the start and end vertices
        self.vertices = np.vstack((vertices_start, vertices_end))

        # Get the faces
        faces = [
            [self.n_sides] + list(range(self.n_sides)),  # First face (start cap)
            [self.n_sides] + list(range(self.n_sides, 2 * self.n_sides)),
        ]  # Second face (end cap)

        # Side faces (quads)
        for i in range(self.n_sides):
            next_i = (i + 1) % self.n_sides
            faces.append([4, i, next_i, self.n_sides + next_i, self.n_sides + i])

        self.faces = np.hstack(faces)

    def plot(self, pl, opacity=0.5):
        """
        Plots the tunnel on the given axes.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object to use for plotting.
        """
        pl.add_points(
            np.array([self.start, self.end]),
            color=STRUCTURES_COLOR[self._structure_type],
            point_size=10,
            render_points_as_spheres=True,
            edge_color=STRUCTURES_COLOR[self._structure_type],
        )
        # Create a polygon for the first 4 vertices
        poly = pv.PolyData(self.vertices, self.faces)
        # Add the polygon to the plotter
        pl.add_mesh(
            poly,
            show_edges=True,
            show_vertices=True,
            vertex_color=STRUCTURES_COLOR[self._structure_type],
            color=STRUCTURES_COLOR[self._structure_type],
            edge_opacity=1.0,
            opacity=opacity,
            render_points_as_spheres=True,
            point_size=5,
        )

    def possible_intersections(self, frac, pl):
        """
        Checks if the tunnel can possibly intersect with a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to check for possible intersections.

        Returns
        -------
        bool
            True if the tunnel can possibly intersect with the fracture, False otherwise.
        """
        # Check if the fracture center + radius is between the start and end points of the tunnel
        frac_radius = frac.radius
        start = gf.map_3d_to_2d(self.start, frac)
        end = gf.map_3d_to_2d(self.end, frac)
        # make the line equation of the tunnel in the type of ax + by + c = 0
        a = end.imag - start.imag
        b = start.real - end.real
        c = end.real * start.imag - start.real * end.imag
        dist = np.abs(c) / np.sqrt(a**2 + b**2)
        if dist > frac_radius * (1 + 1e-10):
            # The tunnel is too far away from the fracture to intersect
            return False
        return True

    def frac_intersections(self, fractures, pl=None):
        """
        Checks if the tunnel intersects with a given fracture.

        Parameters
        ----------
        fractures : andfn.fracture.Fracture
            The fracture to check for intersections with the tunnel.
        pl : pyvista.Plotter
            The plotter object to use for plotting the intersection points.

        Returns
        -------
        bool
            True if the tunnel elements are added to the fracture, False otherwise.
        """
        if not isinstance(fractures, list):
            fractures = [fractures]
        for frac in fractures:
            # Check if the tunnel can possibly intersect with the fracture
            if not self.possible_intersections(frac, pl):
                return False
            # calculate the intersection points between line between the verticies and the fracture plane
            pnts = []
            for i in range(self.n_sides - 1):
                pnt = self.line_plane_intersection(
                    self.vertices[i],
                    self.vertices[i + self.n_sides],
                    frac.center,
                    frac.normal,
                )
                if pnt is not None:
                    pnts.append(pnt)
                pnt = self.line_plane_intersection(
                    self.vertices[i], self.vertices[i + 1], frac.center, frac.normal
                )
                if pnt is not None:
                    pnts.append(pnt)
                pnt = self.line_plane_intersection(
                    self.vertices[self.n_sides + i],
                    self.vertices[self.n_sides + i + 1],
                    frac.center,
                    frac.normal,
                )
                if pnt is not None:
                    pnts.append(pnt)
            pnt = self.line_plane_intersection(
                self.vertices[self.n_sides - 1],
                self.vertices[self.n_sides + self.n_sides - 1],
                frac.center,
                frac.normal,
            )
            if pnt is not None:
                pnts.append(pnt)
            pnt = self.line_plane_intersection(
                self.vertices[0],
                self.vertices[self.n_sides - 1],
                frac.center,
                frac.normal,
            )
            if pnt is not None:
                pnts.append(pnt)
            pnt = self.line_plane_intersection(
                self.vertices[self.n_sides],
                self.vertices[self.n_sides * 2 - 1],
                frac.center,
                frac.normal,
            )
            if pnt is not None:
                pnts.append(pnt)

            if len(pnts) == 0:
                # No intersection points found
                continue

            int_pnts = []
            for i in range(len(pnts) - 1):
                # map to plane and check if there is an intersection point between the points and the boundary of the fracture
                z1 = gf.map_3d_to_2d(pnts[i], frac)
                z2 = gf.map_3d_to_2d(pnts[i + 1], frac)
                z3, z4 = gf.line_circle_intersection(z1, z2, frac.radius)
                if z3 is not None:
                    # Check is z3 or z4 is between z1 and z2
                    if np.all(
                        np.abs(np.abs(z3 - z1) + np.abs(z3 - z2) - np.abs(z2 - z1))
                        < 1e-10
                    ):
                        # map the intersection point back to 3d
                        pnt3 = gf.map_2d_to_3d(z3, frac)
                        int_pnts.append(pnt3)
                    if np.all(
                        np.abs(np.abs(z4 - z1) + np.abs(z4 - z2) - np.abs(z2 - z1))
                        < 1e-10
                    ):
                        # map the intersection point back to 3d
                        pnt4 = gf.map_2d_to_3d(z4, frac)
                        int_pnts.append(pnt4)
            # Check if the first and last point of the tunnel intersects with the fracture
            z1 = gf.map_3d_to_2d(pnts[0], frac)
            z2 = gf.map_3d_to_2d(pnts[-1], frac)
            z3, z4 = gf.line_circle_intersection(z1, z2, frac.radius)
            if z3 is not None:
                # Check is z3 or z4 is between z1 and z2
                if np.all(
                    np.abs(np.abs(z3 - z1) + np.abs(z3 - z2) - np.abs(z2 - z1)) < 1e-10
                ):
                    # map the intersection point back to 3d
                    pnt3 = gf.map_2d_to_3d(z3, frac)
                    int_pnts.append(pnt3)
                if np.all(
                    np.abs(np.abs(z4 - z1) + np.abs(z4 - z2) - np.abs(z2 - z1)) < 1e-10
                ):
                    # map the intersection point back to 3d
                    pnt4 = gf.map_2d_to_3d(z4, frac)
                    int_pnts.append(pnt4)

            if len(int_pnts) > 0:
                pnts.insert(0, int_pnts[0])
                pnts.append(int_pnts[1])
            pnts_inside = []
            pnts_outside = []
            for i, pnt in enumerate(pnts):
                if self.inside_fracture(pnt, frac):
                    pnts_inside.append(pnt)
                else:
                    pnts_outside.append(pnt)

            # plot the vertices in 3d
            if pl:
                pl.add_points(
                    np.array(pnts),
                    color="red",
                    point_size=4,
                    render_points_as_spheres=True,
                )
                if len(pnts_inside) > 0:
                    pl.add_points(
                        np.array(pnts_inside),
                        color="green",
                        point_size=4,
                        render_points_as_spheres=True,
                    )

            # Create constant head elements for the tunnel in this fracture
            self.assign_elements(frac, pnts_inside, pnts)
            # Check if there are any other elements in the fracture that are inside the tunnel
            self.check_internal_elements(frac)

    def assign_elements(self, frac, pnts_inside, pnts):
        """
        Assigns constant head elements for the tunnel in a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to create constant head elements in.
        pnts_inside : list of np.ndarray
            The points inside the fracture where the tunnel intersects.
        pnts : list of np.ndarray
            The points of the tunnel that intersect with the fracture.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def inside_fracture(pnt, frac):
        """
        Checks if a point is inside a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to check if the tunnel is inside.

        Returns
        -------
        bool
            True if the tunnel is inside the fracture, False otherwise.
        """
        z = gf.map_3d_to_2d(pnt, frac)
        if np.abs(z) > frac.radius * (1 + 1e-10):
            return False
        return True

    @staticmethod
    def line_plane_intersection(line_start, line_end, plane_point, plane_normal):
        """
        Calculates the intersection point between a line and a plane.

        Parameters
        ----------
        line_start : np.ndarray
            The start point of the line.
        line_end : np.ndarray
            The end point of the line.
        plane_point : np.ndarray
            A point on the plane.
        plane_normal : np.ndarray
            The normal vector of the plane.

        Returns
        -------
        np.ndarray or None
            The intersection point if it exists, otherwise None.
        """
        line_direction = line_end - line_start
        d = np.dot(plane_normal, (plane_point - line_start)) / np.dot(
            plane_normal, line_direction
        )

        if 0 <= d <= 1:
            return line_start + d * line_direction
        return None

    def check_internal_elements(self, frac, atol=1e-1):
        """
        Checks if the tunnel intersects with a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to check for intersections with the tunnel.

        Returns
        -------
        bool
            True if the tunnel intersects with the fracture, False otherwise.
        """
        # Check if the elements are crossing any intersection elements
        elem_prism = self.elements
        elem_frac = frac.elements

        line_line_intersection = gf.line_line_intersection

        for elem1 in elem_prism:
            if elem1.frac0 != frac:
                continue
            for elem2 in elem_frac:
                if isinstance(elem2, BoundingCircle):
                    continue
                endpoints2 = elem2.endpoints0
                if isinstance(elem2, Intersection):
                    if frac == elem2.frac1:
                        endpoints2 = elem2.endpoints1
                # Check if the two elements are crossing each other
                z = line_line_intersection(
                    elem1.endpoints0[0],
                    elem1.endpoints0[1],
                    endpoints2[0],
                    endpoints2[1],
                )
                if z:
                    # If there is an intersection point, the tunnel intersects with the fracture
                    chi1 = gf.map_z_line_to_chi(z, elem1.endpoints0)
                    chi2 = gf.map_z_line_to_chi(z, endpoints2)
                    if np.abs(np.imag(chi1)) > atol and np.abs(np.imag(chi2)) > atol:
                        chi = gf.map_z_line_to_chi(endpoints2[0], elem1.endpoints0)
                        if frac == elem2.frac0:
                            if np.imag(chi) < 0:
                                elem2.endpoints0[0] = z
                            else:
                                elem2.endpoints0[1] = z
                        else:
                            if np.imag(chi) < 0:
                                elem2.endpoints1[0] = z
                            else:
                                elem2.endpoints1[1] = z


class ConstantHeadPrism(RegularPolygonPrism):
    """
    Class for constant head regular polygonal prism tunnels.
    """

    def __init__(self, label, radius, start, end, head=0, n_sides=4, **kwargs):
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
        super().__init__(
            label, radius, start, end, n_sides, _structure_type=0, **kwargs
        )
        self.head = head

    def assign_elements(self, frac, pnts_inside, pnts):
        """
        Creates constant head elements for the tunnel in a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to create constant head elements in.
        pnts_inside : list of np.ndarray
            The points inside the fracture where the tunnel intersects.
        pnts : list of np.ndarray
            The points of the tunnel that intersect with the fracture.
        """
        if len(pnts_inside) < 2:
            return
        for j in range(len(pnts_inside) - 1):
            # Create a constant head line for each segment of the tunnel inside the fracture
            z0 = gf.map_3d_to_2d(pnts_inside[j], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[j + 1], frac)
            ch = ConstantHeadLine(
                f"tunnel_{self.label}_frac_{frac.label}_{j}",
                np.array([z0, z1]),
                self.head,
                frac,
            )
            self.elements.append(ch)
        if len(pnts_inside) == len(pnts):
            z0 = gf.map_3d_to_2d(pnts_inside[0], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[-1], frac)
            ch = ConstantHeadLine(
                f"tunnel_{self.label}_frac_{frac.label}_{len(pnts_inside)}",
                np.array([z0, z1]),
                self.head,
                frac,
            )
            self.elements.append(ch)
        self.fracs.append(frac)


class ImpermeablePrism(RegularPolygonPrism):
    """
    Class for impermeable regular polygonal prims.
    """

    def __init__(self, label, radius, start, end, n_sides=4, **kwargs):
        """
        Initializes the impermeable cylinder class.

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
        super().__init__(
            label, radius, start, end, n_sides, _structure_type=1, **kwargs
        )

    def assign_elements(self, frac, pnts_inside, pnts):
        """
        Creates constant head elements for the tunnel in a given fracture.

        Parameters
        ----------
        frac : andfn.fracture.Fracture
            The fracture to create constant head elements in.
        pnts_inside : list of np.ndarray
            The points inside the fracture where the tunnel intersects.
        pnts : list of np.ndarray
            The points of the tunnel that intersect with the fracture.
        """
        if len(pnts_inside) < 2:
            return
        for j in range(len(pnts_inside) - 1):
            # Create a constant head line for each segment of the tunnel inside the fracture
            z0 = gf.map_3d_to_2d(pnts_inside[j], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[j + 1], frac)
            ch = ImpermeableLine(
                f"tunnel_{self.label}_frac_{frac.label}_{j}", np.array([z0, z1]), frac
            )
            self.elements.append(ch)
        if len(pnts_inside) == len(pnts):
            z0 = gf.map_3d_to_2d(pnts_inside[0], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[-1], frac)
            ch = ImpermeableLine(
                f"tunnel_{self.label}_frac_{frac.label}_{len(pnts_inside)}",
                np.array([z0, z1]),
                frac,
            )
            self.elements.append(ch)
        self.fracs.append(frac)
