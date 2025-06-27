"""
Notes
-----
This module contains the underground structures classes.
"""

import numpy as np
import pyvista as pv
import andfn.geometry_functions as gf

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
    d = np.dot(plane_normal, (plane_point - line_start)) / np.dot(plane_normal, line_direction)

    if 0 <= d <= 1:
        return line_start + d * line_direction
    return None


class Tunnel(UndergroundStructure):
    """
    Class for tunnels.
    """

    def __init__(self, label, radius, start, end, n_sides=4, **kwargs):
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
        self.start = start.astype(np.float64)
        self.end = end.astype(np.float64)
        if n_sides < 3:
            raise ValueError("n_sides must be at least 3 for a tunnel.")
        self.n_sides = n_sides

        # Calculate the vertices of the tunnel
        self.vertices = None
        self.faces = None
        self.get_vertices_and_faces()

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
        vertices_start = np.real(z)[:, np.newaxis] * x2_vec + np.imag(z)[:, np.newaxis] * x0_vec + self.start
        vertices_end = np.real(z)[:, np.newaxis] * x2_vec + np.imag(z)[:, np.newaxis] * x0_vec + self.end
        # Combine the start and end vertices
        self.vertices = np.vstack((vertices_start, vertices_end))

        # Get the faces
        faces = [[self.n_sides] + list(range(self.n_sides)),                    # First face (start cap)
                 [self.n_sides] + list(range(self.n_sides, 2 * self.n_sides))]  # Second face (end cap)

        # Side faces (quads)
        for i in range(self.n_sides):
            next_i = (i + 1) % self.n_sides
            faces.append([
                4,
                i,
                next_i,
                self.n_sides + next_i,
                self.n_sides + i
            ])

        self.faces = np.hstack(faces)




    def plot(self, pl):
        """
        Plots the tunnel on the given axes.

        Parameters
        ----------
        pl : pyvista.Plotter
            The plotter object to use for plotting.
        """
        pl.add_points(
            self.vertices,
            scalars=self.vertices[:, 2], cmap="viridis",
            point_size=4,
            render_points_as_spheres=True
        )
        pl.add_points(
            np.array([self.start, self.end]),
            color="orange",
            point_size=4,
            render_points_as_spheres=True
        )
        # Create a polygon for the first 4 vertices
        poly = pv.PolyData(self.vertices, self.faces)
        # Add the polygon to the plotter
        pl.add_mesh(
            poly,
            color="blue",
            show_edges=True,
            edge_color="black",
            opacity=0.5
        )

    def frac_intersections(self, frac, pl):
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
        # calculate the intersection points between line between the verticies and the fracture plane
        pnts = []
        for i in range(self.n_sides-1):
            pnt = line_plane_intersection(
                self.vertices[i],
                self.vertices[i + self.n_sides],
                frac.center,
                frac.normal
            )
            if pnt is not None:
                pnts.append(pnt)
            pnt = line_plane_intersection(
                self.vertices[i],
                self.vertices[i + 1],
                frac.center,
                frac.normal
            )
            if pnt is not None:
                pnts.append(pnt)
            pnt = line_plane_intersection(
                self.vertices[self.n_sides + i],
                self.vertices[self.n_sides + i + 1],
                frac.center,
                frac.normal
            )
            if pnt is not None:
                pnts.append(pnt)
        pnt = line_plane_intersection(
            self.vertices[self.n_sides-1],
            self.vertices[self.n_sides + self.n_sides-1],
            frac.center,
            frac.normal
        )
        if pnt is not None:
            pnts.append(pnt)
        pnt = line_plane_intersection(
            self.vertices[0],
            self.vertices[self.n_sides-1],
            frac.center,
            frac.normal
        )
        if pnt is not None:
            pnts.append(pnt)
        pnt = line_plane_intersection(
            self.vertices[self.n_sides],
            self.vertices[self.n_sides*2-1],
            frac.center,
            frac.normal
        )
        if pnt is not None:
            pnts.append(pnt)

        int_pnts = []
        for i in range(len(pnts)-1):
            # map to plane and check if there is an intersection point between the points and the boundary of the fracture
            z1 = gf.map_3d_to_2d(pnts[i], frac)
            z2 = gf.map_3d_to_2d(pnts[i + 1], frac)
            z3, z4 = gf.line_circle_intersection(z1, z2, frac.radius)
            if z3 is not None:
                # Check is z3 or z4 is between z1 and z2
                if np.all(np.abs(np.abs(z3 - z1) + np.abs(z3 - z2) - np.abs(z2 - z1)) < 1e-10):
                    # map the intersection point back to 3d
                    pnt3 = gf.map_2d_to_3d(z3, frac)
                    int_pnts.append(pnt3)
                if np.all(np.abs(np.abs(z4 - z1) + np.abs(z4 - z2) - np.abs(z2 - z1)) < 1e-10):
                    # map the intersection point back to 3d
                    pnt4 = gf.map_2d_to_3d(z4, frac)
                    int_pnts.append(pnt4)
        # Check if the first and last point of the tunnel intersects with the fracture
        z1 = gf.map_3d_to_2d(pnts[0], frac)
        z2 = gf.map_3d_to_2d(pnts[-1], frac)
        z3, z4 = gf.line_circle_intersection(z1, z2, frac.radius)
        if z3 is not None:
            # Check is z3 or z4 is between z1 and z2
            if np.all(np.abs(np.abs(z3 - z1) + np.abs(z3 - z2) - np.abs(z2 - z1)) < 1e-10):
                # map the intersection point back to 3d
                pnt3 = gf.map_2d_to_3d(z3, frac)
                int_pnts.append(pnt3)
            if np.all(np.abs(np.abs(z4 - z1) + np.abs(z4 - z2) - np.abs(z2 - z1)) < 1e-10):
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
        pl.add_points(
            np.array(pnts),
            color="red",
            point_size=4,
            render_points_as_spheres=True
        )
        if len(pnts_inside) > 0:
            pl.add_points(
                np.array(pnts_inside),
                color="green",
                point_size=8,
                render_points_as_spheres=True
            )
        if len(pnts_inside) == len(pnts):
            pl.add_mesh(
                pv.MultipleLines(np.array(pnts + [pnts[0]])),
                color="red",
                opacity=1.0,
                line_width=5,
            )
        elif len(pnts_inside) > 1:
            pl.add_mesh(
                pv.MultipleLines(np.array(pnts_inside)),
                color="red",
                opacity=1.0,
                line_width=5,
            )

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
        if np.abs(z) > frac.radius*(1+1e-10):
            return False
        return True