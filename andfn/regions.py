"""
Notes
-----
This module contains the region classes.
"""

import numpy as np
import pyvista as pv
import andfn.geometry_functions as gf

from andfn.const_head import ConstantHeadLine

REGION_COLOR = {0: "00FF00", 1: "0000FF"}


class Region:
    """
    Base class for regions.
    """

    def __init__(self, label, _region_type, **kwargs):
        """
        Initializes the region class.

        Parameters
        ----------
        label : str or int
            The label of the underground structure.
        kwargs : dict
            Additional keyword arguments.
        """
        self.label = label
        self.fracs = None
        self._region_type = _region_type

        # Set the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        """
        Returns the string representation of the region.

        Returns
        -------
        str
            The string representation of the element.
        """
        return f"{self.__class__.__name__}: {self.label}"


class RectangularRegion(Region):
    """
    Class for rectangular regions.
    """

    def __init__(self, label, center, x_vec, y_vec, z_vec, xl, yl, zl, **kwargs):
        """
        Initializes the rectangular region class.

        Parameters
        ----------
        label : str or int
            The label of the underground structure.
        x_vec : list or np.ndarray
            The x coordinate vector of the rectangular region.
        y_vec : list or np.ndarray
            The y coordinate vector of the rectangular region.
        z_vec : list or np.ndarray
            The z coordinate vector of the rectangular region.
        xl : float
            The length of the rectangular region in the x direction.
        yl : float
            The length of the rectangular region in the y direction.
        zl : float
            The length of the rectangular region in the z direction.
        kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(label, _region_type=0, **kwargs)
        self.center = np.array(center)
        self.x_vec = np.array(x_vec)
        self.x_vec = self.x_vec / np.linalg.norm(self.x_vec)
        self.y_vec = np.array(y_vec)
        self.y_vec = self.y_vec / np.linalg.norm(self.y_vec)
        self.z_vec = np.array(z_vec)
        self.z_vec = self.z_vec / np.linalg.norm(self.z_vec)
        self.xl = xl
        self.yl = yl
        self.zl = zl

        # Check that the vectors are orthogonal
        if not np.isclose(np.dot(self.x_vec, self.y_vec), 0):
            raise ValueError("The x and y vectors are not orthogonal.")
        if not np.isclose(np.dot(self.x_vec, self.z_vec), 0):
            raise ValueError("The x and z vectors are not orthogonal.")
        if not np.isclose(np.dot(self.y_vec, self.z_vec), 0):
            raise ValueError("The y and z vectors are not orthogonal.")

        self.vertices = None
        self.faces = None
        self.faces_dict = None
        self.get_vertices_faces()

        # Make lists for the ConstantHeadLine elements and the fractures
        self.fracs = []
        self.elements = []

    def get_vertices_faces(self):
        """
        Gets the vertices and faces of the rectangular region.

        Returns
        -------
        vertices : np.ndarray
            The vertices of the rectangular region.
        faces : np.ndarray
            The faces of the rectangular region.
        """
        if self.vertices is not None and self.faces is not None:
            return self.vertices, self.faces

        # Create the corners of the rectangular region
        self.vertices = np.array(
            [
                [
                    self.center
                    + 0.5 * self.xl * self.x_vec
                    + 0.5 * self.yl * self.y_vec
                    + 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    - 0.5 * self.xl * self.x_vec
                    + 0.5 * self.yl * self.y_vec
                    + 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    - 0.5 * self.xl * self.x_vec
                    - 0.5 * self.yl * self.y_vec
                    + 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    + 0.5 * self.xl * self.x_vec
                    - 0.5 * self.yl * self.y_vec
                    + 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    + 0.5 * self.xl * self.x_vec
                    + 0.5 * self.yl * self.y_vec
                    - 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    - 0.5 * self.xl * self.x_vec
                    + 0.5 * self.yl * self.y_vec
                    - 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    - 0.5 * self.xl * self.x_vec
                    - 0.5 * self.yl * self.y_vec
                    - 0.5 * self.zl * self.z_vec
                ],
                [
                    self.center
                    + 0.5 * self.xl * self.x_vec
                    - 0.5 * self.yl * self.y_vec
                    - 0.5 * self.zl * self.z_vec
                ],
            ]
        ).reshape(-1, 3)
        # The vertices are ordered as follows:
        # 0: top front right
        # 1: top front left
        # 2: top back left
        # 3: top back right
        # 4: bottom front right
        # 5: bottom front left
        # 6: bottom back left
        # 7: bottom back right

        # Create the faces of the rectangular region
        self.faces = np.hstack(
            [
                [4, 0, 1, 2, 3],
                [4, 4, 5, 6, 7],
                [4, 0, 1, 5, 4],
                [4, 1, 2, 6, 5],
                [4, 2, 3, 7, 6],
                [4, 3, 0, 4, 7],
            ]
        )
        self.faces_dict = {
            "top": [0, 1, 2, 3],
            "bottom": [4, 5, 6, 7],
            "front": [0, 1, 5, 4],
            "back": [2, 3, 7, 6],
            "left": [1, 2, 6, 5],
            "right": [0, 3, 7, 4],
        }

    def rotate(self, angle, axis):
        """
        Rotates the rectangular region around a given axis by a given angle.

        Parameters
        ----------
        angle : float
            The angle to rotate by in degrees (in degrees).
        axis : list or np.ndarray
            The axis to rotate around.
        """
        # Create the rotation matrix
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        angle_rad = np.radians(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        R = np.array(
            [
                [
                    cos_angle + axis[0] ** 2 * (1 - cos_angle),
                    axis[0] * axis[1] * (1 - cos_angle) - axis[2] * sin_angle,
                    axis[0] * axis[2] * (1 - cos_angle) + axis[1] * sin_angle,
                ],
                [
                    axis[1] * axis[0] * (1 - cos_angle) + axis[2] * sin_angle,
                    cos_angle + axis[1] ** 2 * (1 - cos_angle),
                    axis[1] * axis[2] * (1 - cos_angle) - axis[0] * sin_angle,
                ],
                [
                    axis[2] * axis[0] * (1 - cos_angle) - axis[1] * sin_angle,
                    axis[2] * axis[1] * (1 - cos_angle) + axis[0] * sin_angle,
                    cos_angle + axis[2] ** 2 * (1 - cos_angle),
                ],
            ]
        )
        # Rotate the x, y, and z vectors
        self.x_vec = R @ self.x_vec
        self.y_vec = R @ self.y_vec
        self.z_vec = R @ self.z_vec
        # Rotate the vertices
        self.vertices = (R @ (self.vertices - self.center).T).T + self.center
        # Update the faces
        self.get_vertices_faces()

    def plot(self, pl, opacity=0.5, **kwargs):
        """
        Plots the rectangular region.

        Parameters
        ----------
        pl : pyvista.Plotter
            The pyvista plotter to plot on.
        opacity : float, optional
            The opacity of the rectangular region, by default 0.5.
        kwargs : dict
            Additional keyword arguments for the plot.
        """

        # Create the pyvista mesh
        mesh = pv.PolyData(self.vertices, self.faces)
        pl.add_mesh(
            mesh,
            show_edges=True,
            show_vertices=True,
            vertex_color=REGION_COLOR[self._region_type],
            color=REGION_COLOR[self._region_type],
            edge_opacity=1.0,
            opacity=opacity,
            render_points_as_spheres=True,
            point_size=5,
            **kwargs,
        )

    def map_point(self, point):
        """
        Maps a point from the global coordinate system to the local coordinate system of the rectangular region.

        Parameters
        ----------
        point : list or np.ndarray
            The point to map.

        Returns
        -------
        np.ndarray
            The mapped point in the local coordinate system of the rectangular region.
        """
        point = np.array(point)
        local_point = point - self.center
        local_point = np.array(
            [
                np.dot(local_point, self.x_vec),
                np.dot(local_point, self.y_vec),
                np.dot(local_point, self.z_vec),
            ]
        )
        return local_point

    def check_point(self, point):
        """
        Checks if a point is inside the rectangular region.

        Parameters
        ----------
        point : list or np.ndarray
            The point to check.

        Returns
        -------
        bool
            True if the point is inside the rectangular region, False otherwise.
        """
        point = np.array(point)
        local_point = self.map_point(point)
        return (
            (abs(local_point[0]) <= 0.5 * self.xl)
            and (abs(local_point[1]) <= 0.5 * self.yl)
            and (abs(local_point[2]) <= 0.5 * self.zl)
        )

    def check_fractures(self, fractures):
        """
        Checks if any of the fractures are inside the rectangular region.

        Parameters
        ----------
        fractures : list of andfn.fracture.Fracture
            The fractures to check.

        Returns
        -------
        inside_fractures : list of andfn.fracture.Fracture
            The fractures that are inside the rectangular region.
        outside_fractures : list of andfn.fracture.Fracture
            The fractures that are outside the rectangular region.
        """
        inside_fractures = []
        outside_fractures = []
        for frac in fractures:
            if self.check_point(frac.center):
                inside_fractures.append(frac)
            else:
                outside_fractures.append(frac)
        return inside_fractures, outside_fractures

    def possible_intersections(self, frac):
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
        # Check if the fracture is within the bounding box of the tunnel
        diagonal = np.sqrt(self.xl**2 + self.yl**2 + self.zl**2)
        distance = np.linalg.norm(frac.center - self.center)
        return distance <= diagonal / 2 + frac.radius

    def frac_intersections(self, fractures, face, head):
        """
        Checks if the tunnel intersects with a given fracture.

        Parameters
        ----------
        fractures : andfn.fracture.Fracture
            The fracture to check for intersections with the tunnel.
        face : int | str
            The face of the rectangular region to check for intersections with the tunnel.
            They are numbered as follows:

            0/top: top face (x-y plane)

            1/bottom: bottom face (x-y plane)

            2/front: front face (x-z plane)

            3/back: back face (x-z plane)

            4/left: left face (y-z plane)

            5/right: right face (y-z plane)

        head : float
            The head value to assign to the constant head elements created for the tunnel in the fracture.

        Returns
        -------
        bool
            True if the tunnel elements are added to the fracture, False otherwise.
        """
        if not isinstance(fractures, list):
            fractures = [fractures]
        for frac in fractures:
            # Check if the tunnel can possibly intersect with the fracture
            if not self.possible_intersections(frac):
                continue
            # calculate the intersection points between line between the verticies and the fracture plane
            # The vertices are ordered as follows:
            face_map = {
                0: "top",
                1: "bottom",
                2: "front",
                3: "back",
                4: "left",
                5: "right",
            }
            if isinstance(face, int):
                try:
                    face = face_map[face]
                except KeyError:
                    raise ValueError("Face must be an integer between 0 and 5.")
            pnts = []
            verts = self.faces_dict[face]
            for i in range(4):
                pnt = self.line_plane_intersection(
                    self.vertices[verts[i]],
                    self.vertices[verts[(i + 1) % 4]],
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

            # Create constant head elements for the tunnel in this fracture
            self.assign_elements(frac, pnts_inside, pnts, head)

    def assign_elements(self, frac, pnts_inside, pnts, head):
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
        if len(pnts_inside) == len(pnts):
            z0 = gf.map_3d_to_2d(pnts_inside[0], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[-1], frac)
            ch = ConstantHeadLine(
                f"tunnel_{self.label}_frac_{frac.label}_{11}",
                np.array([z0, z1]),
                head,
                frac,
            )
            self.elements.append(ch)
        else:
            # Create a constant head line for each segment of the tunnel inside the fracture
            z0 = gf.map_3d_to_2d(pnts_inside[0], frac)
            z1 = gf.map_3d_to_2d(pnts_inside[1], frac)
            ch = ConstantHeadLine(
                f"tunnel_{self.label}_frac_{frac.label}_{0}",
                np.array([z0, z1]),
                head,
                frac,
            )
            self.elements.append(ch)
        self.fracs.append(frac)

    # TODO: Move of geometry functions to geometry_functions.py
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


if __name__ == "__main__":
    print("\n")
    print("--------------------------------------------------------------------")
    print("\t TESTING THE REGION CLASSES")
    print("--------------------------------------------------------------------")

    # Create a rectangular region
    rect_region = RectangularRegion(
        label="rect_region",
        center=[0, 0, 0],
        x_vec=[np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        y_vec=[-np.sqrt(2) / 2, np.sqrt(2) / 2, 0],
        z_vec=[0, 0, 1],
        xl=3,
        yl=3,
        zl=3,
    )

    # Add some points to check if they are inside the rectangular region
    points = np.random.uniform(-3, 3, size=(100, 3))

    # Plot the rectangular region
    pl = pv.Plotter()
    rect_region.plot(pl)
    # add the points to the plot
    for point in points:
        if rect_region.check_point(point):
            pl.add_mesh(pv.Sphere(radius=0.1, center=point), color="blue")
        else:
            pl.add_mesh(pv.Sphere(radius=0.1, center=point), color="red")
    # Add descriptive text
    pl.add_text(
        "Red points are outside the rectangular region",
        position=(10, 40),
        font_size=10,
        color="black",
    )
    pl.add_text("Red", position=(10, 40), font_size=10, color="red")
    pl.add_text(
        "Blue points are inside the rectangular region",
        position=(10, 10),
        font_size=10,
        color="black",
    )
    pl.add_text("Blue", position=(10, 10), font_size=10, color="blue")

    # show axes
    pl.show_axes()
    pl.show()
