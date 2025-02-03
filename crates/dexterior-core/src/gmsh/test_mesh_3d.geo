// simple 3d mesh for unit tests in the gmsh module

Mesh.SaveAll = 1;

Point(1) = {-1, -1, 0, 0.2};
Point(2) = {1, -1, 0, 0.2};
Point(3) = {1, 1, 0, 0.2};
Point(4) = {-1, 1, 0, 0.2};
Point(5) = {0, 0, 1, 0.2};

// bottom face of the tetrahedron
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Point(100) = {1, 2, 3, 4};
Physical Curve(100) = {1, 2, 3, 4};
Physical Surface(100) = {1};

// side faces
Line(5) = {1, 5};
Line(6) = {2, 5};
Line(7) = {3, 5};
Line(8) = {4, 5};
Curve Loop(2) = {1, 6, -5};
Plane Surface(2) = {2};
Curve Loop(3) = {2, 7, -6};
Plane Surface(3) = {3};
Curve Loop(4) = {3, 8, -7};
Plane Surface(4) = {4};
Curve Loop(5) = {4, 5, -8};
Plane Surface(5) = {5};

Surface Loop(1) = {1, 2, 3, 4, 5};
Volume(1) = {1};
