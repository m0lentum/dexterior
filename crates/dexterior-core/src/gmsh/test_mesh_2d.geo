// simple 2d mesh for unit tests in the gmsh module

Point(1) = {-1, 0, 0, 0.2};
Point(2) = {1, 0, 0, 0.2};
Point(3) = {0, 1, 0, 0.2};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};
Curve Loop(1) = {1, 2, 3};
Plane Surface(1) = {1};

Physical Point(100) = {1, 2};
Physical Curve(100) = {1};
