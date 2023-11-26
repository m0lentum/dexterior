// square 2D mesh with domain [0, pi] x [0, pi]

lc = Pi / 20.0;
Point(1) = {0, 0, 0, lc};
Point(2) = {Pi, 0, 0, lc};
Point(3) = {Pi, Pi, 0, lc};
Point(4) = {0, Pi, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
