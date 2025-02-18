// 2d square with a star-shaped hole for acoustic scatterer simulation

Mesh.SaveAll = 1;
Mesh.Smoothing = 100;

scaling = 4.;
size = scaling * Pi;
lc = Pi / 12.0;

// outer square
Point(1) = {-size, -size, 0, lc};
Point(2) = {size, -size, 0, lc};
Point(3) = {size, size, 0, lc};
Point(4) = {-size, size, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
square_cl = newcl;
Curve Loop(square_cl) = {1, 2, 3, 4};
// naming physical groups with strings not yet supported.
// group 10 is the outer boundary
Physical Point(10) = {1, 2, 3, 4};
Physical Curve(10) = {1, 2, 3, 4};

// inner star
r_outer = scaling * Pi / 2;
r_inner = scaling * Pi / 6;
points = 5;
angle_incr = Pi * 2 / points;
For i In {0 : points-1}
  po = newp; Point(po) = {r_outer * Sin(i * angle_incr), r_outer * Cos(i * angle_incr), 0, lc};
  pi = newp; Point(pi) = {r_inner * Sin((i + 0.5) * angle_incr), r_inner * Cos((i + 0.5) * angle_incr), 0, lc};
  star_points[2*i] = po;
  star_points[2*i+1] = pi;
EndFor
For i In {0 : 2*(points-1)}
  l = newl; Line(l) = {star_points[i], star_points[i+1]};
  star_lines[i] = l;
EndFor
l = newl; Line(l) = {star_points[2*points-1], star_points[0]};
star_lines[2*points-1] = l;
star_cl = newcl;
Curve Loop(star_cl) = {star_lines[]};

Physical Point(100) = {star_points[]};
Physical Curve(100) = {star_lines[]};

Plane Surface(1) = {square_cl, star_cl};
