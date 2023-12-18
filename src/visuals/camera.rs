//! Tools for creating and controlling interactive cameras.

use nalgebra as na;

/// A camera determines where the viewer of a visualization is in 3D space.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// The position, angle and scaling of the viewer.
    ///
    /// The negative Z axis of this space is the view direction.
    pub pose: na::Similarity3<f32>,
    /// Whether the camera moves in a 2D plane
    /// or orbits around a 3D point.
    pub control_mode: CameraControlMode,
    /// How the camera projects the world onto the screen.
    pub projection: Projection,
}

/// How the camera responds to mouse controls (not implemented yet!).
#[derive(Clone, Copy, Debug)]
pub enum CameraControlMode {
    /// Move in a 2D plane, never rotate. Best for 2D.
    Planar2D,
    /// Rotate around a point. Best for 3D.
    Spherical3D,
}

/// How a [`Camera`] projects the world onto the screen.
#[derive(Clone, Copy, Debug)]
pub enum Projection {
    /// Orthographic projection without clipping planes. Used for 2D.
    Orthographic(
        /// The dimensions of the display area.
        na::Vector2<f32>,
    ),
    /// Perspective projection for 3D views, not implemented yet.
    Perspective,
}

impl From<Projection> for na::Matrix4<f32> {
    fn from(proj: Projection) -> na::Matrix4<f32> {
        match proj {
            Projection::Orthographic(diag) => na::Matrix4::from_diagonal(&na::Vector4::new(
                2.0 / (diag.x),
                2.0 / (diag.y),
                0.0,
                1.0,
            )),
            Projection::Perspective => {
                todo!("Perspective projection is not implemented yet")
            }
        }
    }
}

impl Camera {
    /// Create a camera with good defaults for 2D.
    pub fn new_2d(
        min_corner: na::Vector2<f32>,
        max_corner: na::Vector2<f32>,
        padding: f32,
    ) -> Self {
        let diagonal = max_corner - min_corner;
        let center = (min_corner + max_corner) / 2.0;
        Self {
            // place the camera in the center of the view area above the mesh,
            // facing the xy plane (-z is the look direction, hence no rotation)
            pose: na::Similarity3::new(
                na::Vector3::new(center.x, center.y, 1.0),
                na::Vector3::zeros(),
                1.0,
            ),
            control_mode: CameraControlMode::Planar2D,
            projection: Projection::Orthographic(diagonal + na::Vector2::new(padding, padding)),
        }
    }

    pub(crate) fn view_projection_matrix(&self) -> na::Matrix4<f32> {
        // TODO: adapt to changing window aspect ratio to avoid stretching
        na::Matrix4::from(self.projection) * na::Matrix4::from(self.pose.inverse())
    }
}
