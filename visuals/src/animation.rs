//! Parameters for animated visuals.

use dexterior as dex;

use super::{color_map, pipelines::Painter};

/// An animated visualization of simulation data.
///
/// To display the animation in a window,
/// use [`RenderWindow::run_animation`][crate::RenderWindow::run_animation].
/// See the crate examples for usage.
pub struct Animation<'mesh, StepFn>
where
    StepFn: FnMut(&mut Painter),
{
    /// The mesh that the simulation is derived from.
    ///
    /// For now, only 2D meshes are supported.
    /// This will change in the future.
    pub mesh: &'mesh dex::SimplicialMesh<2>,
    /// Control parameters.
    pub params: AnimationParams,
    /// A closure that runs the simulation forward for a frame
    /// and determines what to draw.
    ///
    /// This closure takes one parameter, a [`Painter`],
    /// whose methods can be used to draw visualizations.
    /// Any state and environment variables are expected
    /// to be borrowed from the surrounding scope.
    pub step: StepFn,
}

/// Parameters to control aspects of an [`Animation`].
#[derive(Clone, Debug)]
pub struct AnimationParams {
    /// List of color maps available during drawing.
    /// Default: [`all_builtins`][color_map::builtin_color_maps::all_builtins].
    pub color_maps: Vec<color_map::ColorMap>,
    /// Range of values the color map maps to. Default: `None`.
    ///
    /// Can also be set with [`Painter::set_color_map_range`][Painter::set_color_map_range].
    /// See its documentation for more information.
    pub color_map_range: Option<std::ops::Range<f32>>,
    /// Set the active color map on startup. Default: `None`.
    ///
    /// If `None`, the color map is set to the first one in the `self.color_maps` array.
    pub initial_color_map: Option<String>,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            color_maps: super::color_map::builtin_color_maps::all_builtins(),
            color_map_range: None,
            initial_color_map: None,
        }
    }
}
