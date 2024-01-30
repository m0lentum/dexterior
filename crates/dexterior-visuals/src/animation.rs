//! Parameters for animated visuals.

use dexterior_core as dex;

use super::{color_map, pipelines::Painter};

/// An animated visualization of simulation data.
///
/// To display the animation in a window,
/// use [`RenderWindow::run_animation`][crate::RenderWindow::run_animation].
///
/// The `State` type must implement [`AnimationState`]
/// to facilitate interpolated rendering.
/// If you don't care about this, you can simply
/// leave the impl block empty:
/// ```
/// # use dexterior_visuals::AnimationState;
/// # #[derive(Clone)]
/// # struct MyState;
/// impl AnimationState for MyState {}
/// ```
/// Note that `MyState` must still implement `Clone`.
/// To enable interpolation, implement the `interpolate` member function
/// and call [`lerp`][dexterior_core::Cochain::lerp] for each cochain in your state
/// (and decide how/whether to interpolate anything else in your state that isn't a cochain):
/// ```
/// # use dexterior_visuals::AnimationState;
/// # use dexterior_core as dexterior;
/// use dexterior as dex;
/// #[derive(Clone)]
/// struct MyState {
///     p: dex::Cochain<0, dex::Primal>,
///     v: dex::Cochain<1, dex::Primal>,
/// };
/// impl AnimationState for MyState {
///     fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
///         Self {
///             p: old.p.lerp(&new.p, t),
///             v: old.v.lerp(&new.v, t),
///         }
///     }
/// }
/// ```
/// Note: this could be done with a macro, but it's a low-priority development,
/// so for now you have to write this boilerplate by hand.
pub struct Animation<'mesh, State, StepFn, DrawFn>
where
    State: AnimationState,
    StepFn: FnMut(&mut State),
    DrawFn: FnMut(&State, &mut Painter),
{
    /// The mesh that the simulation is derived from.
    ///
    /// For now, only 2D meshes are supported.
    /// This will change in the future.
    pub mesh: &'mesh dex::SimplicialMesh<2>,
    /// Time in seconds per simulation timestep.
    pub dt: f64,
    /// Control parameters.
    pub params: AnimationParams,
    /// Initial state of the simulation.
    pub state: State,
    /// A function that runs the simulation forward for a timestep.
    pub step: StepFn,
    /// A function that draws the simulation state.
    pub draw: DrawFn,
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
    /// Maximum number of timesteps to simulate between draws.
    /// Default: 4.
    ///
    /// If the simulation can't keep up with real time,
    /// it will slow down to this value.
    /// This serves as a controlled bound to avoid a spiral
    /// where the simulation falls farther and farther behind,
    /// trying to do more and more steps to catch up
    /// and ultimately throttling itself to a stop.
    pub max_steps_per_frame: usize,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            color_maps: super::color_map::builtin_color_maps::all_builtins(),
            color_map_range: None,
            initial_color_map: None,
            max_steps_per_frame: 4,
        }
    }
}

/// Trait required from states given to [`Animation`]s.
pub trait AnimationState: Clone {
    /// Interpolate between two states.
    ///
    /// Used to create smooth animations even if the simulation timestep is long.
    /// The default implementation does not do any interpolation.
    #[allow(unused_variables)]
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        new.clone()
    }
}
