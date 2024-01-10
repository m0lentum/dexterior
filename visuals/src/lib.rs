//! Real-time visualization for simulations implemented with [`dexterior`].

pub mod animation;
pub use animation::{Animation, AnimationParams};

pub mod camera;

pub mod color_map;
pub use color_map::builtin_color_maps;

pub(crate) mod pipelines;
pub use pipelines::{
    line::{CapStyle, CapsStyle, JoinStyle, LineDrawingMode, LineParameters, LineWidth},
    Painter,
};

pub mod render_window;
pub use render_window::{RenderWindow, WindowInitError, WindowParams};

pub use palette;
