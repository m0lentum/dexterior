//! Real-time visualization for simulations implemented with `dexterior`.
//!
//! This crate is in an early state of development,
//! currently only supporting a handful of 2D use cases.
//! More features and an extension to 3D are planned in the future,
//! and major API changes are likely to occur as a result.
//!
//! To visualize a simulation,
//! first create a [`RenderWindow`],
//! then call [`run_animation`][`RenderWindow::run_animation`] on it.
//! See the examples in the [dexterior repo][repo] for concrete usage.
//!
//! [repo]: https://github.com/m0lentum/dexterior

#![warn(missing_docs)]

mod animation;
#[doc(inline)]
pub use animation::{Animation, AnimationParams, AnimationState};

mod camera;

mod color_map;
#[doc(inline)]
pub use color_map::{builtin_color_maps, Color, ColorMap};

pub(crate) mod pipelines;
#[doc(inline)]
pub use pipelines::{
    axes::AxesParams,
    line::{CapStyle, CapsStyle, JoinStyle, LineDrawingMode, LineParams, LineWidth},
    ArrowParams, Painter, WireframeParams,
};

mod render_window;
#[doc(inline)]
pub use render_window::{RenderWindow, WindowParams};

pub use palette;
