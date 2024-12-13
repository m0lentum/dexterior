use std::rc::Rc;

use glyphon as gh;
use itertools::izip;
use nalgebra as na;

use crate::{
    camera::Camera,
    render_window::{ActiveRenderWindow, RenderContext},
};

pub use gh::Color as TextColor;

pub(crate) struct TextPipeline {
    font_system: gh::FontSystem,
    swash_cache: gh::SwashCache,
    viewport: gh::Viewport,
    atlas: gh::TextAtlas,
    renderer: gh::TextRenderer,
    /// Glyphon requires us to draw all the text in one call,
    /// so we collect each text request into a queue and flush it at the end of a frame.
    /// Buffers are wrapped in Rcs to facilitate caching.
    pub draw_queue: Vec<Rc<TextBuffer>>,
}

/// Parameters for displaying text.
#[derive(Clone, Copy, Debug)]
pub struct TextParams<'a, const DIM: usize> {
    /// Content of the text to be displayed. Default: "".
    pub text: &'a str,
    /// Position of the anchor point in simulation space. Default: origin.
    pub position: na::SVector<f64, DIM>,
    /// Placement of the anchor point relative to the text. Default: center.
    pub anchor: TextAnchor,
    /// Fixed width for the text area in pixels. Default: None.
    /// If not given, the text is laid out based on its own dimensions.
    pub area_width: Option<f32>,
    /// Fixed height for the text area in pixels. Default: None.
    /// If not given, the text is laid out based on its own dimensions.
    pub area_height: Option<f32>,
    /// Font size of the text. Default: 30.
    pub font_size: f32,
    /// Line height of the text. Default: 42.
    pub line_height: f32,
    /// Color of the text. Default: black.
    pub color: TextColor,
    /// Set of cosmic-text attributes for the text buffer.
    /// Default: cosmic-text defaults with sans-serif font family.
    ///
    /// Types for the fields can be accessed through the library's re-export of [`glyphon`].
    pub attrs: gh::Attrs<'a>,
}

impl<'a, const DIM: usize> Default for TextParams<'a, DIM> {
    fn default() -> Self {
        Self {
            text: "",
            position: na::SVector::zeros(),
            anchor: TextAnchor::Center,
            area_width: None,
            area_height: None,
            font_size: 30.,
            line_height: 42.,
            color: TextColor::rgb(0, 0, 0),
            attrs: gh::Attrs::new().family(gh::Family::SansSerif),
        }
    }
}

/// Where to place a text field
/// relative to the position given in [`TextParams`].
///
/// For example, with an anchor of `TopMid`, the text will be placed
/// relative to position `x` as follows:
/// ```text
/// ----x----
/// |content|
/// ---------
/// ```
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub enum TextAnchor {
    TopLeft,
    TopMid,
    TopRight,
    MidLeft,
    Center,
    MidRight,
    BottomLeft,
    BottomMid,
    BottomRight,
}

/// A text buffer ready for rendering, created with [`TextPipeline::create_buffer`].
pub(crate) struct TextBuffer {
    buffer: gh::Buffer,
    position: na::Vector3<f32>,
    anchor: TextAnchor,
    color: TextColor,
}

/// A renderable piece of text that can be retained between frames
/// and drawn multiple times with [`Painter::cached_text`][crate::Painter::cached_text].
///
/// Currently only for internal use,
/// as creating these is only possible in the user-facing API in the `draw` method,
/// which requires the user to do some cumbersome first-frame initialization.
/// Drawing large amounts of custom text is probably a rare use case,
/// so the perf cost of redoing layout each frame is acceptable
/// to simplify the API, at least for now.
pub(crate) struct CachedText {
    buf: Rc<TextBuffer>,
}

impl TextBuffer {
    pub fn cache(self) -> CachedText {
        CachedText { buf: Rc::new(self) }
    }
}

impl TextPipeline {
    pub fn new(window: &ActiveRenderWindow) -> Self {
        let font_system = gh::FontSystem::new();
        let swash_cache = gh::SwashCache::new();
        let cache = gh::Cache::new(&window.device);
        let viewport = gh::Viewport::new(&window.device, &cache);
        let mut atlas = gh::TextAtlas::new(
            &window.device,
            &window.queue,
            &cache,
            window.swapchain_format(),
        );
        let renderer =
            gh::TextRenderer::new(&mut atlas, &window.device, window.multisample_state(), None);

        Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            renderer,
            draw_queue: Vec::new(),
        }
    }

    /// Create a renderable text buffer.
    /// This can be cached if desired.
    pub fn create_buffer<const DIM: usize>(&mut self, params: TextParams<'_, DIM>) -> TextBuffer {
        let mut buffer = gh::Buffer::new(
            &mut self.font_system,
            gh::Metrics {
                font_size: params.font_size,
                line_height: params.line_height,
            },
        );

        buffer.set_size(&mut self.font_system, params.area_width, params.area_height);
        buffer.set_text(
            &mut self.font_system,
            params.text,
            params.attrs,
            gh::Shaping::Advanced,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        // embed/project the position to 3D space for rendering
        let mut position = na::Vector3::zeros();
        for (param_c, c) in izip!(params.position.iter(), position.iter_mut()) {
            *c = *param_c as f32;
        }

        TextBuffer {
            buffer,
            position,
            anchor: params.anchor,
            color: params.color,
        }
    }

    /// Queue a buffer for drawing at the end of the frame.
    #[inline]
    pub fn queue_buffer(&mut self, text: &CachedText) {
        self.draw_queue.push(text.buf.clone());
    }

    /// Create a text buffer and queue it to be drawn this frame.
    /// Convenience method for text that isn't cached
    /// and thus gets dropped at the end of the frame.
    #[inline]
    pub fn create_and_queue<const DIM: usize>(&mut self, params: TextParams<'_, DIM>) {
        let buf = self.create_buffer(params);
        self.queue_buffer(&buf.cache());
    }

    /// Draw all text buffers that have been queued during this frame.
    pub fn draw(&mut self, ctx: &mut RenderContext, camera: &Camera) {
        self.viewport.update(
            ctx.queue,
            gh::Resolution {
                width: ctx.viewport_size.0,
                height: ctx.viewport_size.1,
            },
        );

        let view_proj = camera.view_projection_matrix(ctx.viewport_size);
        let half_vp = (
            ctx.viewport_size.0 as f32 / 2.,
            ctx.viewport_size.1 as f32 / 2.,
        );

        let areas = self.draw_queue.iter().map(|buf| {
            let anchor_pos_ndc =
                view_proj * na::Vector4::new(buf.position.x, buf.position.y, buf.position.z, 1.);
            let anchor_pixel = na::Vector2::new(
                anchor_pos_ndc.x * half_vp.0 + half_vp.0,
                -anchor_pos_ndc.y * half_vp.1 + half_vp.1,
            );

            let (width, height) = buf.buffer.size();
            let width = match width {
                Some(w) => w,
                // no explicit width given, compute from layout
                None => buf
                    .buffer
                    .layout_runs()
                    .map(|l| l.line_w)
                    .max_by(f32::total_cmp)
                    .unwrap_or(0.),
            };
            let height = match height {
                Some(h) => h,
                None => buf
                    .buffer
                    .layout_runs()
                    .last()
                    .map(|l| l.line_top + l.line_height)
                    .unwrap_or(0.),
            };

            use TextAnchor::*;
            let top_left_pixel = anchor_pixel
                - match buf.anchor {
                    TopLeft => na::Vector2::zeros(),
                    TopMid => na::Vector2::new(width / 2., 0.),
                    TopRight => na::Vector2::new(width, 0.),
                    MidLeft => na::Vector2::new(0., height / 2.),
                    Center => na::Vector2::new(width / 2., height / 2.),
                    MidRight => na::Vector2::new(width, height / 2.),
                    BottomLeft => na::Vector2::new(0., height),
                    BottomMid => na::Vector2::new(width / 2., height),
                    BottomRight => na::Vector2::new(width, height),
                };

            gh::TextArea {
                buffer: &buf.buffer,
                left: top_left_pixel.x,
                top: top_left_pixel.y,
                scale: 1.,
                bounds: gh::TextBounds {
                    left: top_left_pixel.x as i32,
                    top: top_left_pixel.y as i32,
                    right: (top_left_pixel.x + width + 1.) as i32,
                    bottom: (top_left_pixel.y + height + 1.) as i32,
                },
                default_color: buf.color,
                custom_glyphs: &[],
            }
        });

        self.renderer
            .prepare(
                ctx.device,
                ctx.queue,
                &mut self.font_system,
                &mut self.atlas,
                &self.viewport,
                areas,
                &mut self.swash_cache,
            )
            .expect("Failed to render text");

        let mut pass = ctx.pass("text");
        self.renderer
            .render(&self.atlas, &self.viewport, &mut pass)
            .expect("Failed to render text");

        self.draw_queue.clear();
    }
}
