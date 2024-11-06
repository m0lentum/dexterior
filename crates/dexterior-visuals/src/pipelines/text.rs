use glyphon as gh;
use itertools::izip;
use nalgebra as na;

use crate::{
    camera::Camera,
    render_window::{ActiveRenderWindow, RenderContext},
};

pub(crate) struct TextPipeline {
    font_system: gh::FontSystem,
    swash_cache: gh::SwashCache,
    viewport: gh::Viewport,
    atlas: gh::TextAtlas,
    renderer: gh::TextRenderer,
}

#[derive(Clone, Copy, Debug)]
pub struct TextParams<'a, const DIM: usize> {
    pub text: &'a str,
    pub position: na::SVector<f64, DIM>,
    pub anchor: TextAnchor,
    pub area_width: Option<f32>,
    pub area_height: Option<f32>,
    pub font_size: f32,
    pub line_height: f32,
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

pub(crate) struct TextBuffer {
    buffer: gh::Buffer,
    position: na::Vector3<f32>,
    anchor: TextAnchor,
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
        }
    }

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
            gh::Attrs::new().family(gh::Family::SansSerif),
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
        }
    }

    pub fn draw(&mut self, ctx: &mut RenderContext, camera: &Camera, buffers: &[&TextBuffer]) {
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

        let areas = buffers.iter().map(|buf| {
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
                // TODO configurable color
                default_color: gh::Color::rgb(0, 0, 0),
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
    }
}
