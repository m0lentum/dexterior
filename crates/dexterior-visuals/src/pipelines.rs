mod resources;
use resources::SharedResources;

pub(crate) mod line;
use line::{LineDrawingMode, LineParams, LinePipeline};

pub(crate) mod axes;
use axes::AxesParams;

mod vertex_colors;
use vertex_colors::{VertexColorVariant, VertexColorsPipeline};

pub(crate) mod text;
use text::{CachedText, TextParams, TextPipeline};

//

use itertools::izip;
use nalgebra as na;
use std::{collections::HashMap, sync::OnceLock};

use crate::{
    camera::Camera,
    render_window::{ActiveRenderWindow, RenderContext},
};
use dexterior_core as dex;

pub(crate) struct Renderer {
    // pipelines that aren't always used are created lazily on demand
    gradient_pl: OnceLock<VertexColorsPipeline>,
    flat_tri_pl: OnceLock<VertexColorsPipeline>,
    line_pl: LinePipeline,
    text_pl: TextPipeline,
    // some GPU resources are shared between different pipelines
    pub resources: SharedResources,
    // map from names to indices in the color map collection
    // for picking a map by name
    color_map_names: HashMap<String, usize>,
    state: RendererState,
}

/// Any state that varies during a frame should be here if possible
/// (not inside one of the pipeline modules),
/// especially if it needs to be reset between frames.
pub(crate) struct RendererState {
    /// Index of the active color map in the array of all available color maps.
    color_map_idx: usize,
    /// Range of values mapped onto the color map.
    color_map_range: Option<std::ops::Range<f32>>,
    /// Number of color-mapped draw calls made this frame.
    /// Used to determine which buffer to use for the next call,
    /// since we can't upload to the same one multiple times per frame.
    next_color_data: usize,
    /// Same for the line renderer,
    /// which builds a buffer per call during a frame.
    next_line_data: usize,
}

impl Renderer {
    pub fn new(window: &ActiveRenderWindow, params: &crate::AnimationParams) -> Self {
        let resources = SharedResources::new(window, &params.color_maps);

        let color_map_names: HashMap<String, usize> = params
            .color_maps
            .iter()
            .enumerate()
            .map(|(idx, map)| (map.name.clone(), idx))
            .collect();

        let color_map_idx = if let Some(&idx) = params
            .initial_color_map
            .as_ref()
            .and_then(|name| color_map_names.get(name))
        {
            idx
        } else {
            0
        };

        Self {
            gradient_pl: OnceLock::new(),
            flat_tri_pl: OnceLock::new(),
            line_pl: LinePipeline::new(window, &resources),
            text_pl: TextPipeline::new(window),
            resources,
            color_map_names,
            state: RendererState {
                color_map_idx,
                color_map_range: params.color_map_range.clone(),
                next_color_data: 0,
                next_line_data: 0,
            },
        }
    }

    /// Activate the next color map in order.
    pub fn cycle_color_maps(&mut self) {
        self.state.color_map_idx = (self.state.color_map_idx + 1) % self.color_map_names.len();
    }

    /// Perform any draw operations that need to be batched (text, currently).
    pub fn draw_batched(&mut self, ctx: &mut RenderContext, camera: &Camera) {
        self.text_pl.draw(ctx, camera);
    }

    /// Reset any state accumulated within a frame
    /// to prepare for the next one.
    pub fn end_frame(&mut self) {
        self.state.next_color_data = 0;
        self.state.next_line_data = 0;
    }
}

/// The main user interface for drawing graphics.
///
/// See [`Animation`][super::animation::Animation]
/// and the crate examples for usage.
pub struct Painter<'a, 'ctx: 'a> {
    pub(crate) ctx: &'a mut RenderContext<'ctx>,
    pub(crate) rend: &'a mut Renderer,
    pub(crate) mesh: &'a dex::SimplicialMesh<2>,
}

impl<'a, 'ctx: 'a> Painter<'a, 'ctx> {
    /// Set the active color map by name.
    ///
    /// If the given name is not found in the loaded maps,
    /// does not change the map.
    pub fn set_color_map(&mut self, name: &str) {
        if let Some(idx) = self.rend.color_map_names.get(name) {
            self.rend.state.color_map_idx = *idx;
        }
    }

    /// Set the active color map by its index in the array of loaded maps.
    pub fn set_color_map_index(&mut self, idx: usize) {
        self.rend.state.color_map_idx = idx;
    }

    /// Set the range of values that gets mapped onto the active color map.
    /// Values outside the range are clamped to its ends.
    ///
    /// If the color map is never set, the range is computed
    /// as the minimum and maximum of the given values
    /// whenever a renderer using the color map is called.
    /// This is usually undesirable as it will change the range when the simulation state changes,
    /// so setting a constant value is recommended.
    pub fn set_color_map_range(&mut self, range: std::ops::Range<f32>) {
        self.rend.state.color_map_range = Some(range);
    }

    /// Draw a primal 0-cochain by coloring mesh vertices
    /// according to the active color map and interpolating colors for the triangles between.
    pub fn vertex_colors(&mut self, c: &dex::Cochain<0, dex::Primal>) {
        self.draw_vertex_colors(c.values.as_slice());
    }

    /// Draw a dual 2-cochain by setting its values as colors
    /// at the corresponding primal mesh vertices.
    pub fn vertex_colors_dual(&mut self, c: &dex::Cochain<2, dex::Dual>) {
        self.draw_vertex_colors(c.values.as_slice());
    }

    fn draw_vertex_colors(&mut self, values: &[f64]) {
        let gradient_pl = self.rend.gradient_pl.get_or_init(|| {
            VertexColorsPipeline::new(
                self.ctx,
                self.mesh,
                &self.rend.resources,
                VertexColorVariant::InterpolatedVertices,
            )
        });
        gradient_pl.draw(
            &mut self.rend.resources,
            self.ctx,
            &mut self.rend.state,
            values,
        );
    }

    /// Draw a primal 2-cochain as flat-colored triangles.
    pub fn triangle_colors(&mut self, c: &dex::Cochain<2, dex::Primal>) {
        self.draw_triangle_colors(c.values.as_slice());
    }

    /// Draw a dual 0-cochain as flat colors
    /// on the corresponding primal mesh triangles.
    pub fn triangle_colors_dual(&mut self, c: &dex::Cochain<0, dex::Dual>) {
        self.draw_triangle_colors(c.values.as_slice());
    }

    fn draw_triangle_colors(&mut self, values: &[f64]) {
        let tri_pl = self.rend.flat_tri_pl.get_or_init(|| {
            VertexColorsPipeline::new(
                self.ctx,
                self.mesh,
                &self.rend.resources,
                VertexColorVariant::FlatTriangles,
            )
        });
        tri_pl.draw(
            &mut self.rend.resources,
            self.ctx,
            &mut self.rend.state,
            values,
        );
    }

    /// Draw a primal 1-cochain representing a velocity in the edge tangent direction
    /// as arrows interpolated at triangle barycenters.
    pub fn velocity_arrows(&mut self, c: &dex::Cochain<1, dex::Primal>, params: ArrowParams) {
        self.arrows(c, params, |v| v);
    }

    /// Draw a primal 1-cochain representing a flux in the edge normal direction
    /// as arrows interpolated at triangle barycenters.
    pub fn flux_arrows(&mut self, c: &dex::Cochain<1, dex::Primal>, params: ArrowParams) {
        // the interpolation works as if the cochain was integrated in the tangent direction;
        // the actual direction of flux is orthogonal to that
        self.arrows(c, params, |v| na::Vector2::new(v.y, -v.x));
    }

    /// Draw interpolated arrows potentially with a transformation applied.
    /// An abstraction to reduce duplication between velocity and flux arrow methods.
    fn arrows(
        &mut self,
        c: &dex::Cochain<1, dex::Primal>,
        params: ArrowParams,
        map_arrow: impl Fn(na::Vector2<f64>) -> na::Vector2<f64>,
    ) {
        let mut arrow_segments: Vec<na::Vector3<f64>> = Vec::new();

        let arrow_vecs: Vec<na::Vector2<f64>> = self
            .mesh
            .barycentric_interpolate_1(c)
            .into_iter()
            .map(map_arrow)
            .collect();
        for (bary, arrow) in izip!(self.mesh.barycenters::<2>(), arrow_vecs) {
            arrow_segments.push(na::Vector3::new(bary.x, bary.y, 0.));
            let end = bary + params.scaling * arrow;
            arrow_segments.push(na::Vector3::new(end.x, end.y, 0.));
        }

        let params = LineParams {
            width: params.width,
            color: params.color,
            caps: crate::CapsStyle::arrows(),
            ..Default::default()
        };
        self.lines(params, LineDrawingMode::List, &arrow_segments);
    }

    /// Draw a wireframe model of the simulation mesh.
    pub fn wireframe(&mut self, params: WireframeParams) {
        let params = params.line_params();
        let points = Self::wireframe_points(self.mesh.simplices::<1>());
        self.lines(params, LineDrawingMode::List, &points);
    }

    /// Draw a wireframe of a subset of edges in the simulation mesh.
    pub fn wireframe_subset(
        &mut self,
        params: WireframeParams,
        subset: &dex::Subset<1, dex::Primal>,
    ) {
        let params = params.line_params();
        let points = Self::wireframe_points(self.mesh.simplices_in(subset));
        self.lines(params, LineDrawingMode::List, &points);
    }

    fn wireframe_points(simplices: dex::SimplexIter<'_, 1, 2>) -> Vec<na::Vector3<f64>> {
        simplices
            .flat_map(|edge| {
                // collect into an array to avoid nested borrow problems
                let mut vs = [na::Vector3::zeros(); 2];
                for (idx, vert) in edge.vertices().enumerate() {
                    vs[idx] = na::Vector3::new(vert.x, vert.y, 0.);
                }
                vs
            })
            .collect()
    }

    /// Draw a wireframe model of the dual mesh.
    pub fn dual_wireframe(&mut self, params: WireframeParams) {
        let params = params.line_params();
        let points = Self::dual_wireframe_points(self.mesh.dual_cells::<1>());
        self.lines(params, LineDrawingMode::List, &points);
    }

    /// Draw a wireframe of a subset of edges in the dual mesh.
    pub fn dual_wireframe_subset(
        &mut self,
        subset: &dex::Subset<1, dex::Dual>,
        params: WireframeParams,
    ) {
        let params = params.line_params();
        let points = Self::dual_wireframe_points(self.mesh.dual_cells_in(subset));
        self.lines(params, LineDrawingMode::List, &points);
    }

    fn dual_wireframe_points(cells: dex::DualCellIter<'_, 1, 2>) -> Vec<na::Vector3<f64>> {
        cells
            .flat_map(|dual_edge| {
                let mut end_tris = dual_edge.dual().coboundary();
                let start = end_tris.next().unwrap().1.circumcenter();
                let start = na::Vector3::new(start.x, start.y, 0.);
                let end = match end_tris.next() {
                    Some((_, other_tri)) => other_tri.circumcenter(),
                    // boundary edges don't have a second coboundary simplex,
                    // instead the dual edge ends at the boundary
                    // (iterating over these points should be added
                    // to the dual cell iterator at some point,
                    // but doing it properly in a dimension-agnostic way
                    // takes a bit of effort)
                    None => dual_edge.dual().circumcenter(),
                };
                let end = na::Vector3::new(end.x, end.y, 0.);
                [start, end]
            })
            .collect()
    }

    /// Draw a set of axes around the mesh.
    pub fn axes_2d(&mut self, params: AxesParams) {
        axes::axes_2d(self, params);
    }

    /// Draw a list of line segments.
    ///
    /// Every two points in `points` define a distinct segment,
    /// with a gap left between them.
    #[inline]
    pub fn line_list(&mut self, params: LineParams, points: &[na::Vector3<f64>]) {
        self.lines(params, LineDrawingMode::List, points);
    }

    /// Draw a strip of line segments.
    ///
    /// Every point in `points` is connected
    /// to both the next and previous one with a line segments.
    #[inline]
    pub fn line_strip(&mut self, params: LineParams, points: &[na::Vector3<f64>]) {
        self.lines(params, LineDrawingMode::Strip, points);
    }

    fn lines(&mut self, params: LineParams, mode: LineDrawingMode, points: &[na::Vector3<f64>]) {
        let points: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rend.line_pl.draw(
            &self.rend.resources,
            &mut self.rend.state,
            self.ctx,
            params,
            mode,
            &points,
        );
    }

    /// Draw a custom piece of text.
    #[inline]
    pub fn text<const POS_DIM: usize>(&mut self, text: TextParams<'_, POS_DIM>) {
        self.rend.text_pl.create_and_queue(text);
    }

    /// Draw a piece of text that has been previously cached.
    /// More efficient than [`text`][Self::text] for unchanging text.
    ///
    /// Currently not exposed to the user
    /// because it's difficult to create a cached text buffer in the user-facing API
    /// and users are unlikely to need large amounts of custom text.
    #[inline]
    pub(crate) fn cached_text(&mut self, text: &CachedText) {
        self.rend.text_pl.queue_buffer(text);
    }
}

/// Parameters to configure the drawing of mesh wireframes.
#[derive(Clone, Copy, Debug)]
pub struct WireframeParams {
    /// Width of the lines.
    /// A good value depends on the scale of the mesh.
    /// Default: 0.01 world space units.
    pub width: crate::LineWidth,
    /// Color of the lines.
    /// Default: black.
    pub color: palette::LinSrgb,
}

impl Default for WireframeParams {
    fn default() -> Self {
        Self {
            width: crate::LineWidth::WorldUnits(0.01),
            color: palette::named::BLACK.into(),
        }
    }
}

impl WireframeParams {
    fn line_params(&self) -> LineParams {
        LineParams {
            width: self.width,
            color: self.color,
            joins: crate::JoinStyle::None,
            caps: crate::CapsStyle::both(crate::CapStyle::Circle),
        }
    }
}

/// Parameters to configure the drawing of arrows.
#[derive(Clone, Copy, Debug)]
pub struct ArrowParams {
    /// Coefficient to scale the arrow length by.
    /// A good value depends on the scale of the mesh.
    /// Default: 0.1.
    pub scaling: f64,
    /// Width of the arrows.
    /// Default: 0.01 world space units.
    pub width: crate::LineWidth,
    /// Color of the arrows.
    /// Default: black.
    pub color: palette::LinSrgb,
}

impl Default for ArrowParams {
    fn default() -> Self {
        Self {
            scaling: 0.1,
            width: crate::LineWidth::WorldUnits(0.01),
            color: palette::named::BLACK.into(),
        }
    }
}
