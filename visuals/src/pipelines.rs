mod resources;
use resources::SharedResources;

pub(crate) mod line;
use line::{LineDrawingMode, LineParams, LinePipeline};

pub(crate) mod axes;
use axes::AxesParams;

mod vertex_colors;
use vertex_colors::VertexColorsPipeline;

//

use itertools::{izip, Itertools};
use nalgebra as na;
use std::collections::HashMap;

use crate::render_window::{RenderContext, RenderWindow};
use dexterior as dex;

pub(crate) struct Renderer {
    gradient_pl: VertexColorsPipeline,
    line_pl: LinePipeline,
    // some GPU resources are shared between different pipelines
    pub resources: SharedResources,
    // map from names to indices in the color map collection
    // for picking a map by name
    color_map_names: HashMap<String, usize>,
    state: State,
}

struct State {
    color_map_idx: usize,
    color_map_range: Option<std::ops::Range<f32>>,
}

impl Renderer {
    pub fn new(
        window: &RenderWindow,
        mesh: &dex::SimplicialMesh<2>,
        params: &crate::AnimationParams,
    ) -> Self {
        let resources = SharedResources::new(window, mesh, &params.color_maps);

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
            gradient_pl: VertexColorsPipeline::new(window, mesh, &resources),
            line_pl: LinePipeline::new(window, &resources),
            resources,
            color_map_names,
            state: State {
                color_map_idx,
                color_map_range: params.color_map_range.clone(),
            },
        }
    }

    /// Activate the next color map in order.
    pub fn cycle_color_maps(&mut self) {
        self.state.color_map_idx = (self.state.color_map_idx + 1) % self.color_map_names.len();
    }

    /// Reset any state accumulated within a frame
    /// to prepare for the next one.
    pub fn end_frame(&mut self) {
        self.line_pl.end_frame();
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
        let vals_as_f32: Vec<f32> = c.values.iter().map(|&v| v as f32).collect();

        let color_map_range = if let Some(r) = &self.rend.state.color_map_range {
            r.clone()
        } else {
            use itertools::MinMaxResult::*;
            match vals_as_f32.iter().minmax() {
                NoElements | OneElement(_) => -1.0..1.0,
                MinMax(&l, &u) => l..u,
            }
        };

        self.rend.resources.upload_data_buffer(
            self.rend.state.color_map_idx,
            color_map_range,
            &vals_as_f32,
            self.ctx,
        );
        self.rend.gradient_pl.draw(&self.rend.resources, self.ctx);
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
        let params = LineParams {
            width: params.width,
            color: params.color,
            joins: crate::JoinStyle::None,
            caps: crate::CapsStyle::both(crate::CapStyle::Circle),
        };
        // TODO: caching of static lines to avoid re-uploading this every time
        // (also same for axes)
        let mut points: Vec<na::Vector3<f64>> = Vec::with_capacity(self.mesh.simplex_count::<1>());
        for simplex in self.mesh.simplices::<1>() {
            points.extend(simplex.vertices().map(|v| na::Vector3::new(v.x, v.y, 0.)));
        }
        self.lines(params, LineDrawingMode::List, &points);
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
        self.rend
            .line_pl
            .draw(&self.rend.resources, self.ctx, params, mode, &points);
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
