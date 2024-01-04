mod resources;
use resources::SharedResources;

mod line;
use line::{LineDrawingMode, LinePipeline};

mod wireframe;
use wireframe::WireframePipeline;

mod vertex_colors;
use vertex_colors::VertexColorsPipeline;

//

use itertools::Itertools;
use nalgebra as na;
use std::collections::HashMap;

use super::render_window::{RenderContext, RenderWindow};
use dexterior as dex;

pub(crate) struct Renderer {
    vertex_color_pl: WireframePipeline,
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
            vertex_color_pl: WireframePipeline::new(window, mesh, &resources),
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

    /// Draw a wireframe model of the simulation mesh.
    pub fn wireframe(&mut self) {
        self.rend
            .vertex_color_pl
            .draw(&self.rend.resources, self.ctx);
    }

    /// Draw a set of axes around the mesh.
    ///
    /// This is a super rough sketch and will be refined with configuration options later.
    pub fn axes_2d(&mut self) {
        // TODO: parameterize these
        const PADDING: f32 = 0.2;
        const TICK_INTERVAL: f32 = 1.0;
        const TICK_LENGTH: f32 = 0.05;

        // generate main axis lines
        let bounds = self.mesh.bounds();
        let origin = [
            bounds.min.x as f32 - PADDING,
            bounds.min.y as f32 - PADDING,
            0.0,
        ];
        let x_max = [bounds.max.x as f32 + PADDING, origin[1], 0.0];
        let y_max = [origin[0], bounds.max.y as f32 + PADDING, 0.0];

        let mut points = vec![origin, x_max, origin, y_max];

        // generate tick lines at regular intervals

        // this should generalize to 3D fairly easily, I hope
        for axis_idx in 0..2 {
            let cross_axis_idx = (axis_idx + 1) % 2;

            // start at the first multiple of TICK_INTERVAL in global space,
            // not on the axis line (which can be anywhere)
            let initial_tick =
                origin[axis_idx] + TICK_INTERVAL - origin[axis_idx].rem_euclid(TICK_INTERVAL);
            let bound_size = bounds.max[axis_idx] as f32 - initial_tick;
            let tick_count = (bound_size / TICK_INTERVAL) as usize;

            for tick_idx in 1..=tick_count {
                let mut tick_points = [[0., 0., 0.], [0., 0., 0.]];

                let tick_coord = initial_tick + tick_idx as f32 * TICK_INTERVAL;

                tick_points[0][axis_idx] = tick_coord;
                tick_points[0][cross_axis_idx] = origin[cross_axis_idx] + TICK_LENGTH / 2.;
                tick_points[1][axis_idx] = tick_coord;
                tick_points[1][cross_axis_idx] = origin[cross_axis_idx] - TICK_LENGTH / 2.;
                points.extend_from_slice(&tick_points);
            }
        }

        self.rend.line_pl.draw(
            &self.rend.resources,
            self.ctx,
            LineDrawingMode::List,
            &points,
        );
    }

    /// Draw a list of line segments.
    ///
    /// Every two points in `points` define a distinct segment,
    /// with a gap left between them.
    pub fn line_list(&mut self, points: &[na::Vector3<f64>]) {
        self.lines(points, LineDrawingMode::List);
    }

    /// Draw a strip of line segments.
    ///
    /// Every point in `points` is connected
    /// to both the next and previous one with a line segments.
    pub fn line_strip(&mut self, points: &[na::Vector3<f64>]) {
        self.lines(points, LineDrawingMode::Strip);
    }

    fn lines(&mut self, points: &[na::Vector3<f64>], mode: LineDrawingMode) {
        let points: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        self.rend
            .line_pl
            .draw(&self.rend.resources, self.ctx, mode, &points);
    }
}
