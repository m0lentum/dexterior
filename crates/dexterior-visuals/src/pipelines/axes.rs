use crate::{
    pipelines::{
        text::{TextAnchor, TextBuffer, TextParams},
        LineDrawingMode,
    },
    CapStyle, CapsStyle, LineParams, LineWidth,
};

use nalgebra as na;

/// Parameters to configure the drawing of axis lines.
#[derive(Clone, Copy, Debug)]
pub struct AxesParams {
    /// Distance between the extreme corners of the mesh
    /// and the axis lines in world space.
    /// Default: 0.2.
    pub padding: f32,
    /// Interval between major tick marks along the axis line.
    /// Default: 1.0.
    pub tick_interval: f32,
    /// Length of major ticks along the axis line in world space.
    /// Default: 0.05.
    pub tick_length: f32,
    /// Number of minor tick lines between each major one.
    /// Default: 9.
    pub minor_ticks: usize,
    /// Width of the main axis lines and major ticks.
    /// Minor ticks have a fraction of this width.
    /// Default: 4 screenspace pixels.
    pub width: LineWidth,
    /// Color of the lines.
    /// Default: `palette::named::BLACK.into()`.
    pub color: palette::LinSrgb,
}

impl Default for AxesParams {
    fn default() -> Self {
        Self {
            padding: 0.2,
            tick_interval: 1.,
            tick_length: 0.05,
            minor_ticks: 9,
            width: LineWidth::ScreenPixels(4.),
            color: palette::named::BLACK.into(),
        }
    }
}

pub(crate) fn axes_2d(painter: &mut super::Painter, params: AxesParams) {
    let mut draw = |line_params: LineParams, points: &[[f32; 3]]| {
        painter.rend.line_pl.draw(
            &painter.rend.resources,
            &mut painter.rend.state,
            painter.ctx,
            line_params,
            LineDrawingMode::List,
            points,
        );
    };

    let params_proto = LineParams {
        width: params.width,
        color: params.color,
        ..Default::default()
    };

    // generate main axis lines
    let bounds = painter.mesh.bounds();
    let origin = [
        bounds.min.x as f32 - params.padding,
        bounds.min.y as f32 - params.padding,
        0.0,
    ];
    let x_max = [bounds.max.x as f32 + params.padding, origin[1], 0.0];
    let y_max = [origin[0], bounds.max.y as f32 + params.padding, 0.0];

    draw(
        LineParams {
            caps: CapsStyle {
                start: CapStyle::Circle,
                end: CapStyle::Arrow,
            },
            ..params_proto
        },
        &[origin, x_max, origin, y_max],
    );

    // generate tick lines at regular intervals

    let minor_tick_interval = params.tick_interval / (params.minor_ticks + 1) as f32;
    let minor_tick_halflength = params.tick_length * 0.75 / 2.;

    let mut major_tick_points: Vec<[f32; 3]> = Vec::new();
    let mut minor_tick_points: Vec<[f32; 3]> = Vec::new();
    let mut labels: Vec<TextBuffer> = Vec::new();
    // this should generalize to 3D fairly easily, I hope
    for axis_idx in 0..2 {
        let cross_axis_idx = (axis_idx + 1) % 2;

        // start at the first multiple of TICK_INTERVAL in global space,
        // not on the axis line (which can be anywhere)
        let initial_tick = origin[axis_idx] + params.tick_interval
            - origin[axis_idx].rem_euclid(params.tick_interval);
        let bound_size = bounds.max[axis_idx] as f32 - initial_tick;
        let tick_count = (bound_size / params.tick_interval) as usize;

        for major_idx in 0..=tick_count {
            let mut tick_points = [[0., 0., 0.], [0., 0., 0.]];

            let major_coord = initial_tick + major_idx as f32 * params.tick_interval;

            tick_points[0][axis_idx] = major_coord;
            tick_points[0][cross_axis_idx] = origin[cross_axis_idx] + params.tick_length / 2.;
            tick_points[1][axis_idx] = major_coord;
            tick_points[1][cross_axis_idx] = origin[cross_axis_idx] - params.tick_length / 2.;
            major_tick_points.extend_from_slice(&tick_points);

            labels.push(painter.rend.text_pl.create_buffer(TextParams {
                text: &format!(" {major_coord:.1} "),
                position: na::Vector3::new(
                    tick_points[1][0] as f64,
                    tick_points[1][1] as f64,
                    tick_points[1][2] as f64,
                ),
                anchor: if axis_idx == 0 {
                    TextAnchor::TopMid
                } else {
                    TextAnchor::MidRight
                },
                font_size: 16.,
                line_height: 24.,
                attrs: glyphon::Attrs::new().weight(glyphon::Weight::BOLD),
                ..Default::default()
            }));

            if major_idx == tick_count {
                break;
            }

            // minor ticks

            for minor_idx in 1..=params.minor_ticks {
                let minor_coord = major_coord + minor_idx as f32 * minor_tick_interval;

                tick_points[0][axis_idx] = minor_coord;
                tick_points[0][cross_axis_idx] = origin[cross_axis_idx] + minor_tick_halflength;
                tick_points[1][axis_idx] = minor_coord;
                tick_points[1][cross_axis_idx] = origin[cross_axis_idx] - minor_tick_halflength;
                minor_tick_points.extend_from_slice(&tick_points);
            }
        }
    }

    // TODO: caching of lines like we cache text;
    // only generate all of this once and store it until parameters change

    draw(params_proto, &major_tick_points);
    draw(
        LineParams {
            width: params.width * 0.5,
            ..params_proto
        },
        &minor_tick_points,
    );
    for label in labels {
        painter.cached_text(&label.cache());
    }
}
