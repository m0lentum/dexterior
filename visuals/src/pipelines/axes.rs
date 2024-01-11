use crate::{pipelines::LineDrawingMode, CapStyle, CapsStyle, LineParameters, LineWidth};

/// Parameters to configure the drawing of axis lines.
#[derive(Clone, Copy, Debug)]
pub struct AxesParameters {
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
    /// Default: 3.
    pub minor_ticks: usize,
    /// Width of the main axis lines and major ticks.
    /// Minor ticks have a fraction of this width.
    /// Default: 4 screenspace pixels.
    pub width: LineWidth,
    /// Color of the lines.
    /// Default: `palette::named::BLACK.into()`.
    pub color: palette::LinSrgb,
}

impl Default for AxesParameters {
    fn default() -> Self {
        Self {
            padding: 0.2,
            tick_interval: 1.,
            tick_length: 0.05,
            minor_ticks: 3,
            width: LineWidth::ScreenPixels(4.),
            color: palette::named::BLACK.into(),
        }
    }
}

pub(crate) fn axes_2d(painter: &mut super::Painter, params: AxesParameters) {
    let mut draw = |line_params: LineParameters, points: &[[f32; 3]]| {
        painter.rend.line_pl.draw(
            &painter.rend.resources,
            painter.ctx,
            line_params,
            LineDrawingMode::List,
            points,
        );
    };

    let params_proto = LineParameters {
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
        LineParameters {
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

    let mut major_tick_points = Vec::new();
    let mut minor_tick_points = Vec::new();
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

    draw(params_proto, &major_tick_points);
    draw(
        LineParameters {
            width: params.width * 0.5,
            ..params_proto
        },
        &minor_tick_points,
    );
}
