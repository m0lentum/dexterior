use nalgebra as na;
use std::borrow::Cow;

use crate::render_window::RenderContext;

//
// user-facing parameters
//

/// Parameters for rendering operations that draw lines.
#[derive(Clone, Copy, Debug)]
pub struct LineParameters {
    /// Width of the line, either in pixels or in world units.
    /// Default: 1 screenspace pixel.
    pub width: LineWidth,
    /// Color of the line in linear sRGB space.
    /// Default: `palette::named::BLACK.into()`.
    pub color: palette::LinSrgb,
}

impl Default for LineParameters {
    fn default() -> Self {
        Self {
            width: LineWidth::ScreenPixels(1.0),
            color: palette::named::BLACK.into(),
        }
    }
}

/// The width of a line and which space it's defined in.
#[derive(Clone, Copy, Debug)]
pub enum LineWidth {
    /// Keep a constant width in screen space regardless of zoom level and distance.
    ScreenPixels(f32),
    /// Set the width in world space.
    WorldUnits(f32),
}

/// How to interpret point data given to the line renderer.
#[derive(Clone, Copy, Debug)]
pub enum LineDrawingMode {
    /// Every two points are a separate line segment with a gap between them.
    List,
    /// Every point is connected together with a line segment.
    Strip,
}

//
// renderer
//

/// A versatile instanced line renderer.
///
/// Based on [this blog post by Rye Terrell].
/// (https://wwwtyro.net/2019/11/18/instanced-lines.html)
pub(crate) struct LinePipeline {
    // pipelines for different instance step modes
    list_pipeline: wgpu::RenderPipeline,
    strip_pipeline: wgpu::RenderPipeline,
    point_pipeline: wgpu::RenderPipeline,
    // geometry for segments and joins
    primitives: Primitives,
    // list of instance buffers and uniform bind groups,
    // one for each set of lines drawn within a frame,
    // to be able to draw all of them
    // without submitting commands for buffer writes between each draw
    instance_bufs: Vec<DynamicInstanceBuffer>,
    params_bind_group_layout: wgpu::BindGroupLayout,
    // keep track of how many draw calls have been made this frame
    // to decide which instance buffer to use
    next_draw_index: usize,
}

/// A collection of all instance primitives we need
/// for line segments, caps, and joins.
struct Primitives {
    segment: InstanceGeometry,
    circle_join: InstanceGeometry,
}

/// Uniform parameters for the shaders.
#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct ParamUniforms {
    // scaling in screenspace or worldspace
    // (corresponds to `ScalingMode` below,
    // but encase doesn't understand enums)
    placement_mode: u32,
    width: f32,
    // note: this can't be a [f32; 4]
    // because encase will interpret it as a shader-side array
    color: na::Vector4<f32>,
}

/// Line scaling in screenspace or worldspace
#[derive(Clone, Copy)]
enum ScalingMode {
    WorldSpace = 0,
    ScreenSpace = 1,
}

/// A vertex in the instance geometry.
type Vertex = [f32; 2];

impl LinePipeline {
    pub fn new(window: &crate::RenderWindow, res: &super::SharedResources) -> Self {
        let label = Some("line");

        let segment_shader = window
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/line_segment.wgsl"
                ))),
            });

        let circle_join_shader = window
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/line_join_unoriented.wgsl"
                ))),
            });

        // uniforms

        let params_buf_size = <ParamUniforms as encase::ShaderType>::min_size();
        let params_bind_group_layout =
            window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label,
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Some(params_buf_size),
                        },
                        count: None,
                    }],
                });

        // pipeline

        let pipeline_layout =
            window
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label,
                    bind_group_layouts: &[&res.frame_bind_group_layout, &params_bind_group_layout],
                    push_constant_ranges: &[],
                });

        /// Different rates of stepping through the instance buffer and different shaders
        /// are needed for different parts of drawing the lines.
        /// A line list needs to step twice more per instance
        /// than a line strip, and circle joins just need an endpoint per instance.
        enum InstanceStepMode {
            LineList,
            LineStrip,
            Points,
        }

        let pipeline = |mode: InstanceStepMode| {
            use InstanceStepMode::*;
            let array_stride = match mode {
                // two points per step for line lists
                LineList => 2 * 3 * 4,
                // just one point for the rest,
                // since strips treat each point as both a start and an end
                LineStrip | Points => 3 * 4,
            };
            let attributes = match mode {
                LineList | LineStrip => [
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 3 * 4,
                        shader_location: 2,
                    },
                ]
                .as_slice(),
                Points => [wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 1,
                }]
                .as_slice(),
            };
            let module = match mode {
                LineList | LineStrip => &segment_shader,
                Points => &circle_join_shader,
            };

            window
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label,
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: "vs_main",
                        buffers: &[
                            // vertices of a single line segment instance
                            wgpu::VertexBufferLayout {
                                // two floats for (x,y) coordinates
                                array_stride: 2 * 4,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &[
                                    // position
                                    wgpu::VertexAttribute {
                                        format: wgpu::VertexFormat::Float32x2,
                                        offset: 0,
                                        shader_location: 0,
                                    },
                                ],
                            },
                            // index buffer containing start and end points of line segments
                            wgpu::VertexBufferLayout {
                                array_stride,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes,
                            },
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &segment_shader,
                        entry_point: "fs_main",
                        targets: &[Some(window.swapchain_format().into())],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: window.multisample_state(),
                    multiview: None,
                })
        };

        Self {
            list_pipeline: pipeline(InstanceStepMode::LineList),
            strip_pipeline: pipeline(InstanceStepMode::LineStrip),
            point_pipeline: pipeline(InstanceStepMode::Points),
            primitives: Self::generate_instance_geometry(&window.device),
            instance_bufs: Vec::new(),
            params_bind_group_layout,
            next_draw_index: 0,
        }
    }

    /// Generate vertex and index buffers for line segments, joins and caps.
    fn generate_instance_geometry(device: &wgpu::Device) -> Primitives {
        // a line segment is just a rectangle
        // with coordinates chosen so that it's easy to transform
        // using the difference between two points and a thickness value
        let segment = InstanceGeometry::upload(
            device,
            "line segment",
            &[[0., 0.5], [0., -0.5], [1., -0.5], [1., 0.5]],
            &[0, 1, 2, 0, 2, 3],
        );

        // TODO: the resolution of the circle should vary
        // depending on line thickness.
        // how should we do that?
        let circle_vert_count = 16;
        let angle_increment = std::f32::consts::TAU / circle_vert_count as f32;
        let circle_verts: Vec<Vertex> = (0..circle_vert_count)
            .map(|i| {
                let angle = i as f32 * angle_increment;
                [0.5 * f32::cos(angle), 0.5 * f32::sin(angle)]
            })
            .collect();
        let circle_indices: Vec<u16> = (1..circle_vert_count - 1)
            .flat_map(|i| [0, i, i + 1])
            .collect();
        let circle_join =
            InstanceGeometry::upload(device, "circle join", &circle_verts, &circle_indices);

        Primitives {
            segment,
            circle_join,
        }
    }

    /// Upload an instance buffer containing segment endpoints to the GPU,
    /// returning the buffer that was written to.
    fn upload_draw_data(
        &mut self,
        ctx: &mut RenderContext,
        params: ParamUniforms,
        points: &[[f32; 3]],
    ) {
        let mut params_bytes = encase::UniformBuffer::new(Vec::new());
        params_bytes.write(&params).unwrap();

        if self.next_draw_index >= self.instance_bufs.len() {
            // this is more than we've drawn in a frame before,
            // add another instance buffer
            self.instance_bufs.push(DynamicInstanceBuffer::write_new(
                ctx.device,
                &self.params_bind_group_layout,
                &params_bytes.into_inner(),
                bytemuck::cast_slice(points),
            ));
        } else {
            self.instance_bufs[self.next_draw_index].write(
                ctx,
                &params_bytes.into_inner(),
                bytemuck::cast_slice(points),
            );
        }
    }

    /// Reset internal state between frames.
    pub fn end_frame(&mut self) {
        self.next_draw_index = 0;
    }

    pub fn draw(
        &mut self,
        res: &super::SharedResources,
        ctx: &mut RenderContext,
        params: LineParameters,
        mode: LineDrawingMode,
        points: &[[f32; 3]],
    ) {
        // upload line segments and uniforms

        let params_unif = ParamUniforms {
            placement_mode: match params.width {
                LineWidth::WorldUnits(_) => ScalingMode::WorldSpace as u32,
                LineWidth::ScreenPixels(_) => ScalingMode::ScreenSpace as u32,
            },
            width: match params.width {
                LineWidth::WorldUnits(w) => w,
                LineWidth::ScreenPixels(p) => p,
            },
            color: na::Vector4::new(params.color.red, params.color.green, params.color.blue, 1.0),
        };
        self.upload_draw_data(ctx, params_unif, points);

        let instance = &self.instance_bufs[self.next_draw_index];

        // setup a render pass

        let mut pass = ctx.pass("lines");
        pass.set_bind_group(0, &res.frame_bind_group, &[]);
        pass.set_bind_group(1, &instance.params_bind_group, &[]);

        // draw segments

        let segment_count = match mode {
            LineDrawingMode::List => points.len() / 2,
            LineDrawingMode::Strip => points.len() - 1,
        };

        pass.set_pipeline(match mode {
            LineDrawingMode::List => &self.list_pipeline,
            LineDrawingMode::Strip => &self.strip_pipeline,
        });
        pass.set_vertex_buffer(0, self.primitives.segment.vertex_buf.slice(..));
        pass.set_index_buffer(
            self.primitives.segment.index_buf.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.set_vertex_buffer(1, instance.segment_buf.slice(..));

        pass.draw_indexed(
            0..self.primitives.segment.index_count,
            0,
            0..segment_count as _,
        );

        // draw circle joins and caps
        // TODO: allow other types of cap (joins are probably fine to be always circles)

        let point_count = points.len();

        pass.set_pipeline(&self.point_pipeline);
        pass.set_vertex_buffer(0, self.primitives.circle_join.vertex_buf.slice(..));
        pass.set_index_buffer(
            self.primitives.circle_join.index_buf.slice(..),
            wgpu::IndexFormat::Uint16,
        );
        pass.draw_indexed(
            0..self.primitives.circle_join.index_count,
            0,
            0..point_count as _,
        );

        self.next_draw_index += 1;
    }
}

//
// utility types
//

/// Vertex and index buffer to hold a mesh instance.
struct InstanceGeometry {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl InstanceGeometry {
    /// Upload vertices and indices to the GPU.
    fn upload(device: &wgpu::Device, label: &str, vertices: &[Vertex], indices: &[u16]) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        }
    }
}

/// An instance buffer that is reallocated if it doesn't have enough capacity.
/// We keep a list of these for drawing multiple lines
/// without pausing to submit commands in between.
struct DynamicInstanceBuffer {
    segment_buf: wgpu::Buffer,
    capacity: usize,
    params_buf: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
}

impl DynamicInstanceBuffer {
    /// Write instance data to a new buffer.
    fn write_new(
        device: &wgpu::Device,
        params_bg_layout: &wgpu::BindGroupLayout,
        params_data: &[u8],
        segment_data: &[u8],
    ) -> Self {
        use wgpu::util::DeviceExt;
        let segment_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line segments"),
            contents: segment_data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let capacity = segment_data.len();

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line parameters"),
            contents: params_data,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("line parameters"),
            layout: params_bg_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });

        Self {
            segment_buf,
            capacity,
            params_buf,
            params_bind_group,
        }
    }

    /// Write instance data to an existing buffer, reallocating if necessary.
    fn write(&mut self, ctx: &mut RenderContext, params_data: &[u8], segment_data: &[u8]) {
        if segment_data.len() > self.capacity {
            // not enough capacity, reallocate
            use wgpu::util::DeviceExt;
            self.segment_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("line segments"),
                    contents: segment_data,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
            self.capacity = segment_data.len();
        } else {
            ctx.queue.write_buffer(&self.segment_buf, 0, segment_data);
        }

        ctx.queue.write_buffer(&self.params_buf, 0, params_data);
    }
}
