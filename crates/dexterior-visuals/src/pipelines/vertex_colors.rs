use crate::render_window::RenderContext;
use dexterior_core as dex;

/// there are two variants of this pipeline,
/// one for drawing 0-cochains as vertex colors
/// and another for drawing 2-cochains as flat triangles
pub(crate) enum VertexColorVariant {
    InterpolatedVertices,
    FlatTriangles,
}

pub(crate) struct VertexColorsPipeline {
    variant: VertexColorVariant,
    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl VertexColorsPipeline {
    pub fn new(
        ctx: &super::RenderContext,
        mesh: &dex::SimplicialMesh<2>,
        res: &super::SharedResources,
        variant: VertexColorVariant,
    ) -> Self {
        let label = Some("vertex colors");

        let shader = ctx
            .device
            .create_shader_module(wgpu::include_wgsl!("../shaders/vertex_colors.wgsl"));

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label,
                bind_group_layouts: &[
                    &res.frame_bind_group_layout,
                    &res.colormap_params_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[
                        wgpu::VertexBufferLayout {
                            // 3D position vectors
                            array_stride: 4 * 3,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            }],
                        },
                        // cochain values in a secondary vertex buffer
                        // so positions don't need reuploading
                        wgpu::VertexBufferLayout {
                            array_stride: 4,
                            step_mode: wgpu::VertexStepMode::Vertex,
                            attributes: &[wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32,
                                offset: 0,
                                shader_location: 1,
                            }],
                        },
                    ],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(ctx.target_format.into())],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: ctx.multisample_state,
                multiview: None,
                cache: None,
            });

        // upload vertices and indices

        let (vertices, indices): (Vec<[f32; 3]>, Vec<u16>) = match variant {
            // when drawing 0-cochains, draw as a typical triangle mesh
            VertexColorVariant::InterpolatedVertices => (
                mesh.vertices
                    .iter()
                    .cloned()
                    .map(|v| [v.x as f32, v.y as f32, 0.])
                    .collect(),
                mesh.indices::<2>().flatten().map(|i| *i as u16).collect(),
            ),
            // when drawing 2-cochains, vertices need to be duplicated
            // to draw the same flat color over a whole triangle
            VertexColorVariant::FlatTriangles => (
                mesh.simplices::<2>()
                    .flat_map(|s| s.vertices().collect::<Vec<_>>())
                    .map(|v| [v.x as f32, v.y as f32, 0.])
                    .collect(),
                // technically we wouldn't need an index buffer at all here,
                // but having one lets us use the same drawing method for both cases
                (0..3 * mesh.simplex_count::<2>() as u16).collect(),
            ),
        };

        use wgpu::util::DeviceExt;
        let vertex_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        Self {
            variant,
            pipeline,
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        }
    }

    pub fn draw(
        &self,
        res: &mut super::SharedResources,
        ctx: &mut RenderContext,
        state: &mut super::RendererState,
        data: &[f64],
    ) {
        // we need to copy the data values for each vertex of a flat triangle,
        // since these values are stored in a vertex buffer
        // and it's not possible to only have it increment once every 3 vertices
        let duplicate_count = match self.variant {
            VertexColorVariant::InterpolatedVertices => 1,
            VertexColorVariant::FlatTriangles => 3,
        };
        let data: Vec<f32> = data
            .iter()
            .flat_map(|v| std::iter::repeat(*v as f32).take(duplicate_count))
            .collect();

        res.upload_color_data(ctx, state, &data);
        let color_data = res.latest_colored_data(state);

        let mut pass = ctx.pass("vertex colors");

        pass.set_pipeline(&self.pipeline);

        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, color_data.data_buf.slice(..));
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);

        pass.set_bind_group(0, &res.frame_bind_group, &[]);
        pass.set_bind_group(1, &color_data.bind_group, &[]);

        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
