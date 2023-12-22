use std::borrow::Cow;

use crate::render_window::RenderContext;

pub(crate) struct WireframePipeline {
    pipeline: wgpu::RenderPipeline,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl WireframePipeline {
    pub fn new(
        window: &crate::RenderWindow,
        mesh: &dexterior::SimplicialMesh<2>,
        res: &super::SharedResources,
    ) -> Self {
        let label = Some("wireframe");

        let shader = window
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/wireframe.wgsl"
                ))),
            });

        let pipeline_layout =
            window
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label,
                    bind_group_layouts: &[&res.frame_bind_group_layout],
                    push_constant_ranges: &[],
                });
        let pipeline = window
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label,
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[super::SharedResources::vertex_buf_layout()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(window.swapchain_format().into())],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: window.multisample_state(),
                multiview: None,
            });

        // upload indices

        // 1-simplices are the line segments, so those contain the indices needed
        // for drawing the wireframe as a line list
        let indices: Vec<u16> = mesh.indices::<1>().flatten().map(|i| *i as u16).collect();

        use wgpu::util::DeviceExt;
        let index_buf = window
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        Self {
            pipeline,
            index_buf,
            index_count: indices.len() as u32,
        }
    }

    pub fn draw(&self, res: &super::SharedResources, ctx: &mut RenderContext) {
        let mut pass = ctx.pass("wireframe");
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, res.vertex_buf.slice(..));
        pass.set_bind_group(0, &res.frame_bind_group, &[]);
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
