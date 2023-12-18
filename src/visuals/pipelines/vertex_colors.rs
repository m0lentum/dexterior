use std::borrow::Cow;

use crate::visuals::render_window::RenderContext;

pub(crate) struct VertexColorsPipeline {
    pipeline: wgpu::RenderPipeline,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl VertexColorsPipeline {
    pub fn new(
        window: &crate::RenderWindow,
        mesh: &crate::SimplicialMesh<2>,
        res: &super::SharedResources,
    ) -> Self {
        let label = Some("vertex colors");

        let shader = window
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                    "../shaders/vertex_colors.wgsl"
                ))),
            });

        let pipeline_layout =
            window
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label,
                    bind_group_layouts: &[
                        &res.frame_bind_group_layout,
                        &res.data.bind_group_layout,
                    ],
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
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: window.multisample_state(),
                multiview: None,
            });

        // upload indices

        // for 0-cochains we set vertex colors to cochain values
        // and draw triangles, letting interpolation color the space between
        let indices: Vec<u16> = mesh.simplices[2]
            .indices
            .iter()
            .map(|i| *i as u16)
            .collect();

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
        let mut pass = ctx.pass("vertex colors");
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, res.vertex_buf.slice(..));
        pass.set_bind_group(0, &res.frame_bind_group, &[]);
        pass.set_bind_group(
            1,
            // this was uploaded by the method in `pipelines.rs` calling this
            &res.data.buf_and_bg.as_ref().unwrap().1,
            &[],
        );
        pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}
