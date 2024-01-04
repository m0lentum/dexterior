use bytemuck::{Pod, Zeroable};
use std::mem::size_of;

use nalgebra as na;

use crate::{
    camera::Camera,
    color_map::{self, ColorMap},
    render_window::RenderContext,
};

/// GPU resources (buffers, bind groups)
/// that are shared between multiple render pipelines.
pub(crate) struct SharedResources {
    /// Storage buffer (not vertex buffer!) for vertices, used for vertex pulling.
    pub vertex_buf: wgpu::Buffer,
    pub camera_uniform_buf: wgpu::Buffer,
    /// Single array texture containing all registered colormaps.
    /// These don't need to accessed, but also aren't allowed to be dropped,
    /// hence the underscores.
    pub _colormap_texture: wgpu::Texture,
    pub _colormap_view: wgpu::TextureView,
    pub _colormap_sampler: wgpu::Sampler,
    /// Bind group for things that are constant for a frame
    /// (camera, colormap textures).
    pub frame_bind_group: wgpu::BindGroup,
    pub frame_bind_group_layout: wgpu::BindGroupLayout,
    /// Buffer and bind group for a generic vector of data
    /// (at the time of writing, always a cochain)
    /// and color mapping parameters.
    pub data: DynamicDataBuffer,
}

/// Position of a vertex for the vertex buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Position {
    pos: [f32; 3],
}

impl From<na::Vector2<f64>> for Position {
    fn from(v: na::Vector2<f64>) -> Self {
        Self {
            pos: [v.x as f32, v.y as f32, 0.0],
        }
    }
}

/// Uniform buffer for the camera.
#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct CameraUniforms {
    view_proj: na::Matrix4<f32>,
    // camera basis vectors used for drawing billboards in the line renderer
    basis: na::Matrix3<f32>,
}

/// Storage buffer for generic vector data and color mapping parameters.
#[derive(Clone, Debug, encase::ShaderType)]
struct DataStorage<'a> {
    color_map_idx: u32,
    color_map_range_start: f32,
    color_map_range_length: f32,
    #[size(runtime)]
    data: &'a [f32],
}

/// A storage buffer for computed data that is automatically resized on write when necessary.
pub struct DynamicDataBuffer {
    pub cpu_storage: encase::StorageBuffer<Vec<u8>>,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub buf_and_bg: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    pub len: usize,
}

impl SharedResources {
    pub fn new(
        window: &crate::RenderWindow,
        mesh: &dexterior::SimplicialMesh<2>,
        colormaps: &[ColorMap],
    ) -> Self {
        let vertices: Vec<Position> = mesh.vertices.iter().cloned().map(Position::from).collect();

        use wgpu::util::DeviceExt;
        let vertex_buf = window
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("mesh vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let camera_uniform_buf_size = <CameraUniforms as encase::ShaderType>::min_size();
        let camera_uniform_buf = window.device.create_buffer(&wgpu::BufferDescriptor {
            size: camera_uniform_buf_size.get(),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label: Some("camera"),
            mapped_at_creation: false,
        });

        //
        // colormap textures
        //

        let colormap_size = wgpu::Extent3d {
            width: color_map::LUT_SIZE as u32,
            height: 1,
            depth_or_array_layers: colormaps.len() as u32,
        };
        let colormap_texture = window.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap"),
            size: colormap_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: color_map::TEX_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (map_idx, map) in colormaps.iter().enumerate() {
            window.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &colormap_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: map_idx as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&map.lut),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(colormap_size.width * size_of::<color_map::Color>() as u32),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: colormap_size.width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

        let colormap_view = colormap_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("colormap"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let colormap_sampler = window.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("colormap"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        //
        // frame bind group
        //

        let frame_bind_group_layout =
            window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: Some(camera_uniform_buf_size),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("frame"),
                });
        let frame_bind_group = window.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &frame_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&colormap_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&colormap_sampler),
                },
            ],
            label: Some("frame"),
        });

        //
        // data bind group
        //

        let data_bind_group_layout =
            window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("data"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            // parameters plus one element in the data buffer
                            min_binding_size: wgpu::BufferSize::new(4 * 3 + 4),
                        },
                        count: None,
                    }],
                });
        let data = DynamicDataBuffer {
            cpu_storage: encase::StorageBuffer::new(Vec::new()),
            bind_group_layout: data_bind_group_layout,
            buf_and_bg: None,
            len: 0,
        };

        Self {
            vertex_buf,
            camera_uniform_buf,
            _colormap_texture: colormap_texture,
            _colormap_view: colormap_view,
            _colormap_sampler: colormap_sampler,
            frame_bind_group_layout,
            frame_bind_group,
            data,
        }
    }

    pub fn vertex_buf_layout<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Position>() as _,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
            ],
        }
    }

    pub fn upload_frame_uniforms(&self, camera: &Camera, ctx: &mut RenderContext) {
        let uniforms = CameraUniforms {
            view_proj: camera.view_projection_matrix(),
            basis: camera.pose.isometry.rotation.to_rotation_matrix().into(),
        };
        let mut uniform_bytes = encase::UniformBuffer::new(Vec::new());
        uniform_bytes.write(&uniforms).unwrap();
        ctx.queue
            .write_buffer(&self.camera_uniform_buf, 0, &uniform_bytes.into_inner());
    }

    pub fn upload_data_buffer(
        &mut self,
        color_map_idx: usize,
        color_map_range: std::ops::Range<f32>,
        data: &[f32],
        ctx: &mut RenderContext,
    ) {
        assert!(!data.is_empty(), "Data must have at least one element");

        self.data.cpu_storage.as_mut().clear();
        self.data
            .cpu_storage
            .write(&DataStorage {
                color_map_idx: color_map_idx as u32,
                color_map_range_start: color_map_range.start,
                color_map_range_length: color_map_range.end - color_map_range.start,
                data,
            })
            .expect("Failed to write data");

        if data.len() > self.data.len || self.data.buf_and_bg.is_none() {
            // allocate a bigger buffer and create a new bind group
            use wgpu::util::DeviceExt;
            let buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("data"),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    contents: self.data.cpu_storage.as_ref(),
                });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("data"),
                layout: &self.data.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buf.as_entire_buffer_binding()),
                }],
            });

            self.data.buf_and_bg = Some((buf, bind_group));
            self.data.len = data.len();
        } else {
            // the previous condition made sure these exist
            let (buf, _) = self.data.buf_and_bg.as_ref().unwrap();
            ctx.queue
                .write_buffer(buf, 0, self.data.cpu_storage.as_ref());
        }
    }
}
