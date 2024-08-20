use std::mem::size_of;

use nalgebra as na;

use crate::{
    camera::Camera,
    color_map::{self, ColorMap},
    render_window::{ActiveRenderWindow, RenderContext},
};

/// GPU resources (buffers, bind groups)
/// that are shared between multiple render pipelines.
pub(crate) struct SharedResources {
    pub camera_uniform_buf: wgpu::Buffer,
    /// Single array texture containing all registered colormaps.
    /// These don't need to accessed, but also aren't allowed to be dropped,
    /// hence the underscores.
    _colormap_texture: wgpu::Texture,
    _colormap_view: wgpu::TextureView,
    _colormap_sampler: wgpu::Sampler,
    /// Bind group for things that are constant for a frame
    /// (camera, colormap textures).
    pub frame_bind_group: wgpu::BindGroup,
    pub frame_bind_group_layout: wgpu::BindGroupLayout,
    /// Buffers and bind groups for color mapping parameters and data.
    /// In a Vec because each draw call in a frame needs its own buffer
    /// to avoid overwriting others.
    /// The state determining which buffer to use lives in [`super::RendererState`].
    pub color_data: Vec<DynamicDataBuffer>,
    pub color_data_bind_group_layout: wgpu::BindGroupLayout,
}

/// Uniform buffer for the camera.
#[derive(Clone, Copy, Debug, encase::ShaderType)]
struct CameraUniforms {
    view_proj: na::Matrix4<f32>,
    // camera basis vectors used for drawing billboards in the line renderer
    basis: na::Matrix3<f32>,
    // viewport resolution used for scaling things to pixels
    resolution: na::Vector2<f32>,
}

/// Storage buffer for generic vector data and color mapping parameters.
#[derive(Clone, Debug, encase::ShaderType)]
struct DataStorage<'a> {
    color_map_idx: u32,
    color_map_range_start: f32,
    color_map_range_length: f32,
    #[size(runtime)]
    values: &'a [f32],
}

/// A storage buffer for computed data that is automatically resized on write when necessary.
pub struct DynamicDataBuffer {
    pub cpu_storage: encase::StorageBuffer<Vec<u8>>,
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub len: usize,
}

impl SharedResources {
    pub fn new(window: &ActiveRenderWindow, colormaps: &[ColorMap]) -> Self {
        //
        // camera
        //

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
        // color data bind group
        //

        let color_data_bind_group_layout =
            window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("color data"),
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

        Self {
            camera_uniform_buf,
            _colormap_texture: colormap_texture,
            _colormap_view: colormap_view,
            _colormap_sampler: colormap_sampler,
            frame_bind_group_layout,
            frame_bind_group,
            color_data: Vec::new(),
            color_data_bind_group_layout,
        }
    }

    pub fn upload_frame_uniforms(&self, camera: &Camera, ctx: &mut RenderContext) {
        let uniforms = CameraUniforms {
            view_proj: camera.view_projection_matrix(ctx.viewport_size),
            basis: camera.pose.isometry.rotation.to_rotation_matrix().into(),
            resolution: na::Vector2::new(ctx.viewport_size.0 as f32, ctx.viewport_size.1 as f32),
        };
        let mut uniform_bytes = encase::UniformBuffer::new(Vec::new());
        uniform_bytes.write(&uniforms).unwrap();
        ctx.queue
            .write_buffer(&self.camera_uniform_buf, 0, &uniform_bytes.into_inner());
    }

    /// Send color mapping data to the GPU.
    pub fn upload_color_data(
        &mut self,
        ctx: &mut RenderContext,
        state: &mut super::RendererState,
        values: &[f64],
    ) {
        assert!(!values.is_empty(), "Data must have at least one element");

        // convert values to f32 (given as f64 because simulation variables are f64)
        let values: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        // compute color map range from the data if it hasn't been given as a parameter
        let color_map_range = if let Some(r) = &state.color_map_range {
            r.clone()
        } else {
            use itertools::{Itertools, MinMaxResult::*};
            match values.iter().minmax() {
                NoElements | OneElement(_) => -1.0..1.0,
                MinMax(&l, &u) => l..u,
            }
        };

        let gpu_data = DataStorage {
            color_map_idx: state.color_map_idx as u32,
            color_map_range_start: color_map_range.start,
            color_map_range_length: color_map_range.end - color_map_range.start,
            values: &values,
        };

        // method to create a new storage buffer if necessary
        // (either making a new one or old doesn't have enough space)
        let reallocate = |contents: &[u8]| -> (wgpu::Buffer, wgpu::BindGroup) {
            use wgpu::util::DeviceExt;
            let buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("color data"),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    contents,
                });
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("color data"),
                layout: &self.color_data_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(buffer.as_entire_buffer_binding()),
                }],
            });
            (buffer, bind_group)
        };

        if state.next_color_data >= self.color_data.len() {
            // more draw calls in a frame than have been made before, create a new buffer

            let mut cpu_storage = encase::StorageBuffer::new(Vec::new());
            cpu_storage.write(&gpu_data).unwrap();
            let (buffer, bind_group) = reallocate(cpu_storage.as_ref());
            self.color_data.push(DynamicDataBuffer {
                cpu_storage,
                buffer,
                bind_group,
                len: 0,
            });
        } else {
            // reusing an existing buffer, resizing if necessary

            let color_data = &mut self.color_data[state.next_color_data];

            color_data.cpu_storage.as_mut().clear();
            color_data.cpu_storage.write(&gpu_data).unwrap();

            if values.len() > color_data.len {
                // allocate a bigger buffer and create a new bind group

                let (buffer, bind_group) = reallocate(color_data.cpu_storage.as_ref());
                color_data.buffer = buffer;
                color_data.bind_group = bind_group;
                color_data.len = values.len();
            } else {
                ctx.queue
                    .write_buffer(&color_data.buffer, 0, color_data.cpu_storage.as_ref());
            }
        }

        state.next_color_data += 1;
    }

    /// Get the bind group for the last uploaded color data.
    ///
    /// This will panic if no data has been uploaded.
    #[inline]
    pub fn latest_color_bind_group(&self, state: &super::RendererState) -> &wgpu::BindGroup {
        &self.color_data[state.next_color_data - 1].bind_group
    }
}
