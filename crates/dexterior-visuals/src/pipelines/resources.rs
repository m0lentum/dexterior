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
    pub color_data: Vec<ColorMappedData>,
    pub colormap_params_bind_group_layout: wgpu::BindGroupLayout,
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

/// Uniform buffer for color mapping parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorMapUniforms {
    layer_idx: u32,
    range_start: f32,
    range_length: f32,
    // pad to 16 bytes for webgl compatibility
    _pad: u32,
}

pub(crate) struct ColorMappedData {
    /// Buffer of values, generally consumed as a vertex buffer.
    pub data_buf: wgpu::Buffer,
    /// Number of values currently stored in the data buffer.
    pub data_len: usize,
    /// Number of values that can fit in the data buffer.
    pub data_capacity: usize,
    /// Buffer containing a ColorMapUniforms.
    pub uniform_buf: wgpu::Buffer,
    /// Bind group binding the uniform buffer.
    pub bind_group: wgpu::BindGroup,
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
        // colormap
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
            min_filter: wgpu::FilterMode::Linear,
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
        // colormap parameters
        //

        let colormap_params_bind_group_layout =
            window
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("colormap parameters"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                size_of::<ColorMapUniforms>() as u64
                            ),
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
            colormap_params_bind_group_layout,
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
        values: &[f32],
    ) {
        assert!(!values.is_empty(), "Data must have at least one element");

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

        let colormap_unif = ColorMapUniforms {
            layer_idx: state.color_map_idx as u32,
            range_start: color_map_range.start,
            range_length: color_map_range.end - color_map_range.start,
            _pad: 0,
        };

        // method to create a new data buffer if necessary
        // (either making a new one or old doesn't have enough space)
        use wgpu::util::DeviceExt;
        let reallocate = || -> wgpu::Buffer {
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("color mapped data"),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    contents: bytemuck::cast_slice(values),
                })
        };

        if state.next_color_data >= self.color_data.len() {
            // more draw calls in a frame than have been made before, create a new buffer

            let data_buf = reallocate();

            let uniform_buf = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("colormap parameters"),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    contents: bytemuck::bytes_of(&colormap_unif),
                });

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("colormap parameters"),
                layout: &self.colormap_params_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                }],
            });
            self.color_data.push(ColorMappedData {
                data_buf,
                data_len: values.len(),
                data_capacity: values.len(),
                uniform_buf,
                bind_group,
            });
        } else {
            // reusing an existing buffer, resizing if necessary

            let color_data = &mut self.color_data[state.next_color_data];

            if values.len() > color_data.data_capacity {
                color_data.data_buf = reallocate();
                color_data.data_capacity = values.len();
            } else {
                ctx.queue
                    .write_buffer(&color_data.data_buf, 0, bytemuck::cast_slice(values));
            }
            color_data.data_len = values.len();

            ctx.queue.write_buffer(
                &color_data.uniform_buf,
                0,
                bytemuck::bytes_of(&colormap_unif),
            );
        }

        state.next_color_data += 1;
    }

    /// Get the last uploaded color data.
    ///
    /// This will panic if no data has been uploaded.
    #[inline]
    pub fn latest_colored_data(&self, state: &super::RendererState) -> &ColorMappedData {
        &self.color_data[state.next_color_data - 1]
    }
}
