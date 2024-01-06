//! Low-level resources for window creation and rendering.

use winit::{
    event::{Event, VirtualKeyCode, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use nalgebra as na;

use super::{camera::Camera, pipelines as pl};

/// An error that occurred when creating a [`RenderWindow`].
#[derive(thiserror::Error, Debug)]
pub enum WindowInitError {
    /// Failed to create the [`winit`] window.
    #[error("Failed to create winit window")]
    WindowOsError(#[from] winit::error::OsError),
    /// Failed to create a [`wgpu::Surface`] for the window.
    #[error("Failed to create wgpu surface")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    /// Failed to get a [`wgpu::Adapter`].
    #[error("Failed to get wgpu adapter")]
    CreateAdapterError,
    /// Failed to get a [`wgpu::Device`].
    #[error("Failed to get wgpu device")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

/// Parameters for the creation of a [`RenderWindow`].
#[derive(Clone, Copy, Debug)]
pub struct WindowParams {
    /// Initial width of the window in pixels. Default: 800.
    pub width: usize,
    /// Initial height of the window in pixels. Default: 800.
    pub height: usize,
    /// Samples used for anti-aliasing. Default: 4.
    pub msaa_samples: u32,
}

impl Default for WindowParams {
    fn default() -> Self {
        Self {
            width: 800,
            height: 800,
            msaa_samples: 4,
        }
    }
}

/// A window for drawing real-time graphics.
///
/// See [`run_animation`][Self::run_animation], [`Animation`][crate::Animation],
/// and the crate examples for how to draw into the window once created.
pub struct RenderWindow {
    // window needs to be kept around for it to not close,
    // but doesn't need to be accessed
    _window: Window,
    // event loop in an option for convenience:
    // we'll `take` it out to run the loop and still retain access to `&self`
    // (otherwise we'd have to partial borrow everything while we're in the event loop)
    event_loop: Option<EventLoop<()>>,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    surface: wgpu::Surface,
    surface_config: wgpu::SurfaceConfiguration,
    swapchain_format: wgpu::TextureFormat,
    msaa_samples: u32,
    msaa_texture: wgpu::Texture,
}

impl RenderWindow {
    /// Create a new render window.
    pub fn new(params: WindowParams) -> Result<Self, WindowInitError> {
        futures::executor::block_on(Self::build_async(params))
    }

    /// Inner async method to create the window and wgpu context.
    /// Async is needed for the wgpu init,
    /// but I don't want users to need to use it.
    /// (Maybe this could be exposed for users who already run in async contexts though?)
    async fn build_async(params: WindowParams) -> Result<Self, WindowInitError> {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("dexterior")
            .with_inner_size(winit::dpi::LogicalSize {
                width: params.width as f64,
                height: params.height as f64,
            })
            .build(&event_loop)?;

        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface(&window)? };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or(WindowInitError::CreateAdapterError)?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await?;

        let window_size = window.inner_size();

        let swapchain_format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let swapchain_capabilities = surface.get_capabilities(&adapter);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_format,
            width: window_size.width,
            height: window_size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: swapchain_capabilities.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let msaa_texture =
            Self::create_msaa_texture(&device, swapchain_format, params.msaa_samples, window_size);

        Ok(Self {
            _window: window,
            event_loop: Some(event_loop),
            device,
            queue,
            surface,
            surface_config,
            swapchain_format,
            msaa_samples: params.msaa_samples,
            msaa_texture,
        })
    }

    /// Create a multisampled texture to render to.
    fn create_msaa_texture(
        device: &wgpu::Device,
        swapchain_format: wgpu::TextureFormat,
        msaa_samples: u32,
        window_size: winit::dpi::PhysicalSize<u32>,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screen multisample"),
            size: wgpu::Extent3d {
                width: window_size.width,
                height: window_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: msaa_samples,
            dimension: wgpu::TextureDimension::D2,
            format: swapchain_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    }

    /// Reconfigure the swapchain and recreate the MSAA texture when the window size has changed.
    fn resize_swapchain(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size == self.window_size() {
            return;
        }
        self.surface_config.width = new_size.width;
        self.surface_config.height = new_size.height;
        self.surface.configure(&self.device, &self.surface_config);
        self.msaa_texture = Self::create_msaa_texture(
            &self.device,
            self.swapchain_format,
            self.msaa_samples,
            new_size,
        );
    }

    /// Get the format of the swapchain texture being rendered to.
    #[inline]
    pub(crate) fn swapchain_format(&self) -> wgpu::TextureFormat {
        self.swapchain_format
    }

    /// Get the size of the render window in physical pixels.
    #[inline]
    pub(crate) fn window_size(&self) -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }

    /// Get the multisample state used by the window.
    #[inline]
    pub(crate) fn multisample_state(&self) -> wgpu::MultisampleState {
        wgpu::MultisampleState {
            count: self.msaa_samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }

    /// Grab the next swapchain texture and start drawing on it.
    fn begin_frame(&mut self) -> RenderContext<'_> {
        let surface_tex = self
            .surface
            .get_current_texture()
            .expect("Failed to get next swapchain texture");
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let msaa_view = self
            .msaa_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        RenderContext {
            clear_color: Some(wgpu::Color::WHITE),
            surface_tex,
            target: msaa_view,
            resolve_target: surface_view,
            encoder,
            device: &self.device,
            queue: &mut self.queue,
            viewport_size: (self.surface_config.width, self.surface_config.height),
        }
    }

    /// Play an [`Animation`][crate::Animation] in the window.
    ///
    /// # Controls
    /// - `Q`: end the animation and return from this function
    /// - `N`: swap to the next color map (only works if
    ///   [`Painter::set_color_map`][pl::Painter::set_color_map] is not called by the animation)
    pub fn run_animation<StepFn>(&mut self, mut anim: super::animation::Animation<StepFn>)
    where
        StepFn: FnMut(&mut crate::Painter),
    {
        let mut renderer = pl::Renderer::new(self, anim.mesh, &anim.params);

        // get the (x, y) bounds of the mesh to construct a camera
        // with a fitting viewport
        // (3D cameras will need different construction once I get around to that,
        // think about how to abstract this.
        // also it could be fun to allow looking at a 2D mesh with a 3D camera)
        let b = anim.mesh.bounds();
        let camera = Camera::new_2d(
            na::Vector2::new(b.min.x as f32, b.min.y as f32),
            na::Vector2::new(b.max.x as f32, b.max.y as f32),
            1.0,
        );

        // take the event loop out of `self` to be able to call methods on `self` inside the loop
        let mut event_loop = self.event_loop.take().unwrap();
        use winit::platform::run_return::EventLoopExtRunReturn;
        event_loop.run_return(|event, _, control_flow| {
            control_flow.set_poll();
            match event {
                Event::MainEventsCleared => {
                    // TODO: timing control

                    let mut ctx = self.begin_frame();
                    renderer.resources.upload_frame_uniforms(&camera, &mut ctx);

                    let mut painter = pl::Painter {
                        ctx: &mut ctx,
                        rend: &mut renderer,
                        mesh: anim.mesh,
                        camera: &camera,
                    };
                    (anim.step)(&mut painter);

                    ctx.queue.submit(Some(ctx.encoder.finish()));
                    renderer.end_frame();
                    ctx.surface_tex.present();
                }
                Event::WindowEvent { event, .. } => {
                    // TODO: camera controls

                    match event {
                        WindowEvent::CloseRequested => {
                            control_flow.set_exit();
                        }
                        WindowEvent::Resized(new_size) => {
                            self.resize_swapchain(new_size);
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            if input.state == winit::event::ElementState::Pressed {
                                // key mapping similar to matplotlib where applicable
                                match input.virtual_keycode {
                                    Some(VirtualKeyCode::Q) => {
                                        control_flow.set_exit();
                                    }
                                    Some(VirtualKeyCode::N) => {
                                        renderer.cycle_color_maps();
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        });
        self.event_loop = Some(event_loop);
    }
}

pub(crate) struct RenderContext<'a> {
    // if this is set, first pass automatically clears the framebuffer
    clear_color: Option<wgpu::Color>,
    surface_tex: wgpu::SurfaceTexture,
    pub target: wgpu::TextureView,
    pub resolve_target: wgpu::TextureView,
    pub encoder: wgpu::CommandEncoder,
    pub device: &'a wgpu::Device,
    pub queue: &'a mut wgpu::Queue,
    pub viewport_size: (u32, u32),
}

impl<'a> RenderContext<'a> {
    /// Start a render pass with default parameters.
    pub fn pass(&mut self, label: &str) -> wgpu::RenderPass {
        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target,
                resolve_target: Some(&self.resolve_target),
                ops: wgpu::Operations {
                    load: if let Some(c) = self.clear_color.take() {
                        wgpu::LoadOp::Clear(c)
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        })
    }
}
