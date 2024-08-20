//! Low-level resources for window creation and rendering.

use std::time::Instant;

use winit::{
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use nalgebra as na;

use super::{camera::Camera, pipelines as pl};

/// An error that occurred when creating a [`RenderWindow`].
///
/// Unfortunately with the current winit version
/// the window has to be created in a trait method from which
/// this can't be propagated to the user, hence why it's private now
/// (still used internally to make use of the `?` operator).
#[derive(thiserror::Error, Debug)]
#[allow(clippy::enum_variant_names)]
enum WindowInitError {
    /// Failed to create [`winit`] event loop.
    #[error("Failed to create winit event loop")]
    EventLoopError(#[from] winit::error::EventLoopError),
    /// Failed to create the [`winit`] window.
    #[error("Failed to create winit window")]
    WindowOsError(#[from] winit::error::OsError),
    /// Failed to get a window handle.
    #[error("Failed to get a window handle")]
    HandleError,
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

//
// user-facing API
//

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
/// and the examples in the [dexterior repo](https://github.com/m0lentum/dexterior)
/// for how to draw into the window once created.
pub struct RenderWindow {
    // RenderWindow is just a wrapper to implement winit's `ApplicationHandler` on,
    // all the actual resources are created on application resume
    // and stored in `ActiveRenderWindow`
    params: WindowParams,
    // event loop in an option because we need to take it out to run it
    event_loop: Option<EventLoop<()>>,
}

impl RenderWindow {
    /// Create a new render window.
    pub fn new(params: WindowParams) -> Result<Self, winit::error::EventLoopError> {
        Ok(Self {
            params,
            event_loop: Some(EventLoop::new()?),
        })
    }

    /// Play an [`Animation`][crate::Animation] in the window.
    ///
    /// # Controls
    /// - `Q`: end the animation and return from this function
    /// - `N`: swap to the next color map (only works if
    ///   [`Painter::set_color_map`][pl::Painter::set_color_map] is not called by the animation)
    ///
    /// # Panics
    ///
    /// Due to architectural limitations in the current version of `winit`,
    /// we cannot propagate errors that occurred in window or render context creation.
    /// If these things fail, this function will panic.
    pub fn run_animation<State, StepFn, DrawFn>(
        &mut self,
        anim: super::animation::Animation<State, StepFn, DrawFn>,
    ) -> Result<(), winit::error::EventLoopError>
    where
        State: crate::AnimationState,
        StepFn: FnMut(&mut State),
        DrawFn: FnMut(&State, &mut crate::Painter),
    {
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

        let mut anim_app = AnimationApp {
            window_params: self.params,
            window: None,
            camera,
            frame_start_t: Instant::now(),
            time_in_frame: 0.,
            prev_state: anim.state.clone(),
            next_state: anim.state.clone(),
            anim,
        };

        let mut event_loop = self.event_loop.take().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
        event_loop.run_app_on_demand(&mut anim_app)?;

        self.event_loop = Some(event_loop);
        Ok(())
    }
}

//
// actual window and wgpu context
//

// An active window (created after the event loop is started)
// and wgpu rendering context.
pub(crate) struct ActiveRenderWindow {
    window: Window,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    swapchain_format: wgpu::TextureFormat,
    msaa_samples: u32,
    msaa_texture: wgpu::Texture,
}

impl ActiveRenderWindow {
    async fn new(
        event_loop: &ActiveEventLoop,
        params: WindowParams,
    ) -> Result<Self, WindowInitError> {
        let mut window_attrs = Window::default_attributes();
        window_attrs.title = "dexterior".to_string();
        window_attrs.min_inner_size = Some(
            winit::dpi::LogicalSize {
                width: params.width as f64,
                height: params.height as f64,
            }
            .into(),
        );
        let window = event_loop.create_window(window_attrs)?;

        let instance = wgpu::Instance::default();
        let surface = unsafe {
            instance.create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&window)
                    .map_err(|_| WindowInitError::HandleError)?,
            )?
        };

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
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
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
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let msaa_texture =
            Self::create_msaa_texture(&device, swapchain_format, params.msaa_samples, window_size);

        Ok(Self {
            window,
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

        let multisample_state = self.multisample_state();

        RenderContext {
            clear_color: Some(wgpu::Color::WHITE),
            surface_tex,
            target: msaa_view,
            resolve_target: surface_view,
            encoder,
            device: &self.device,
            queue: &mut self.queue,
            viewport_size: (self.surface_config.width, self.surface_config.height),
            target_format: self.swapchain_format,
            multisample_state,
        }
    }
}

/// An active surface and other context required to draw a frame.
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
    pub target_format: wgpu::TextureFormat,
    pub multisample_state: wgpu::MultisampleState,
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

//
// animation control
//

/// A `winit` app controlling the playback of an animation.
struct AnimationApp<'mesh, State, StepFn, DrawFn>
where
    State: crate::AnimationState,
    StepFn: FnMut(&mut State),
    DrawFn: FnMut(&State, &mut crate::Painter),
{
    window_params: WindowParams,
    window: Option<(ActiveRenderWindow, pl::Renderer)>,
    anim: crate::Animation<'mesh, State, StepFn, DrawFn>,
    camera: Camera,
    // state for the timing of frames
    frame_start_t: Instant,
    time_in_frame: f64,
    // double-buffered simulation state for interpolated drawing
    prev_state: State,
    next_state: State,
}

impl<'mesh, State, StepFn, DrawFn> winit::application::ApplicationHandler
    for AnimationApp<'mesh, State, StepFn, DrawFn>
where
    State: crate::AnimationState,
    StepFn: FnMut(&mut State),
    DrawFn: FnMut(&State, &mut crate::Painter),
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window =
            futures::executor::block_on(ActiveRenderWindow::new(event_loop, self.window_params))
                .expect("Failed to create window");
        let renderer = pl::Renderer::new(&window, &self.anim.params);
        self.window = Some((window, renderer));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some((window, renderer)) = self.window.as_mut() else {
            println!("window not created yet");
            return;
        };

        match event {
            WindowEvent::RedrawRequested => {
                // step as many times as needed to keep up with real time

                let since_last_draw = self.frame_start_t.elapsed().as_secs_f64();
                self.time_in_frame += since_last_draw;

                let mut steps_done = 0;
                while self.time_in_frame > self.anim.dt {
                    // ...but don't go beyond a maximum to avoid the "spiral of death"
                    // where we constantly fall farther and farther behind real time
                    if steps_done < self.anim.params.max_steps_per_frame {
                        self.prev_state = self.next_state.clone();
                        (self.anim.step)(&mut self.next_state);
                        steps_done += 1;
                    }
                    self.time_in_frame -= self.anim.dt;
                }

                // draw

                self.frame_start_t = Instant::now();

                let mut ctx = window.begin_frame();
                renderer
                    .resources
                    .upload_frame_uniforms(&self.camera, &mut ctx);

                let mut painter = pl::Painter {
                    ctx: &mut ctx,
                    rend: renderer,
                    mesh: self.anim.mesh,
                };

                let interpolated_state = State::interpolate(
                    &self.prev_state,
                    &self.next_state,
                    self.time_in_frame / self.anim.dt,
                );
                (self.anim.draw)(&interpolated_state, &mut painter);

                ctx.queue.submit(Some(ctx.encoder.finish()));
                renderer.end_frame();
                ctx.surface_tex.present();

                window.window.request_redraw();
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                window.resize_swapchain(new_size);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let (ElementState::Pressed, PhysicalKey::Code(code)) =
                    (event.state, event.physical_key)
                {
                    // key mapping similar to matplotlib where applicable
                    match code {
                        KeyCode::KeyQ => {
                            event_loop.exit();
                        }
                        KeyCode::KeyN => {
                            renderer.cycle_color_maps();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}
