//! Low-level resources for window creation and rendering.

use web_time::Instant;

#[cfg(target_arch = "wasm32")]
use winit::event_loop::EventLoopProxy;
use winit::{
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use nalgebra as na;

use super::{camera::Camera, pipelines as pl};

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
    ///
    /// Note that MSAA is not supported on WebGL,
    /// so this setting does nothing there.
    pub msaa_samples: u32,
    /// Id of the HTML element to embed this window under on the web.
    /// Default: "wgpu-canvas".
    ///
    /// This is useful for embedding multiple examples
    /// in different locations on one web page.
    /// If an element with this id is not found,
    /// the canvas is appended to the end of the document instead.
    ///
    /// This does not do anything outside of the web platform.
    pub parent_element_id: &'static str,
}

impl Default for WindowParams {
    fn default() -> Self {
        Self {
            width: 800,
            height: 800,
            msaa_samples: 4,
            parent_element_id: "wgpu-canvas",
        }
    }
}

/// Running on wasm requires wrangling wgpu resources through a custom event
/// due to lack of async support directly in ApplicationHandler methods.
/// There's a bunch of messy code here related to this
#[derive(Debug)]
enum CustomEvent {
    #[cfg(target_arch = "wasm32")]
    WindowCreated(ActiveRenderWindow),
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
    event_loop: Option<EventLoop<CustomEvent>>,
}

impl RenderWindow {
    /// Create a new render window.
    pub fn new(params: WindowParams) -> Result<Self, winit::error::EventLoopError> {
        Ok(Self {
            params,
            event_loop: Some(EventLoop::with_user_event().build()?),
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
    ///
    /// # Consecutive animations and WASM
    ///
    /// On native platforms, this function returns after the animation is aborted
    /// by closing the window or pressing q.
    /// Thus you can run multiple animations consecutively in one program.
    /// On the web, on the other hand, this never returns due to limitations
    /// in window handling inside of a browser.
    /// Thus you can only run one animation per program.
    /// This is a limitation that can be worked around
    /// and hopefully will be in the future, but requires some nontrivial work.
    pub fn run_animation<State, StepFn, DrawFn, OnKeyFn>(
        &mut self,
        anim: super::animation::Animation<State, StepFn, DrawFn, OnKeyFn>,
    ) -> Result<(), winit::error::EventLoopError>
    where
        State: crate::AnimationState,
        StepFn: FnMut(&mut State),
        DrawFn: FnMut(&State, &mut crate::Painter),
        OnKeyFn: FnMut(crate::KeyCode, &mut State),
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

        #[allow(unused_mut)]
        let mut event_loop = self.event_loop.take().unwrap();
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut anim_app = AnimationApp {
            #[cfg(target_arch = "wasm32")]
            loop_proxy: Some(event_loop.create_proxy()),

            window_params: self.params,
            window: None,
            camera,
            frame_start_t: Instant::now(),
            time_in_frame: 0.,
            prev_state: anim.state.clone(),
            next_state: anim.state.clone(),
            anim,
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
            event_loop.run_app_on_demand(&mut anim_app)?;
            self.event_loop = Some(event_loop);
            Ok(())
        }

        #[cfg(target_arch = "wasm32")]
        {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("Failed to initialize console logger");
            // using `run_app` instead of the recommended `spawn_app` here
            // because it allows us to use an API with lifetimes
            event_loop.run_app(&mut anim_app)?;
            Ok(())
        }
    }
}

//
// actual window and wgpu context
//

// An active window (created after the event loop is started)
// and wgpu rendering context.
#[derive(Debug)]
pub(crate) struct ActiveRenderWindow {
    _window: Window,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    swapchain_format: wgpu::TextureFormat,
    msaa_samples: u32,
    // msaa texture is only created if multisampling is used
    msaa_tex: Option<wgpu::Texture>,
}

/// Return type for `ActiveRenderWindow::create_rest`.
/// When not on wasm we return the window directly from creation.
/// On wasm we instead maneuver it through a custom event and return nothing
/// because we can't block on futures to get their return values
#[cfg(not(target_arch = "wasm32"))]
type CreateWindowRet = ActiveRenderWindow;
#[cfg(target_arch = "wasm32")]
type CreateWindowRet = ();

impl ActiveRenderWindow {
    /// Create the window separately from the wgpu context.
    /// This is needed to avoid the lifetime of the event loop in the async task,
    /// since wasm requires the task to be 'static
    fn create_window(event_loop: &ActiveEventLoop, params: WindowParams) -> Window {
        let window_attrs = Window::default_attributes()
            .with_title("dexterior")
            .with_inner_size(winit::dpi::LogicalSize {
                width: params.width as f64,
                height: params.height as f64,
            });
        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("No canvas");
            canvas.set_width(params.width as u32);
            canvas.set_height(params.height as u32);
            let canvas = web_sys::Element::from(canvas);
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(
                    |doc| match doc.get_element_by_id(params.parent_element_id) {
                        Some(parent) => parent.append_child(&canvas).ok(),
                        None => doc.body().and_then(|body| body.append_child(&canvas).ok()),
                    },
                )
                .expect("couldn't append canvas to document body");
        }

        window
    }

    /// Create the rest of the contexts besides the window.
    async fn create_rest(
        window: Window,
        params: WindowParams,
        #[cfg(target_arch = "wasm32")] proxy: EventLoopProxy<CustomEvent>,
    ) -> CreateWindowRet {
        let instance = wgpu::Instance::default();
        let surface = unsafe {
            instance
                .create_surface_unsafe(
                    wgpu::SurfaceTargetUnsafe::from_window(&window)
                        .expect("Failed to get window handle"),
                )
                .expect("Failed to create surface")
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .expect("Failed to get adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits::default(),
                    #[cfg(target_arch = "wasm32")]
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                    label: None,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to get device");

        let window_size = window.inner_size();

        #[cfg(target_arch = "wasm32")]
        let swapchain_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        #[cfg(not(target_arch = "wasm32"))]
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

        // MSAA is not supported on WebGL, so always set samples to 1
        #[cfg(not(target_arch = "wasm32"))]
        let msaa_samples = params.msaa_samples;
        #[cfg(target_arch = "wasm32")]
        let msaa_samples = 1;

        // on the web, we can get a situation where the window size is 0 here.
        // in that case, postpone surface configuration until we get a resize event
        let window_has_pixels = surface_config.width != 0 && surface_config.height != 0;
        if window_has_pixels {
            surface.configure(&device, &surface_config);
        }
        let msaa_tex = if msaa_samples > 1 && window_has_pixels {
            Some(Self::create_msaa_texture(
                &device,
                swapchain_format,
                params.msaa_samples,
                window_size,
            ))
        } else {
            None
        };

        let win = Self {
            _window: window,
            device,
            queue,
            surface,
            surface_config,
            swapchain_format,
            msaa_samples,
            msaa_tex,
        };

        // on wasm, the data needs to be maneuvered out through an event
        // because we can't block on futures
        #[cfg(target_arch = "wasm32")]
        proxy
            .send_event(CustomEvent::WindowCreated(win))
            .expect("Successfully created wgpu context but failed to send window event");
        #[cfg(not(target_arch = "wasm32"))]
        win
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
        if self.msaa_samples > 1 {
            self.msaa_tex = Some(Self::create_msaa_texture(
                &self.device,
                self.swapchain_format,
                self.msaa_samples,
                new_size,
            ));
        }
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
        let (target, resolve_target) = if let Some(msaa_tex) = &self.msaa_tex {
            let msaa_view = msaa_tex.create_view(&wgpu::TextureViewDescriptor::default());
            (msaa_view, Some(surface_view))
        } else {
            (surface_view, None)
        };
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let multisample_state = self.multisample_state();

        RenderContext {
            clear_color: Some(wgpu::Color::WHITE),
            surface_tex,
            target,
            resolve_target,
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
    pub resolve_target: Option<wgpu::TextureView>,
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
                resolve_target: self.resolve_target.as_ref(),
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
struct AnimationApp<'mesh, State, StepFn, DrawFn, OnKeyFn>
where
    State: crate::AnimationState,
    StepFn: FnMut(&mut State),
    DrawFn: FnMut(&State, &mut crate::Painter),
    OnKeyFn: FnMut(crate::KeyCode, &mut State),
{
    // event loop proxy allows us to send wgpu context to the active window
    // after creating it in an async future
    #[cfg(target_arch = "wasm32")]
    loop_proxy: Option<EventLoopProxy<CustomEvent>>,

    window_params: WindowParams,
    window: Option<(ActiveRenderWindow, pl::Renderer)>,
    anim: crate::Animation<'mesh, State, StepFn, DrawFn, OnKeyFn>,
    camera: Camera,
    // state for the timing of frames
    frame_start_t: Instant,
    time_in_frame: f64,
    // double-buffered simulation state for interpolated drawing
    prev_state: State,
    next_state: State,
}

impl<'mesh, State, StepFn, DrawFn, OnKeyFn> winit::application::ApplicationHandler<CustomEvent>
    for AnimationApp<'mesh, State, StepFn, DrawFn, OnKeyFn>
where
    State: crate::AnimationState,
    StepFn: FnMut(&mut State),
    DrawFn: FnMut(&State, &mut crate::Painter),
    OnKeyFn: FnMut(crate::KeyCode, &mut State),
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = ActiveRenderWindow::create_window(event_loop, self.window_params);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let active_win = futures::executor::block_on(ActiveRenderWindow::create_rest(
                window,
                self.window_params,
            ));
            let renderer = pl::Renderer::new(&active_win, &self.anim.params);
            self.window = Some((active_win, renderer));
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(ActiveRenderWindow::create_rest(
                window,
                self.window_params,
                self.loop_proxy.take().unwrap(),
            ));
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: CustomEvent) {
        // get the wgpu context that was created in a spawned task
        let CustomEvent::WindowCreated(active_win) = event;
        let renderer = pl::Renderer::new(&active_win, &self.anim.params);

        self.window = Some((active_win, renderer));
    }

    /// step and draw in about_to_wait even though winit recommends against it,
    /// because waiting for RedrawRequested events causes stuttering on web
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        let Some((window, renderer)) = self.window.as_mut() else {
            return;
        };
        if window.surface_config.width == 0 || window.surface_config.height == 0 {
            // we're on the web (most likely)
            // and the canvas hasn't been fully initialized yet, keep waiting
            return;
        }

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
        renderer.draw_batched(&mut ctx, &self.camera);

        ctx.queue.submit(Some(ctx.encoder.finish()));
        renderer.end_frame();
        ctx.surface_tex.present();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some((window, renderer)) = self.window.as_mut() else {
            return;
        };

        match event {
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
                    (self.anim.on_key)(code, &mut self.next_state);
                    // key mapping similar to matplotlib where applicable
                    match code {
                        KeyCode::KeyQ => {
                            // don't exit on web since there's nothing we can do afterwards there
                            #[cfg(not(target_arch = "wasm32"))]
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
