struct Camera {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var colormap_tex: texture_2d_array<f32>;
@group(0) @binding(2)
var colormap_samp: sampler;

struct ColorMapParams {
    layer_idx: u32,
    range_start: f32,
    range_length: f32,
    // pad the size to a multiple of 16 bytes for WebGL support
    _pad: u32,
}
@group(1) @binding(0)
var<uniform> colormap: ColorMapParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color_interp: f32,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) cochain_val: f32,
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);

    let c_val_scaled = (cochain_val - colormap.range_start) / colormap.range_length;
    out.color_interp = clamp(c_val_scaled, 0., 1.);

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    let interp_uv = vec2<f32>(in.color_interp, 0.5);
    let mapped_color = textureSample(colormap_tex, colormap_samp, interp_uv, colormap.layer_idx);
    return mapped_color;
}
