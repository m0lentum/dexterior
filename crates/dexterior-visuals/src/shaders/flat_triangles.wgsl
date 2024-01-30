struct CameraUniforms {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> unif: CameraUniforms;
@group(0) @binding(1)
var colormap_t: texture_2d_array<f32>;
@group(0) @binding(2)
var colormap_s: sampler;

struct DataStorage {
    color_map_idx: u32,
    color_map_range_start: f32,
    color_map_range_length: f32,
    data: array<f32>,
}
@group(1) @binding(0)
var<storage, read> cochain_vals: DataStorage;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color_interp: f32,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @builtin(vertex_index) vert_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = unif.view_proj * vec4<f32>(position, 1.0);

    // three vertices per triangle -> the corresponding cochain value is at vert_idx / 3
    // (other than this, this is the exact same shader as vertex_colors.wgsl)
    let c_val = cochain_vals.data[vert_idx / 3u];
    let c_val_scaled = (c_val - cochain_vals.color_map_range_start) / cochain_vals.color_map_range_length;
    out.color_interp = clamp(c_val_scaled, 0., 1.);

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    let interp_uv = vec2<f32>(in.color_interp, 0.5);
    let mapped_color = textureSample(colormap_t, colormap_s, interp_uv, cochain_vals.color_map_idx);
    return mapped_color;
}
