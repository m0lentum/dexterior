struct CameraUniforms {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> unif: CameraUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    out.clip_position = unif.view_proj * vec4<f32>(position, 1.0);

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
