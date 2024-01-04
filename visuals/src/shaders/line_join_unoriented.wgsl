struct CameraUniforms {
    view_proj: mat4x4<f32>,
    basis: mat3x3<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

struct Parameters {
    width: f32,
}
@group(1) @binding(0)
var<uniform> params: Parameters;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    // position of the individual vertex in the join geometry
    @location(0) pos_local: vec2<f32>,
    // position of an endpoint in the list of line segments
    @location(1) center: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let basis = mat2x3<f32>(camera.basis[0], camera.basis[1]);
    let pos_world = center + params.width * basis * pos_local;
    out.clip_position = camera.view_proj * vec4<f32>(pos_world, 1.0);

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
