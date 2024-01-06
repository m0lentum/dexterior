struct CameraUniforms {
    view_proj: mat4x4<f32>,
    basis: mat3x3<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

struct Parameters {
    width: f32,
    color: vec4<f32>,
}
@group(1) @binding(0)
var<uniform> params: Parameters;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(
    // position of the individual vertex in the line segment instance
    @location(0) pos_local: vec2<f32>,
    // start and end points of the segment from the instance buffer
    @location(1) start_point: vec3<f32>,
    @location(2) end_point: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let x_basis = end_point - start_point;
    // Y basis vector points in a direction orthogonal
    // to both the camera direction and the X basis vector in 3D space
    let y_basis = params.width * normalize(cross(camera.basis[2], x_basis));
    let basis_mat = mat2x3<f32>(x_basis, y_basis);

    let pos_world = start_point + basis_mat * pos_local;
    out.clip_position = camera.view_proj * vec4<f32>(pos_world, 1.0);

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return params.color;
}
