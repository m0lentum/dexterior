struct CameraUniforms {
    view_proj: mat4x4<f32>,
    basis: mat3x3<f32>,
    resolution: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

struct Parameters {
    // enum deciding how to compute line width.
    // 0 = world space
    // 1 = screen space
    scaling_mode: u32,
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

    switch params.scaling_mode {
	// world space scaling
	case 0u: {
	    let x_basis = end_point - start_point;
	    // Y basis vector points in a direction orthogonal
	    // to both the camera direction and the X basis vector in 3D space
	    let y_basis = params.width * normalize(cross(camera.basis[2], x_basis));
	    let basis_mat = mat2x3<f32>(x_basis, y_basis);
	    let pos_world = start_point + basis_mat * pos_local;
	    out.clip_position = camera.view_proj * vec4<f32>(pos_world, 1.0);
	}
	// screen space scaling
	case 1u: {
	    // convert start and end points to screen space
	    // (we only care about scaling here
	    // so no need to add the offset from the bottom left
	    // to get the correct pixel position)

	    let start_clip = (camera.view_proj * vec4<f32>(start_point, 1.));
	    let end_clip = (camera.view_proj * vec4<f32>(end_point, 1.));
	    let start_screen_xy = camera.resolution * (0.5 * start_clip.xy / start_clip.w);
	    let end_screen_xy = camera.resolution * (0.5 * end_clip.xy / end_clip.w);
	    // add z and w components to basis vectors for mapping back to clip space
	    let start_screen = vec4<f32>(start_screen_xy, start_clip.zw);
	    let end_screen = vec4<f32>(end_screen_xy, end_clip.zw);

	    // compute vertex position in screen space

	    let x_basis = end_screen - start_screen;
	    let y_basis = vec4<f32>(params.width * normalize(vec2<f32>(-x_basis.y, x_basis.x)), 0., 0.);
	    let basis_mat = mat2x4(x_basis, y_basis);
	    let pos_screen = start_screen + basis_mat * pos_local;

	    // convert back to clip space

	    out.clip_position = vec4<f32>(
		pos_screen.w * 2. * pos_screen.xy / camera.resolution,
		pos_screen.z,
		pos_screen.w,
	    );
	}
	default: {}
    }

    return out;
}

@fragment
fn fs_main(
    in: VertexOutput
) -> @location(0) vec4<f32> {
    return params.color;
}
