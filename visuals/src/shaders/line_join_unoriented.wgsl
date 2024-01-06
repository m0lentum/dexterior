struct CameraUniforms {
    view_proj: mat4x4<f32>,
    basis: mat3x3<f32>,
    resolution: vec2<f32>,
};
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

struct Parameters {
    // enum deciding how to compute coordinates.
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
    // position of the individual vertex in the join geometry
    @location(0) pos_local: vec2<f32>,
    // position of an endpoint in the list of line segments
    @location(1) center: vec3<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    switch params.scaling_mode {
	// world space scaling
    	case 0u: {
	    let basis = mat2x3<f32>(camera.basis[0], camera.basis[1]);
	    let pos_world = center + params.width * basis * pos_local;
	    out.clip_position = camera.view_proj * vec4<f32>(pos_world, 1.0);
	}
	// screen space scaling
	case 1u: {
	    // convert center to screen space
	    let center_clip = (camera.view_proj * vec4<f32>(center, 1.));
	    let center_screen = camera.resolution * (0.5 * center_clip.xy / center_clip.w);
	    // compute vertex position in screen space
	    let pos_screen = center_screen + params.width * pos_local;
	    // convert back to clip space
	    out.clip_position = vec4<f32>(
		center_clip.w * 2. * pos_screen / camera.resolution,
		center_clip.z,
		center_clip.w,
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
