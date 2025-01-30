use std::f64::consts::PI;

use dexterior as dex;
use dexterior_visuals as dv;
use nalgebra as na;

type Pressure = dex::Cochain<0, dex::Primal>;
type Velocity = dex::Cochain<1, dex::Primal>;
type Shear = dex::Cochain<0, dex::Dual>;

#[derive(Clone, Debug)]
struct State {
    t: f64,
    p: Pressure,
    v: Velocity,
    w: Shear,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            t: old.t + t * (new.t - old.t),
            p: old.p.lerp(&new.p, t),
            v: old.v.lerp(&new.v, t),
            w: old.w.lerp(&new.w, t),
        }
    }
}

struct Ops {
    p_step: dex::Op<Velocity, Pressure>,
    v_step_p: dex::Op<Pressure, Velocity>,
    v_step_w: dex::Op<Shear, Velocity>,
    w_step: dex::Op<Velocity, Shear>,
}

struct MaterialArea {
    verts: dex::Subset<0, dex::Primal>,
    edges: dex::Subset<1, dex::Primal>,
    tris: dex::Subset<2, dex::Primal>,
    // lamè coefficients and other parameters of the material in this area
    // (stiffness encodes the lambda part of lamè coefficients)
    mu: f64,
    density: f64,
    stiffness: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let msh_bytes = include_bytes!("./meshes/2d_layered.msh");
    let mesh = dex::gmsh::load_trimesh_2d(msh_bytes)?;

    // spatially varying material parameters from mesh physical groups

    let layer_count = 8;
    let layers: Vec<MaterialArea> = (1..=layer_count)
        .map(|layer| {
            let group_id = format!("{layer}");
            let verts = mesh.get_subset::<0>(&group_id).expect("Subset not found");
            let edges = mesh.get_subset::<1>(&group_id).expect("Subset not found");
            let tris = mesh.get_subset::<2>(&group_id).expect("Subset not found");

            let lambda = 1.;
            let mu = 1.;
            let density = layer as f64;
            let stiffness = lambda + 2. * mu;

            MaterialArea {
                verts,
                edges,
                tris,
                mu,
                density,
                stiffness,
            }
        })
        .collect();

    let bottom_edges = mesh.get_subset::<1>("990").expect("Subset not found");
    let bottom_verts = mesh.get_subset::<0>("990").expect("Subset not found");

    // other parameters

    let dt = 1. / 60.;
    let pressure_angular_vel = 1.;
    let pressure_wave_vector = na::Vector2::new(0., 1.);
    let shear_angular_vel = 1.;
    let shear_wave_vector = na::Vector2::new(0., 1.);

    // operators

    // spatially varying scaling factors
    let stiffness_scaling = mesh.scaling(|s| {
        let l = layers.iter().find(|l| l.verts.contains(s)).unwrap();
        l.stiffness
    });
    let density_scaling = mesh.scaling(|s| {
        let l = layers.iter().find(|l| l.edges.contains(s)).unwrap();
        1. / l.density
    });
    let mu_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.mu
    });

    let ops = Ops {
        p_step: -dt * stiffness_scaling * mesh.star() * mesh.d() * mesh.star(),
        v_step_p: dt * density_scaling.clone() * mesh.d(),
        v_step_w: dt * density_scaling * mesh.star() * mesh.d(),
        w_step: dt * mu_scaling * mesh.star() * mesh.d(),
    };

    // source terms

    let pressure_source = |p: na::Vector2<f64>, t: f64| -> f64 {
        // only one pulse so we can see individual waves refracting
        // (TODO: turn into an absorbing boundary after that)
        if t > pressure_angular_vel * PI {
            return 0.;
        }
        pressure_angular_vel * f64::sin(pressure_angular_vel * t - pressure_wave_vector.dot(&p))
    };
    let shear_source = |p: na::Vector2<f64>, dir: na::Unit<na::Vector2<f64>>, t: f64| -> f64 {
        if t > shear_angular_vel * PI {
            return 0.;
        }
        let normal = na::Vector2::new(dir.y, -dir.x);
        let vel = -shear_wave_vector * f64::sin(shear_angular_vel * t - shear_wave_vector.dot(&p));
        vel.dot(&normal)
    };

    // run simulation

    let state = State {
        t: 0.,
        p: mesh.new_zero_cochain(),
        v: mesh.new_zero_cochain(),
        w: mesh.new_zero_cochain(),
    };

    let mut window = dv::RenderWindow::new(dv::WindowParams::default())?;
    window.run_animation(dv::Animation {
        mesh: &mesh,
        params: dv::AnimationParams {
            color_map_range: Some(-0.5..0.5),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            // TODO: t is very common in source terms
            // and should probably be provided in the parameters to the step function here,
            // instead of users having to include it in their state
            state.t += dt;
            state.v += &ops.v_step_p * &state.p + &ops.v_step_w * &state.w;
            state.p += &ops.p_step * &state.v;
            state.w += &ops.w_step * &state.v;
            // sources on the bottom edge
            mesh.integrate_overwrite(
                &mut state.p,
                &bottom_verts,
                dex::quadrature::Pointwise(|p| pressure_source(p, state.t)),
            );
            mesh.integrate_overwrite(
                &mut state.v,
                &bottom_edges,
                dex::quadrature::GaussLegendre6(|p, d| shear_source(p, d, state.t)),
            );
        },
        draw: |state, draw| {
            draw.axes_2d(dv::AxesParams::default());
            // draw.vertex_colors(&state.p);
            draw.triangle_colors_dual(&state.w);
            draw.wireframe(dv::WireframeParams::default());
            draw.velocity_arrows(
                &state.v,
                dv::ArrowParams {
                    scaling: 0.3,
                    ..Default::default()
                },
            );
        },
    })?;

    Ok(())
}
