use std::f64::consts::PI;

use dexterior as dex;
use dexterior_visuals as dv;

type Pressure = dex::Cochain<0, dex::Dual>;
type Flux = dex::Cochain<1, dex::Primal>;
type Shear = dex::Cochain<2, dex::Dual>;

#[derive(Clone, Debug)]
struct State {
    t: f64,
    p: Pressure,
    q: Flux,
    w: Shear,
}

impl dv::AnimationState for State {
    fn interpolate(old: &Self, new: &Self, t: f64) -> Self {
        Self {
            t: old.t + t * (new.t - old.t),
            p: old.p.lerp(&new.p, t),
            q: old.q.lerp(&new.q, t),
            w: old.w.lerp(&new.w, t),
        }
    }
}

struct Ops {
    p_step: dex::Op<Flux, Pressure>,
    q_step_p: dex::Op<Pressure, Flux>,
    q_step_w: dex::Op<Shear, Flux>,
    w_step: dex::Op<Flux, Shear>,
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
    p_wave_speed: f64,
    s_wave_speed: f64,
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

            let p_wave_speed = (stiffness / density).sqrt();
            let s_wave_speed = (mu / density).sqrt();

            MaterialArea {
                verts,
                edges,
                tris,
                mu,
                density,
                stiffness,
                p_wave_speed,
                s_wave_speed,
            }
        })
        .collect();

    let boundary_edges = mesh.boundary::<1>();
    let bottom_edges = mesh.get_subset::<1>("990").expect("Subset not found");
    let bottom_verts = mesh.get_subset::<0>("990").expect("Subset not found");
    let bottom_adjacent_dual_verts =
        dex::Subset::from_cell_iter(mesh.simplices_in(&bottom_edges).map(|edge| {
            let (_orientation, adjacent_tri) = edge.coboundary().next().unwrap();
            adjacent_tri.dual()
        }));

    // other parameters

    let dt = 1. / 60.;
    // source waves
    let pressure_angular_vel = 1.;
    let pressure_wave_vector = dex::Vec2::new(0., 2.);
    let shear_angular_vel = 1.;
    let shear_wave_vector = dex::Vec2::new(-1., 1.);

    // operators

    // spatially varying scaling factors
    let stiffness_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.tris.contains(s.dual())).unwrap();
        l.stiffness
    });
    let density_scaling = mesh.scaling(|s| {
        let l = layers.iter().find(|l| l.edges.contains(s)).unwrap();
        1. / l.density
    });
    let mu_scaling = mesh.scaling_dual(|s| {
        let l = layers.iter().find(|l| l.verts.contains(s.dual())).unwrap();
        l.mu
    });

    let ops = Ops {
        p_step: dt * stiffness_scaling * mesh.star() * mesh.d(),
        q_step_p: dt * density_scaling.clone() * mesh.star() * mesh.d(),
        q_step_w: -dt * density_scaling * mesh.d() * mesh.star(),
        w_step: dt * mu_scaling * mesh.d() * mesh.star(),
    };

    // source terms

    let pressure_source = |p: dex::Vec2, t: f64| -> f64 {
        // only one pulse so we can see individual waves refracting
        // (TODO: turn into an absorbing boundary after that)
        if t > pressure_angular_vel * PI {
            return 0.;
        }
        pressure_angular_vel * f64::sin(pressure_angular_vel * t - pressure_wave_vector.dot(&p))
    };
    let shear_source = |p: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        if t > shear_angular_vel * PI {
            return 0.;
        }
        let vel = -shear_wave_vector * f64::sin(shear_angular_vel * t - shear_wave_vector.dot(&p));
        vel.dot(&dir)
    };

    // run simulation

    let state = State {
        t: 0.,
        p: mesh.new_zero_cochain(),
        q: mesh.new_zero_cochain(),
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
            state.q += &ops.q_step_p * &state.p + &ops.q_step_w * &state.w;
            state.p += &ops.p_step * &state.q;
            state.w += &ops.w_step * &state.q;
            // sources on the bottom edge
            mesh.integrate_overwrite(
                &mut state.p,
                &bottom_adjacent_dual_verts,
                dex::quadrature::Pointwise(|p| pressure_source(p, state.t)),
            );
            // TODO this doesn't actually work as a shear source
            // for vertical wavevectors because flux over the horizontal edge is zero
            mesh.integrate_overwrite(
                &mut state.q,
                &bottom_edges,
                dex::quadrature::GaussLegendre6(|p, d| shear_source(p, d, state.t)),
            );
            // absorbing boundary
            for layer in &layers {
                let edges_here = boundary_edges.intersection(&layer.edges);
                for edge in mesh.simplices_in(&edges_here) {
                    let length = edge.volume();
                    // pressure from the adjacent dual vertex
                    let (orientation, tri) = edge.coboundary().next().unwrap();
                    state.q[edge] =
                        -state.p[tri.dual()] * length * orientation as f64 / layer.p_wave_speed;
                    // shear from the dual 2-cells at the boundary vertices
                    // (this is probably wrong, just done by analogy/intuition)
                    for (orientation, vert) in edge.boundary() {
                        state.q[edge] -= 0.5 * state.w[vert.dual()] * length * orientation as f64
                            / layer.s_wave_speed;
                    }
                }
            }

            state.t += dt;
        },
        draw: |state, draw| {
            draw.axes_2d(dv::AxesParams::default());
            draw.triangle_colors_dual(&state.p);
            // draw.vertex_colors_dual(&state.w);
            draw.wireframe(dv::WireframeParams::default());
            draw.flux_arrows(
                &state.q,
                dv::ArrowParams {
                    scaling: 0.3,
                    ..Default::default()
                },
            );
        },
    })?;

    Ok(())
}
