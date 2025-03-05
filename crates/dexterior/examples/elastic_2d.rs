use std::f64::consts::{PI, TAU};

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
    boundary: dex::Subset<1, dex::Primal>,
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

    let mut layers: Vec<MaterialArea> = Vec::new();
    // loop until no more layers found
    // instead of hardcoding layer count
    // so that we can easily change this via gmsh parameters
    let mut layer = 1;
    loop {
        let group_id = format!("{layer}");
        let (Some(verts), Some(edges), Some(tris)) = (
            mesh.get_subset::<0>(&group_id),
            mesh.get_subset::<1>(&group_id),
            mesh.get_subset::<2>(&group_id),
        ) else {
            break;
        };

        let boundary = mesh.subset_boundary(&tris);

        let lambda = 1.;
        let mu = 1.;
        let density = layer as f64;
        let stiffness = lambda + 2. * mu;

        let p_wave_speed = f64::sqrt(stiffness / density);
        let s_wave_speed = f64::sqrt(mu / density);

        layers.push(MaterialArea {
            verts,
            edges,
            tris,
            boundary,
            mu,
            density,
            stiffness,
            p_wave_speed,
            s_wave_speed,
        });
        layer += 1;
    }

    let boundary_edges = mesh.boundary::<1>();
    let bottom_edges = mesh.get_subset::<1>("990").expect("Subset not found");
    let bottom_verts = mesh.get_subset::<0>("990").expect("Subset not found");
    let bottom_adjacent_dual_verts =
        dex::Subset::from_cell_iter(mesh.simplices_in(&bottom_edges).map(|edge| {
            let (_orientation, adjacent_tri) = edge.coboundary().next().unwrap();
            adjacent_tri.dual()
        }));

    // other parameters

    let dt = 1. / 120.;
    // source waves

    // angle of wave propagation between 0 and 90 degrees
    // (0 is horizontal, 90 vertical;
    // this interval is assumed in boundary conditions)
    let wave_angle_deg = 60.;
    let wave_angle = wave_angle_deg * TAU / 360.;
    let wave_dir = dex::Vec2::new(wave_angle.cos(), wave_angle.sin());
    let pressure_wavenumber = 2.;
    let pressure_angular_vel = layers[0].p_wave_speed * pressure_wavenumber;
    let pressure_wave_vector = pressure_wavenumber * wave_dir;
    let shear_wavenumber = 1.;
    let shear_angular_vel = layers[0].s_wave_speed * shear_wavenumber;
    let shear_wave_vector = shear_wavenumber * wave_dir;

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

    let p_pulse_time = TAU / pressure_angular_vel;
    let s_pulse_time = TAU / shear_angular_vel;

    // only send the source out from the middle of the region
    // to avoid it reflecting off the sides and interfering with itself immediately
    let pressure_source = |pos: dex::Vec2, t: f64| -> f64 {
        pressure_angular_vel * f64::sin(pressure_angular_vel * t - pressure_wave_vector.dot(&pos))
    };
    let pressure_source_vec = |pos: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
        // we only want one wave pulse,
        // so for angled pulses we need to compute when it reaches the given point
        let pulse_start_time = wave_dir.dot(&pos) / layers[0].p_wave_speed;
        let pulse_end_time = pulse_start_time + p_pulse_time;
        if t < pulse_start_time || t > pulse_end_time {
            return 0.;
        }

        let normal = dex::Vec2::new(dir.y, -dir.x);
        let vel = -pressure_wave_vector
            * f64::sin(pressure_angular_vel * t - pressure_wave_vector.dot(&pos));
        vel.dot(&normal)
    };
    let shear_source_vec = |p: dex::Vec2, dir: dex::UnitVec2, t: f64| -> f64 {
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
            color_map_range: Some(-4.0..4.0),
            ..Default::default()
        },
        dt,
        state,
        step: |state| {
            state.q += &ops.q_step_p * &state.p + &ops.q_step_w * &state.w;
            // TODO this doesn't work as a shear source
            // for vertical wavevectors because flux over the horizontal edge is zero
            mesh.integrate_overwrite(
                &mut state.q,
                &bottom_edges,
                dex::quadrature::GaussLegendre6(|p, d| pressure_source_vec(p, d, state.t)),
            );
            // absorbing boundary
            for layer in &layers {
                // TODO turn bottom boundary into an absorbing one after pulse has been sent
                let edges_here = boundary_edges
                    .intersection(&layer.edges)
                    .difference(&bottom_edges);

                for edge in mesh.simplices_in(&edges_here) {
                    let length = edge.volume();
                    // pressure from the adjacent dual vertex
                    let (orientation, tri) = edge.coboundary().next().unwrap();
                    state.q[edge] = -state.p[tri.dual()] * length * orientation as f64
                        / (layer.p_wave_speed * layer.density);
                    // shear from the dual 2-cells at the boundary vertices
                    // (this is probably wrong, just done by analogy/intuition)
                    // for (orientation, vert) in edge.boundary() {
                    //     state.q[edge] -= 0.5 * state.w[vert.dual()] * length * orientation as f64
                    //         / layer.s_wave_speed;
                    // }
                }
            }

            state.p += &ops.p_step * &state.q;
            state.w += &ops.w_step * &state.q;

            state.t += dt;
        },
        draw: |state, draw| {
            draw.axes_2d(dv::AxesParams::default());
            draw.triangle_colors_dual(&state.p);
            // draw.vertex_colors_dual(&state.w);
            draw.wireframe(dv::WireframeParams {
                width: dv::LineWidth::ScreenPixels(1.),
                ..Default::default()
            });
            for layer in &layers {
                draw.wireframe_subset(
                    dv::WireframeParams {
                        width: dv::LineWidth::ScreenPixels(3.),
                        ..Default::default()
                    },
                    &layer.boundary,
                )
            }
            // draw.flux_arrows(
            //     &state.q,
            //     dv::ArrowParams {
            //         ..Default::default()
            //     },
            // );
        },
    })?;

    Ok(())
}
