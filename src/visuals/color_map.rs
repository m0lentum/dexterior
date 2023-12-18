//! Color maps for colorizing simulation data.

use itertools::izip;

/// Colors in color maps are represented as 8-bit colors **in linear sRGB space**.
pub type Color = [u8; 4];
pub(crate) const LUT_SIZE: usize = 256;
pub(crate) const TEX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// A map determining how to colorize data.
#[derive(Clone, Debug)]
pub struct ColorMap {
    /// Name used to activate the map with
    /// [`Painter::set_color_map`][crate::Painter::set_color_map].
    pub name: String,
    /// The color map expressed as a lookup table.
    ///
    /// This is a very flexible and efficient method,
    /// as it doesn't restrict us to interpolating in any specific color space
    /// like some kind of parametric gradient representation would,
    /// and allows the use of sampled textures on the GPU.
    pub(crate) lut: [Color; LUT_SIZE],
}

impl ColorMap {
    /// Create a color map from an [`enterpolation`] curve
    /// interpolating [`palette`] colors.
    pub fn from_curve<Curve, Color>(name: String, curve: Curve) -> Self
    where
        Color: palette::IntoColor<palette::Srgb>,
        Curve: enterpolation::Curve<f32, Output = Color>,
    {
        let vals = curve.take(LUT_SIZE);
        let mut lut = [[0; 4]; LUT_SIZE];
        for (color, lut_val) in izip!(vals, lut.iter_mut()) {
            let c_lin: palette::Srgb = color.into_color();
            let as_u8 = |channel: f32| (u8::MAX as f32 * channel).round() as u8;
            *lut_val = [
                as_u8(c_lin.red),
                as_u8(c_lin.green),
                as_u8(c_lin.blue),
                u8::MAX,
            ];
        }
        Self { name, lut }
    }

    /// Create a color map from a function
    /// that takes a float value between 0 and 1 and returns a color.
    pub fn from_fn(&self, name: String, curve: impl Fn(f32) -> Color) -> Self {
        // minus one because we have LUT_SIZE points
        // and thus (LUT_SIZE-1) gaps between points in the interval
        let increment = 1.0 / (LUT_SIZE - 1) as f32;
        Self {
            name,
            lut: std::array::from_fn(|i| (curve)(increment * i as f32)),
        }
    }
}

pub mod builtin_color_maps {
    //! A collection of premade color maps.
    //!
    //! Contains the following maps:
    //! - [`sunset`]
    //! - [`molentum`]
    //!
    //! (TODO: pictures of these would be nice)
    //!
    //! All of these are loaded by default.
    //! The default set can be overridden with [`AnimationParams`][`crate::AnimationParams`].

    use super::ColorMap;
    use enterpolation::linear::ConstEquidistantLinear;
    use palette::{FromColor, Oklab, Srgb};

    /// Convenience function for converting colors
    /// given as sRGB hexadecimal codes in 0xRRGGBB format
    /// (easily obtained from standard color pickers) to Oklab.
    ///
    /// Interpolating in Oklab gives nice perceptually uniform gradients,
    /// which is desirable for color maps.
    fn srgb_hex(val: u32) -> Oklab {
        let srgb_u8 = Srgb::from(val);
        let srgb_float: Srgb<f32> = srgb_u8.into_format();
        Oklab::from_color(srgb_float)
    }

    /// Convenience function for converting floating point sRGB values
    /// to an Oklab color.
    fn srgb_float(r: f32, g: f32, b: f32) -> Oklab {
        Oklab::from_color(Srgb::new(r, g, b))
    }

    /// Convenience function to make a color map
    /// as an array of equally spaced colors in Oklab space.
    fn linear_equidistant<const COUNT: usize>(name: &str, colors: [Oklab; COUNT]) -> ColorMap {
        ColorMap::from_curve(
            name.to_string(),
            ConstEquidistantLinear::equidistant_unchecked(colors),
        )
    }

    /// A collection of all builtin color maps.
    pub fn all_builtins() -> Vec<ColorMap> {
        vec![sunset(), molentum()]
    }

    /// A perceptually uniform map from dark blue to bright yellow.
    pub fn sunset() -> ColorMap {
        linear_equidistant(
            "sunset",
            [
                srgb_float(0.00, 0.05, 0.20),
                srgb_float(0.70, 0.10, 0.20),
                srgb_float(0.95, 0.90, 0.30),
            ],
        )
    }

    /// The author's signature blue and lime color scheme.
    pub fn molentum() -> ColorMap {
        linear_equidistant(
            "molentum",
            [srgb_hex(0x161f2e), srgb_hex(0x278c63), srgb_hex(0xbada55)],
        )
    }
}
