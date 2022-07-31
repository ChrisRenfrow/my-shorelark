use crate::*;

// TODO: Look into refactoring this using ECS (Entity Component System)
#[derive(Debug)]
pub struct Animal {
    pub(crate) position: na::Point2<f32>,
    // The following could also be represented as a vector!
    pub(crate) rotation: na::Rotation2<f32>,
    pub(crate) speed: f32,
}

impl Animal {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.002,
        }
    }

    pub fn position(&self) -> na::Point2<f32> {
        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }
}
