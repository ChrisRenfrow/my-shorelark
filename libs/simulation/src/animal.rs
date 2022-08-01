use crate::*;

// TODO: Look into refactoring this using ECS (Entity Component System)
pub struct Animal {
    pub(crate) position: na::Point2<f32>,
    // The following could also be represented as a vector!
    pub(crate) rotation: na::Rotation2<f32>,
    pub(crate) speed: f32,
    pub(crate) eye: Eye,
    pub(crate) brain: nn::Network,
}

impl Animal {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();

        let brain = nn::Network::random(
            rng,
            &[
                // Input Layer
                nn::LayerTopology {
                    neurons: eye.cells(),
                },
                // Hidden Layer
                nn::LayerTopology {
                    neurons: 2 * eye.cells(),
                },
                // Output Layer
                nn::LayerTopology { neurons: 2 },
            ],
        );

        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.002,
            eye,
            brain,
        }
    }

    pub fn position(&self) -> na::Point2<f32> {
        self.position
    }

    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }
}
