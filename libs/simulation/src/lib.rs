use nalgebra as na;

pub struct Simulation {
    world: World,
}

#[derive(Debug)]
pub struct World;

// TODO: Look into refactoring this using ECS (Entity Component System)
#[derive(Debug)]
pub struct Animal {
    position: na::Point2<f32>,
    // The following could also be represented as a vector!
    rotation: na::Rotation2<f32>,
    speed: f32,
}

#[derive(Debug)]
pub struct Food {
    position: na::Point2<f32>,
}
