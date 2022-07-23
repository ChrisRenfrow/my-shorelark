use rand::RngCore;

pub trait Individual {
    fn fitness(&self) -> f32;
}

pub trait SelectionMethod {
    fn select<'a, R, I>(&self, rng: &mut R, population: &'a [I]) -> &'a I
    where
        R: RngCore,
        I: Individual;
}

pub struct GeneticAlgorithm;

impl GeneticAlgorithm {
    pub fn new() -> Self {
        Self
    }

    pub fn evolve<I>(&self, population: &[I], evaluate_fitness: &dyn Fn(&I) -> f32) -> Vec<I> {
        assert!(!population.is_empty());

        (0..population.len()).map(|_| todo!()).collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
