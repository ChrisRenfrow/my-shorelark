pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn propagate(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    fn propagate(&self, input: &[f32]) -> f32 {
        assert_eq!(input.en(), self.weights.len());

        let mut output = 0.0;

        output = input
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        output += self.bias;

        output.max(0.0)
    }
}
