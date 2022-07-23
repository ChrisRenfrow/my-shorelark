pub struct Network {
    layers: Vec<Layer>,
}

pub struct LayerTopology {
    pub neurons: usize,
}

impl Network {
    pub fn random(layers: Vec<LayerTopology>) -> Self {
        let mut built_layers = Vec::new();

        for i in 0..(layers.len() - 1) {
            let input_neurons = layers[i].neurons;
            let output_neurons = layers[i + 1].neurons;

            built_layers.push(Layer::random(input_neurons, output_neurons));
        }

        Self {
            layers: built_layers,
        }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn random(input_neurons: usize, output_neurons: usize) -> Self {
        todo!()
    }

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
    pub fn random(output_size: usize) -> Self {
        todo!()
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}
