use rand::Rng;

#[derive(Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
}

pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation_fn: ActivationFunction,
}

impl Neuron {
    pub fn new(num_inputs: usize, activation_fn: ActivationFunction) -> Self {
        // Create a new random number generator
        let mut rng = rand::thread_rng();

        // Initialize weights randomly between -1 and 1
        // This will help the network learn more effectively
        let weights = (0..num_inputs)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // Initialize the bias randomly between -1 and 1
        let bias = rng.gen_range(-1.0..1.0);

        // Create and return the new Neuron instance
        Neuron { weights, bias, activation_fn }
    }

    pub fn activate(&self, inputs: &[f64]) -> f64 {
        // Calculate the weighted sum of inputs and weights with bias
        let output = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum::<f64>() + self.bias;

        // Execute activation function
        match self.activation_fn {
            ActivationFunction::Sigmoid => self.sigmoid(output),
            ActivationFunction::ReLU => self.relu(output),
            ActivationFunction::Tanh => self.tanh(output),
        }
    }

    fn sigmoid(&self, x: f64) -> f64 {
        // Sigmoid activation function: 1 / (1 + e^(-x))
        // This function is used to introduce non-linearity into the network
        1.0 / (1.0 + (-x).exp())
    }
    
    fn relu(&self, x: f64) -> f64 {
        x.max(0.0)
    }
    
    fn tanh(&self, x: f64) -> f64 {
        x.tanh()
    }
}