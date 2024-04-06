use crate::neuron::{ActivationFunction, Neuron};

pub struct Network {
    pub layers: Vec<Vec<Neuron>>,
}

impl Network {
    pub fn new(layer_sizes: &[usize], activation_fn: ActivationFunction) -> Self {
        // Create pairs of adjacent layers and initialize neurons on each
        // layer based on the number of neurons in the previous layer
        let layers = layer_sizes
            .windows(2)
            .map(|w| (0..w[1]).map(|_| Neuron::new(w[0], activation_fn.clone())).collect())
            .collect();
        
        Network { layers }
    }
    
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        // Feed the inputs through the network and return final outputs
        self.forward(inputs).last().unwrap().to_vec()
    }
    
    pub fn train(&mut self, inputs: &[f64], targets: &[f64], learning_rate: f64) {
        let outputs = self.forward(inputs);
        let errors = self.backward(targets, &outputs);
        self.update_weights(inputs, &errors, learning_rate);
    }
    
    fn forward(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut outputs = vec![inputs.to_vec()];
        for layer in &self.layers {
            let layer_outputs: Vec<f64> = layer
                .iter()
                .map(|neuron| neuron.activate(&outputs.last().unwrap()))
                .collect();
            outputs.push(layer_outputs);
        }
        
        outputs
    }

    fn backward(&self, targets: &[f64], outputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut errors = vec![Vec::new(); self.layers.len() + 1];
        errors.last_mut().unwrap().extend(
            outputs
                .last()
                .unwrap()
                .iter()
                .zip(targets)
                .map(|(output, target)| output - target),
        );

        for i in (0..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let layer_errors: Vec<f64> = layer
                .iter()
                .enumerate()
                .map(|(j, neuron)| {
                    let error = errors[i + 1]
                        .iter()
                        .zip(&neuron.weights)
                        .map(|(err, weight)| err * weight)
                        .sum::<f64>();
                    error * self.activation_derivative(outputs[i + 1][j], &neuron.activation_fn)
                })
                .collect();
            errors[i] = layer_errors;
        }
        errors
    }

    fn update_weights(&mut self, inputs: &[f64], errors: &[Vec<f64>], learning_rate: f64) {
        // Update the weights and biases based on the calculated errors
        for (i, layer) in self.layers.iter_mut().enumerate() {
            for (neuron, &error) in layer.iter_mut().zip(errors[i + 1].iter()) {
                neuron.bias -= learning_rate * error;
                neuron.weights = neuron
                    .weights
                    .iter()
                    .zip(inputs.iter())
                    .map(|(&weight, &input)| weight - learning_rate * error * input)
                    .collect();
            }
        }
    }
    
    fn activation_derivative(&self, output: f64, activation_fn: &ActivationFunction) -> f64 {
        // Calculate derivative of activation function
        match activation_fn {
            ActivationFunction::Sigmoid => output * (1.0 - output),
            ActivationFunction::ReLU => if output > 0.0 {1.0} else {0.0},
            ActivationFunction::Tanh => 1.0 - output.powi(2),
            
        }
    }
}