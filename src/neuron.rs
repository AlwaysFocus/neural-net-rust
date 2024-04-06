use rand::Rng;


pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Self {
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
        Neuron { weights, bias }
    }
    
    pub fn activate(&self, inputs: &[f64]) -> f64 {
        // Calculate the weighted sum of inputs and weights
        let weighted_sum: f64 = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(w, i)| w * i)
            .sum();
        
        // Add the bias term to the weighted sum
        let output = weighted_sum + self.bias;
        
        // Apply the sigmoid activation function to the output
        // This squashes the output between 0 and 1
        self.sigmoid(output)
    }
    
    fn sigmoid(&self, x: f64) -> f64 {
        // Sigmoid activation function: 1 / (1 + e^(-x))
        // This function is used to introduce non-linearity into the network
        1.0 / (1.0 + (-x).exp())
    }
}