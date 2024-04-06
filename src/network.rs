use crate::neuron::Neuron;

pub struct Network {
    pub input_neurons: Vec<Neuron>,
    pub hidden_neurons: Vec<Neuron>,
    pub output_neuron: Neuron,
}

impl Network {
    pub fn new(num_inputs: usize, num_hidden: usize) -> Self {
        // Create input neurons, each with a single weight
        // These neurons simply pass the input values to the next layer
        let input_neurons = (0..num_inputs)
            .map(|_| Neuron::new(1))
            .collect();
        
        // Create hidden neurons, each with weights corresponding to the number of input features
        // These neurons process the input data and extract meaningful features
        let hidden_neurons = (0..num_hidden)
            .map(|_| Neuron::new(1))
            .collect();
        
        // Create our output neuron with weights corresponding to the number of hidden neurons
        // This neuron produces the final output of the network
        let output_neuron = Neuron::new(num_hidden);
        
        // Create and return our new Network instance
        Network {
            input_neurons,
            hidden_neurons,
            output_neuron
        }
    }
    
    pub fn predict(&self, inputs: &[f64]) -> f64 {
        // Feed the inputs through the input layer
        let hidden_inputs: Vec<f64> = self
            .input_neurons
            .iter()
            .zip(inputs.iter())
            .map(|(neuron, input)| neuron.activate(&[*input]))
            .collect();
        
        // Feed the outputs of the input layer through the hidden layer
        let hidden_outputs: Vec<f64> = self
            .hidden_neurons
            .iter()
            .map(|neuron| neuron.activate(&hidden_inputs))
            .collect();
        
        // Feed the outputs of the hidden layer through the output neuron
        // Return the final output of the network
        self.output_neuron.activate(&hidden_outputs)
    }
}