mod neuron;
mod network;

use network::Network;

fn main() {
    // Define the number of input features and hidden neurons
    let num_inputs = 2;
    let num_hidden = 3;
    
    // Create a network instance with the specified architecture
    let network = Network::new(num_inputs, num_hidden);
    
    // Define the input values for prediction
    let inputs = &[0.5, 0.8];
    
    // Feed the inputs through the network and get the final output
    let output = network.predict(inputs);
    
    // Print the structure of the neural network
    println!("Neural Network Structure:");
    println!("Input layer: {} neurons", network.input_neurons.len());
    println!("Hidden layer: {} neurons", network.hidden_neurons.len());
    println!("Output layer: 1 neuron");
    
    // Input values
    println!("\nInput values: {:?}", inputs);
    
    // Activations of the input layer neurons
    println!("\nInput Layer Activations:");
    for (i, neuron) in network.input_neurons.iter().enumerate() {
        println!("Neuron {}: {:.4}", i + 1, neuron.activate(&[inputs[i]]));
    }
    
    // Activations of the hidden layer neurons
    println!("\nHidden Layer Activations:");
    for (i, neuron) in network.hidden_neurons.iter().enumerate() {
        let activation = neuron.activate(inputs);
        println!("Neuron {}: {:.4}", i + 1, activation);
    }
    
    // Activation of output neuron
    println!("\nOutput Layer Activation:");
    println!("Output Neuron: {:.4}", output);
    
}