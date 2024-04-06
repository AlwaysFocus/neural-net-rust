mod neuron;
mod network;
mod load_dataset;

use network::Network;
use rand::seq::SliceRandom;
use crate::neuron::ActivationFunction;
use crate::load_dataset::load_titanic_dataset;

fn main() {
    // Load and preprocess the Titanic dataset
    let file_path = "data/titanic.csv";
    let (features, targets) = load_titanic_dataset(file_path);

    // Split the data into training and testing sets
    let test_size = 0.2;
    let test_count = (features.len() as f64 * test_size) as usize;
    let mut indices: Vec<usize> = (0..features.len()).collect();
    indices.shuffle(&mut rand::thread_rng());
    let test_indices = &indices[..test_count];
    let train_indices = &indices[test_count..];

    let train_features: Vec<&Vec<f64>> = train_indices.iter().map(|&i| &features[i]).collect();
    let train_targets: Vec<&f64> = train_indices.iter().map(|&i| &targets[i]).collect();
    let test_features: Vec<&Vec<f64>> = test_indices.iter().map(|&i| &features[i]).collect();
    let test_targets: Vec<&f64> = test_indices.iter().map(|&i| &targets[i]).collect();

    // Create and train the neural network
    let num_epochs = 100;
    let learning_rate = 0.01;
    let layer_sizes = &[7, 10, 5, 1];
    let activation_fn = ActivationFunction::Sigmoid;
    let mut network = Network::new(layer_sizes, activation_fn);

    for epoch in 0..num_epochs {
        for (inputs, &&target) in train_features.iter().zip(&train_targets) {
            network.train(inputs, &[target], learning_rate);
        }
        println!("Epoch {}/{} completed", epoch + 1, num_epochs);
    }

    // Evaluate the trained model on the testing set
    let mut correct_predictions = 0;
    for (inputs, &target) in test_features.iter().zip(&test_targets) {
        let prediction = network.predict(inputs)[0];
        let predicted_class = if prediction >= 0.5 { 1.0 } else { 0.0 };
        if predicted_class == *target {
            correct_predictions += 1;
        }
    }

    let accuracy = correct_predictions as f64 / test_features.len() as f64;
    println!("Accuracy on the testing set: {:.2}%", accuracy * 100.0);
}