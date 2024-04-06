use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TitanicPassenger {
    #[serde(rename = "PassengerId")]
    passenger_id: u32,
    #[serde(rename = "Survived")]
    survived: f64,
    #[serde(rename = "Pclass")]
    pclass: f64,
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Sex")]
    sex: String,
    #[serde(rename = "Age")]
    age: Option<f64>,
    #[serde(rename = "SibSp")]
    sib_sp: f64,
    #[serde(rename = "Parch")]
    parch: f64,
    #[serde(rename = "Ticket")]
    ticket: String,
    #[serde(rename = "Fare")]
    fare: f64,
    #[serde(rename = "Cabin")]
    cabin: Option<String>,
    #[serde(rename = "Embarked")]
    embarked: String,
}

pub fn load_titanic_dataset(file_path: &str) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Load the Titanic dataset from a CSV file
    let mut reader = ReaderBuilder::new()
        .from_path(file_path)
        .expect("Failed to open the CSV file");
    let mut data = Vec::new();

    for result in reader.deserialize() {
        let record: TitanicPassenger = result.expect("Failed to deserialize a record");
        let sex = if record.sex == "male" { 0.0 } else { 1.0 };
        let age = record.age.unwrap_or(0.0);
        let sib_sp = record.sib_sp;
        let parch = record.parch;
        let fare = record.fare;
        let embarked = match record.embarked.as_str() {
            "S" => 0.0,
            "C" => 1.0,
            "Q" => 2.0,
            _ => 0.0,
        };
        let features = vec![record.pclass, sex, age, sib_sp, parch, fare, embarked];
        data.push((features, record.survived));
    }

    // Split the data into features and targets
    let (features, targets): (Vec<_>, Vec<_>) = data.into_iter().unzip();
    (features, targets)
}