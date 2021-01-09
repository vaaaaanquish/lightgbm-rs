extern crate lightgbm;
extern crate csv;
extern crate itertools;

use itertools::zip;
use lightgbm::{Dataset, Booster};

fn main() -> std::io::Result<()> {
    // let feature = vec![vec![1.0, 0.1, 0.2, 0.1],
    //                vec![0.7, 0.4, 0.5, 0.1],
    //                vec![0.9, 0.8, 0.5, 0.1],
    //                vec![0.2, 0.2, 0.8, 0.7],
    //                vec![0.1, 0.7, 1.0, 0.9]];
    // let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    // let train_dataset = Dataset::from_mat(feature, label).unwrap();

    let train_dataset = Dataset::from_file("../lightgbm-sys/lightgbm/examples/binary_classification/binary.train".to_string()).unwrap();

    let mut rdr = csv::ReaderBuilder::new().has_headers(false).delimiter(b'\t').from_path("../lightgbm-sys/lightgbm/examples/binary_classification/binary.test")?;
    let mut test_labels: Vec<i8> = Vec::new();
    let mut test_feature: Vec<Vec<f32>> = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let label = record[0].parse::<i8>().unwrap();
        let feature: Vec<f32> = record.iter().map(|x| x.parse::<f32>().unwrap()).collect::<Vec<f32>>()[1..].to_vec();
        test_labels.push(label);
        test_feature.push(feature);
    }

    let booster = Booster::train(train_dataset).unwrap();
    let result = booster.predict(test_feature).unwrap();

    let mut tp = 0;
    for (label, pred) in zip(&test_labels, &result){
        if label == &(1 as i8) && pred > &(0.5 as f64) {
            tp = tp + 1;
        } else if label == &(0 as i8) && pred <= &(0.5 as f64) {
            tp = tp + 1;
        }
    }
    println!("{} / {}", &tp, result.len());
    Ok(())
}
