extern crate csv;
extern crate itertools;
extern crate lightgbm;
extern crate serde_json;

use itertools::zip;
use lightgbm::{Booster, Dataset};
use serde_json::json;

fn load_file(file_path: &str) -> (Vec<Vec<f64>>, Vec<f32>) {
    let rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_path(file_path);
    let mut labels: Vec<f32> = Vec::new();
    let mut features: Vec<Vec<f64>> = Vec::new();
    for result in rdr.unwrap().records() {
        let record = result.unwrap();
        let label = record[0].parse::<f32>().unwrap();
        let feature: Vec<f64> = record
            .iter()
            .map(|x| x.parse::<f64>().unwrap())
            .collect::<Vec<f64>>()[1..]
            .to_vec();
        labels.push(label);
        features.push(feature);
    }
    (features, labels)
}

fn argmax<T: PartialOrd>(xs: &[T]) -> usize {
    if xs.len() == 1 {
        0
    } else {
        let mut maxval = &xs[0];
        let mut max_ixs: Vec<usize> = vec![0];
        for (i, x) in xs.iter().enumerate().skip(1) {
            if x > maxval {
                maxval = x;
                max_ixs = vec![i];
            } else if x == maxval {
                max_ixs.push(i);
            }
        }
        max_ixs[0]
    }
}

fn main() -> std::io::Result<()> {
    let (train_features, train_labels) = load_file(
        "../../lightgbm-sys/lightgbm/examples/multiclass_classification/multiclass.train",
    );
    let (test_features, test_labels) =
        load_file("../../lightgbm-sys/lightgbm/examples/multiclass_classification/multiclass.test");
    let train_dataset = Dataset::from_mat(train_features, train_labels).unwrap();

    let params = json! {
        {
            "num_iterations": 100,
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 5,
        }
    };

    let booster = Booster::train(train_dataset, &params).unwrap();
    let result = booster.predict(test_features).unwrap();

    let mut tp = 0;
    for (label, pred) in zip(&test_labels, &result) {
        let argmax_pred = argmax(&pred);
        if *label == argmax_pred as f32 {
            tp = tp + 1;
        }
        println!("{}, {}, {:?}", label, argmax_pred, &pred);
    }
    println!("{} / {}", &tp, result.len());
    Ok(())
}
