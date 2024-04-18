use pyo3::prelude::*;
use std::collections::HashMap;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::ThreadRng;
use rand::thread_rng;
use rayon::prelude::*;
use ndarray::prelude::*;

fn sample_index_from_weights(weights: Vec<f64>) -> Option<usize> {
    let mut rng = thread_rng();
    sample_index_from_weights_with_rng(&mut rng, weights)
}

fn sample_index_from_weights_with_rng(rng: &mut ThreadRng, weights: Vec<f64>) -> Option<usize> {
    if weights.iter().any(|&w| w < 0.0) {
        return None; // Return None if any weight is negative
    }

    let dist = WeightedIndex::new(&weights).ok()?;
    Some(dist.sample(rng))
}

struct GameResults {
    home_win: bool,
    away_win: bool,
    home_runs: i8,
    away_runs: i8,
    margin:i8,
    nrfi:bool,
}


fn scale_vectors_by_ratio_f64(a: &Vec<f64>, b: &Vec<f64>, c: &Vec<f64>,ratio: (f64, f64, f64)) -> Vec<f64> {
    let sum_a: f64 = a.iter().sum();
    let sum_b: f64 = b.iter().sum();
    let sum_c: f64 = c.iter().sum();

    // Calculate the combined target sums using the ratio and sums
    let total_ratio = ratio.0 + ratio.1 + ratio.2;
    let max_ratio_sum = sum_a * ratio.0 + sum_b * ratio.1 + sum_c * ratio.2;

    // Check for zero in total ratio to avoid division by zero
    let target_sum_a = if total_ratio != 0.0 { max_ratio_sum * (ratio.0 / total_ratio) } else { 0.0 };
    let target_sum_b = if total_ratio != 0.0 { max_ratio_sum * (ratio.1 / total_ratio) } else { 0.0 };
    let target_sum_c = if total_ratio != 0.0 { max_ratio_sum * (ratio.2 / total_ratio) } else { 0.0 };

    // Check for zero in original sums to avoid division by zero
    let scale_a = if sum_a != 0.0 { target_sum_a / sum_a } else { 0.0 };
    let scale_b = if sum_b != 0.0 { target_sum_b / sum_b } else { 0.0 };
    let scale_c = if sum_c != 0.0 { target_sum_c / sum_c } else { 0.0 };

    // Scale the vectors
    let scaled_a: Vec<f64> = a.iter().map(|&x| x * scale_a).collect();
    let scaled_b: Vec<f64> = b.iter().map(|&x| x * scale_b).collect();
    let scaled_c: Vec<f64> = c.iter().map(|&x| x * scale_c).collect();

    // Sum the scaled vectors
    scaled_a.iter().zip(scaled_b.iter()).zip(scaled_c.iter())
        .map(|((&x, &y), &z)| x + y + z)
        .collect()
}



#[pyfunction] 
fn markov_sim(games: i32,year:i16,transition:Vec<Vec<f64>>,batter_trans:HashMap<String,Vec<Vec<f64>>>,pitcher_trans:Vec<Vec<Vec<f64>>>,lineups:Vec<Vec<String>>) -> (i32,i32,f64,f64,f64,i32,Vec<f64>,Vec<f64>) {

    let game_results: Vec<GameResults> = (0..games).into_par_iter()
    .map(|_i| {
        let mut inning:f32 = 0.0;
        let mut home_runs:i8 = 0;
        let mut away_runs:i8 = 0;
        let mut nrfi:bool = false;
        let mut home_lineup_idx:i64 = 0;
        let mut away_lineup_idx:i64 = 0;
        let mut batter:String;
        let mut pitcher:i8;
        let home_win:bool;
        let away_win:bool;
        'current_game: loop {
            let mut state:i8;
            if inning >= 10.0 && year > 2019{
                state = 3;
            } else {
                state = 1;
            }

            while state <= 24{
                if inning % 1.0 == 0.0{

                    batter = lineups[1][(away_lineup_idx % 9) as usize].to_string();
                    pitcher = 0;
                } else {

                    batter = lineups[0][(home_lineup_idx % 9) as usize].to_string();
                    pitcher = 1;
                }
                let transition_matrix:Vec<Vec<f64>> = batter_trans[&batter].to_vec();

                let pitcher_transition_matrix:&Vec<Vec<f64>> = &pitcher_trans[pitcher as usize];

                let pitcher_row:&Vec<f64> = &pitcher_transition_matrix[state as usize - 1];

                let trans_row:Vec<f64> = transition[state as usize - 1].clone();

                let batter_row = Array::from_vec(transition_matrix[state as usize - 1].clone());

                let final_row:Vec<f64> = scale_vectors_by_ratio_f64(&batter_row.to_vec(),&trans_row.to_vec(),&pitcher_row.to_vec(),(25.0,1.0,150.0));

                let post_state:i8 = sample_index_from_weights(final_row.to_vec()).unwrap() as i8 + 1;
                
                let run_vec:Vec<i8> = match state {
                    1 => vec![
                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    2 => vec![
                        2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    3 => vec![
                        2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    4 => vec![
                        2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    5 => vec![
                        3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    6 => vec![
                        3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    7 => vec![
                        3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    8 => vec![
                        4, 3, 3, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    9 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ],
                    10 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    11 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    12 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    13 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    14 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    15 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    16 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3, 3, 2, 2, 2, 1, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2
                    ],
                    17 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1
                    ],
                    18 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    19 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    20 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 1
                    ],
                    21 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2
                    ],
                    22 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2
                    ],
                    23 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2
                    ],
                    24 => vec![
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 3, 3, 2, 2, 2, 1, 0, 1, 2, 3
                        ],
                    _ => vec![0],
                };

                let runs_scored:i8 = run_vec[post_state as usize - 1];

                if inning % 1.0 == 0.0{
                    away_runs += runs_scored;
                    away_lineup_idx += 1;
                } else {
                    home_runs += runs_scored;
                    home_lineup_idx += 1;
                }

                if (home_runs > away_runs && inning > 9.0) || (home_runs < away_runs && inning > 9.0 && post_state > 24){
                    break 'current_game;
                }
                state = post_state;
            }
            if inning == 0.5{
                if (home_runs + away_runs) > 0{
                    nrfi = false;
                } else{
                    nrfi = true;
                }
            }
            inning += 0.5;
        }
        if away_runs > home_runs{
            away_win = true;
            home_win = false;
        } else {
            home_win = true;
            away_win = false;
        }
        let margin:i8 = away_runs - home_runs;
        GameResults { home_win: home_win, away_win: away_win, home_runs: home_runs, away_runs: away_runs,margin:margin,nrfi:nrfi }
    }).collect();
    // Aggregate the results
    let home_wins:i32 = game_results.iter().filter(|result| result.home_win).count() as i32;
    let away_wins:i32 = game_results.iter().filter(|result| result.away_win).count() as i32;
    let home_runs:f64 = game_results.iter().map(|result| result.home_runs as f64).sum::<f64>();
    let away_runs:f64 = game_results.iter().map(|result| result.away_runs as f64).sum::<f64>();
    let margins:f64 = game_results.iter().map(|result| result.margin as f64).sum::<f64>() / games as f64;
    let nrfis:i32 = game_results.iter().filter(|result| result.nrfi).count() as i32;

    let f_home_runs:Vec<f64> = game_results.iter().map(|result| result.home_runs as f64).collect();
    let f_away_runs:Vec<f64> = game_results.iter().map(|result| result.away_runs as f64).collect();

    (home_wins,away_wins,home_runs,away_runs,margins,nrfis,f_home_runs,f_away_runs)
}

/// A Python module implemented in Rust.
#[pymodule]
fn markov_simulations(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(markov_sim, m)?)?;
    Ok(())
}
