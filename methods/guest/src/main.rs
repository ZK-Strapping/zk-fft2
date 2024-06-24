#![no_main]
// If you want to try std support, also update the guest Cargo.toml file
// #![no_std] // std support is experimental

use risc0_zkvm::guest::env;
use zk_fft_core::*;

risc0_zkvm::guest::entry!(main);

const TRUNC_PRECISION: i32 = 3;

pub fn main() {
    let input: CircuitInput = env::read();

    let output = poly_mul(input.n, &input.ax, input.m, &input.bx);

    // write to the journal
    env::commit(&CircuitJournal { input, output });
}

use std::ops::*;

fn is_power_of_two(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

fn bit_reverse_value(mut x: usize, mut num_bits: usize) -> usize {
    let mut rev_x = 0;
    while num_bits > 0 {
        rev_x <<= 1;
        rev_x |= x & 1;
        x >>= 1;
        num_bits -= 1;
    }
    rev_x
}

fn get_bit_reverse_array(a: &[Complex], n: usize) -> Vec<Complex> {
    assert!(is_power_of_two(n));

    let mut r = vec![Complex(0.0, 0.0); n];
    for i in 0..n {
        let bit_reverse_i = bit_reverse_value(i, (n as f64).log2() as usize);
        r[bit_reverse_i] = a[i];
    }

    r
}

fn find_mth_root_of_unity(m: usize) -> Complex {
    let tmp = 2.0 * std::f64::consts::PI / m as f64;
    let zeta = Complex(tmp.cos(), tmp.sin());
    zeta
}

fn get_psi_powers(m: usize) -> Vec<Complex> {
    let psi = find_mth_root_of_unity(m);
    let mut psi_powers = Vec::with_capacity(m);
    let mut tmp = Complex(1.0, 0.0);
    for i in 0..m {
        psi_powers.push(tmp);
        tmp = tmp * psi;
    }
    psi_powers.push(psi_powers[0]);

    psi_powers
}

fn get_rot_group(n_half: usize, m: usize) -> Vec<usize> {
    let mut p = 1;
    let mut rot_group = Vec::with_capacity(n_half);
    for _ in 0..n_half {
        rot_group.push(p);
        p *= 5;
        p %= m;
    }

    rot_group
}

fn special_fft(a: &Vec<Complex>, n: usize, m: usize) -> Vec<Complex> {
    assert_eq!(a.len(), n);
    assert!(is_power_of_two(n));

    let mut a = get_bit_reverse_array(a, n);
    let psi_powers = get_psi_powers(m);
    let rot_group = get_rot_group(m >> 2, m);

    let mut length_n = 2;
    while length_n <= n {
        for i in (0..n).step_by(length_n) {
            let lenh = length_n >> 1;
            let lenq = length_n << 2;
            let gap = m / lenq;
            for j in (0..lenh).step_by(1) {
                let idx = (rot_group[j] % lenq) * gap;
                let u = a[i + j];
                let mut v = a[i + j + lenh];
                v = v * psi_powers[idx];
                a[i + j] = u + v;
                a[i + j + lenh] = u - v;
            }
        }
        length_n *= 2;
    }

    a
}

fn special_ifft(a: &Vec<Complex>, n: usize, m: usize) -> Vec<Complex> {
    assert_eq!(a.len(), n);
    assert!(is_power_of_two(n));

    let mut a = a.to_vec();

    let mut length_n = n;
    let psi_powers = get_psi_powers(m);
    let rot_group = get_rot_group(m >> 2, m);

    while length_n >= 1 {
        for i in (0..n).step_by(length_n) {
            let lenh = length_n >> 1;
            let lenq = length_n << 2;
            let gap = m / lenq;
            for j in (0..lenh) {
                let idx = (lenq - (rot_group[j] % lenq)) * gap;
                let u = a[i + j] + a[i + j + lenh];
                let mut v = a[i + j] - a[i + j + lenh];
                v = v * psi_powers[idx];
                a[i + j] = u;
                a[i + j + lenh] = v;
            }
        }
        length_n >>= 1;
    }

    a = get_bit_reverse_array(&a, n);

    // Multiply by 1/n and return
    a.iter().map(|&x| Complex(x.0 / (n as f64), x.1 / (n as f64))).collect()
}


fn poly_mul(n: usize, x: &Vec<f64>, m: usize, y: &Vec<f64>) -> Vec<f64> {
    assert_eq!(n, x.len());
    assert_eq!(m, y.len());

    let mut x: Vec<Complex> = x.iter().map(|xi| Complex(*xi, 0.)).collect();
    let mut y: Vec<Complex> = y.iter().map(|yi| Complex(*yi, 0.)).collect();

    let len = (n + m).next_power_of_two();
    x.resize(len, Complex(0., 0.));
    y.resize(len, Complex(0., 0.));

    let mut xx = special_fft(&mut x, len, len * 4);
    let mut yy = special_fft(&mut y, len, len * 4);

    x.iter_mut().zip(&y).for_each(|(xi, &yi)| *xi = *xi * yi);
    xx.iter_mut().zip(&yy).for_each(|(xxi, &yyi)| *xxi = *xxi * yyi);

    let mut z = special_ifft(&mut xx, len, len * 4);

    for zi in &z {
        println!("{} {}", zi.0, zi.1);
    }

    z.iter().map(|zi| corr(zi.0)).collect()
}

#[derive(Clone, Copy, PartialEq)]
struct Complex(f64, f64);

impl Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Complex(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Complex(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Complex(self.0 - rhs.0, self.1 - rhs.1)
    }
}

fn corr(val: f64) -> f64 {
    let mut val2 = (val * 10.0f64.powi(TRUNC_PRECISION)).round();
    if val2.abs() <= f64::EPSILON {
        val2 = 0.0;
    }
    val2 * 10.0f64.powi(-TRUNC_PRECISION)
}
