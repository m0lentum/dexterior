use itertools::{izip, Itertools};

/// Iterator over permutations of a range of indices.
pub struct Permutations {
    num_indices: usize,
    perm_indices: Vec<usize>,
    signs: Vec<i8>,
}

/// A single permutation of a range of indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Permutation<'a> {
    /// The indices ordered according to the permutation.
    pub indices: &'a [usize],
    /// The sign of the permutation
    /// (i.e. parity of the number of swaps required to create it from the original order).
    /// +1 for even, -1 for odd.
    pub sign: i8,
}

impl Permutations {
    pub fn new(num_indices: usize) -> Self {
        assert!(
            num_indices > 0,
            "no use taking a permutation of no elements"
        );

        if num_indices == 1 {
            return Self {
                num_indices,
                perm_indices: vec![0],
                signs: vec![1],
            };
        }

        // recursively compute permutations by inserting the largest index
        // into every slot in a set of permutations of 1 fewer integers.
        // sign is also computed at the same time
        // by keeping track of the number of swaps required to create each permutation
        let sub_perms = Self::new(num_indices - 1);
        let new_idx = num_indices - 1;

        let mut perm_indices = Vec::new();
        let mut signs = Vec::new();
        for sub_perm in sub_perms.iter() {
            for i in 0..num_indices {
                let insert_idx = num_indices - i - 1;
                perm_indices.extend_from_slice(&sub_perm.indices[0..insert_idx]);
                perm_indices.push(new_idx);
                perm_indices.extend_from_slice(&sub_perm.indices[insert_idx..num_indices - 1]);

                let sign = if i % 2 == 1 {
                    -sub_perm.sign
                } else {
                    sub_perm.sign
                };
                signs.push(sign);
            }
        }

        Self {
            num_indices,
            perm_indices,
            signs,
        }
    }

    /// Iterate over the generated permutations.
    pub fn iter(&self) -> impl '_ + Iterator<Item = Permutation<'_>> {
        izip!(
            self.perm_indices.chunks_exact(self.num_indices),
            &self.signs
        )
        .map(|(indices, &sign)| Permutation { indices, sign })
    }
}

/// Check the parity of a permutation of COUNT integers
/// which are not necessarily from the range 0..COUNT.
/// Returns 1 if positive, -1 if negative.
pub fn get_parity(perm: &[usize]) -> i8 {
    // compute the number of distinct "cycles" in the permutation.
    // a cycle is a sequence where we take a permuted index
    // and look at the element at that index,
    // repeating until we arrive at the original index.
    // if the indices are in sorted order, each index is a cycle of one element,
    // but if we've made some swaps there will be cycles of multiple elements.
    // each swap changes the number of cycles by exactly 1,
    // so we can get the parity
    // by checking if the number of cycles differs from COUNT by an even number.

    // ..but first we need to transform the permutation to one of the integers 0..len
    // so that we can work with it. we're dealing with low dimensions here,
    // so do this by checking against a sorted list of the elements
    let mut sorted = perm.iter().collect_vec();
    sorted.sort();
    let as_index = |perm_elem| sorted.iter().position(|&&x| x == perm_elem).unwrap();

    let mut cycle_count = 0;

    'outer: for i in 0..perm.len() {
        let mut cycle_idx = as_index(perm[i]);
        while cycle_idx != i {
            if cycle_idx < i {
                // this cycle was already seen
                continue 'outer;
            }
            cycle_idx = as_index(perm[cycle_idx]);
        }
        cycle_count += 1;
    }

    if (perm.len() - cycle_count) % 2 == 0 {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that permutations are generated correctly.
    #[test]
    fn permutations() {
        // just check a few hand-computed values.
        // we could make a more thorough test
        // e.g. by checking against the known solution in `itertools`,
        // but we won't get checking of signs from that without some extra effort
        // to compute the sign of a permutation after the fact

        let p2_ref = [([0, 1], 1), ([1, 0], -1)];
        let p2_computed = Permutations::new(2);

        itertools::assert_equal(
            p2_computed.iter(),
            p2_ref.iter().map(|(i, s)| Permutation {
                indices: i,
                sign: *s,
            }),
        );
        // also check that the parity computation works for these
        itertools::assert_equal(
            p2_computed.iter().map(|p| p.sign),
            p2_computed.iter().map(|p| get_parity(p.indices)),
        );

        let p3_ref = [
            ([0, 1, 2], 1),
            ([0, 2, 1], -1),
            ([2, 0, 1], 1),
            ([1, 0, 2], -1),
            ([1, 2, 0], 1),
            ([2, 1, 0], -1),
        ];
        let p3_computed = Permutations::new(3);

        itertools::assert_equal(
            p3_computed.iter(),
            p3_ref.iter().map(|(i, s)| Permutation {
                indices: i,
                sign: *s,
            }),
        );
        itertools::assert_equal(
            p3_computed.iter().map(|p| p.sign),
            p3_computed.iter().map(|p| get_parity(p.indices)),
        );

        let p4_ref = [
            ([0, 1, 2, 3], 1),
            ([0, 1, 3, 2], -1),
            ([0, 3, 1, 2], 1),
            ([3, 0, 1, 2], -1),
            ([0, 2, 1, 3], -1),
            ([0, 2, 3, 1], 1),
            ([0, 3, 2, 1], -1),
            ([3, 0, 2, 1], 1),
            ([2, 0, 1, 3], 1),
            ([2, 0, 3, 1], -1),
            ([2, 3, 0, 1], 1),
            ([3, 2, 0, 1], -1),
            ([1, 0, 2, 3], -1),
            ([1, 0, 3, 2], 1),
            ([1, 3, 0, 2], -1),
            ([3, 1, 0, 2], 1),
            ([1, 2, 0, 3], 1),
            ([1, 2, 3, 0], -1),
            ([1, 3, 2, 0], 1),
            ([3, 1, 2, 0], -1),
            ([2, 1, 0, 3], -1),
            ([2, 1, 3, 0], 1),
            ([2, 3, 1, 0], -1),
            ([3, 2, 1, 0], 1),
        ];
        let p4_computed = Permutations::new(4);

        itertools::assert_equal(
            p4_computed.iter(),
            p4_ref.iter().map(|(i, s)| Permutation {
                indices: i,
                sign: *s,
            }),
        );
        itertools::assert_equal(
            p4_computed.iter().map(|p| p.sign),
            p4_computed.iter().map(|p| get_parity(p.indices)),
        );
    }

    /// Check that get_parity computes things correctly
    /// also when the permutation isn't of a range 0..len.
    #[test]
    fn parities_with_gaps() {
        let checks = [
            (vec![2, 5, 8, 10], 1),
            (vec![3, 1, 5], -1),
            (vec![2, 0, 1], 1),
            (vec![8, 9, 7, 6, 15], -1),
            (vec![8, 7, 9, 6, 15], 1),
            (vec![3, 2], -1),
            (vec![5], 1),
            (vec![2, 3], 1),
        ];

        for (indices, parity) in checks {
            assert_eq!(
                parity,
                get_parity(&indices),
                "Permutation {indices:?} got wrong parity"
            );
        }
    }
}
