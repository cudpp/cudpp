#ifndef SIMPLE_SAMPLE__RANDOM_NUMBERS_H
#define SIMPLE_SAMPLE__RANDOM_NUMBERS_H

//! Uses a Fisher-Yates shuffle to randomize the order of the input items.
void Shuffle(const unsigned  num_random_numbers,
                   unsigned *random_numbers);


//! Generate a set of random unique 32-bit numbers that aren't 0xffffffff.
void GenerateUniqueRandomNumbers(unsigned       *random_numbers,
                                 const unsigned  num_random_numbers,
                                 const unsigned  max_number = 0xfffffffe);

//! Replace some of the unique keys with copies of other keys in the array.
unsigned GenerateMultiples(const unsigned  num_random_numbers,
                                 float     chance_of_repeating,
                                 unsigned *random_numbers);

//! Generates a set of random queries from the given input.
void GenerateQueries(const unsigned  size,
                     const float     failure_rate,
                           unsigned *number_pool,
                           unsigned *queries);

#endif
