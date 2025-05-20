#include "include/rad_ml/neural/galois_field.hpp"

#include <iostream>

int main()
{
    std::cout << "Starting GF256 test..." << std::endl;

    // Just initializing the field - this is where it crashes
    std::cout << "Initializing GF256..." << std::endl;
    rad_ml::neural::GF256 field;

    std::cout << "GF256 initialized." << std::endl;

    // If we got here, initialization works
    std::cout << "Testing addition: 5 + 10 = " << (int)field.add(5, 10) << std::endl;

    return 0;
}
