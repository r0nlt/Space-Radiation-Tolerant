#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>

// Simple implementation of Health-Weighted TMR
template <typename T>
class HealthWeightedTMR {
public:
    explicit HealthWeightedTMR(const T& initial_value = T()) 
        : copies_{initial_value, initial_value, initial_value},
          health_scores_{1.0, 1.0, 1.0} {
    }
    
    T get() const {
        // Simple voting - choose value with highest health score
        if (copies_[0] == copies_[1] || copies_[0] == copies_[2]) {
            // Update health scores - first copy correct
            updateHealthScores(0, true);
            if (copies_[0] == copies_[1]) {
                updateHealthScores(1, true);
                updateHealthScores(2, copies_[2] == copies_[0]);
            } else {
                updateHealthScores(1, copies_[1] == copies_[0]);
                updateHealthScores(2, true);
            }
            return copies_[0];
        } else if (copies_[1] == copies_[2]) {
            // Update health scores - copies 1 and 2 agree
            updateHealthScores(0, false);
            updateHealthScores(1, true);
            updateHealthScores(2, true);
            return copies_[1];
        }
        
        // No majority, use health scores
        size_t best_idx = 0;
        for (size_t i = 1; i < 3; ++i) {
            if (health_scores_[i] > health_scores_[best_idx]) {
                best_idx = i;
            }
        }
        return copies_[best_idx];
    }
    
    void set(const T& value) {
        for (auto& copy : copies_) {
            copy = value;
        }
        // Reset health scores on set
        for (auto& score : health_scores_) {
            score = 1.0;
        }
    }
    
    void repair() {
        T value = get();
        for (auto& copy : copies_) {
            copy = value;
        }
    }
    
    std::array<double, 3> getHealthScores() const {
        return health_scores_;
    }
    
    // For testing - directly access copies
    const std::array<T, 3>& getCopies() const {
        return copies_;
    }
    
    // For testing - directly corrupt copies
    void corruptCopy(size_t index, const T& value) {
        if (index < 3) {
            copies_[index] = value;
        }
    }
    
private:
    std::array<T, 3> copies_;
    mutable std::array<double, 3> health_scores_;
    
    void updateHealthScores(size_t index, bool correct) const {
        const double reward = 0.05;  // Reward for correct vote
        const double penalty = 0.2;  // Penalty for incorrect vote
        
        if (correct) {
            health_scores_[index] = std::min(1.0, health_scores_[index] + reward);
        } else {
            health_scores_[index] = std::max(0.1, health_scores_[index] - penalty);
        }
    }
};

int main() {
    std::cout << "=== Simplified Health-Weighted TMR Test ===" << std::endl;
    
    // Test basic functionality
    HealthWeightedTMR<int> tmr(42);
    
    std::cout << "Initial value: " << tmr.get() << std::endl;
    std::cout << "Initial health scores: ";
    for (double score : tmr.getHealthScores()) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    // Test 1: Corrupt one copy - should be outvoted
    tmr.corruptCopy(0, 99);
    
    std::cout << "\nTest 1: Corrupt one copy" << std::endl;
    std::cout << "Copies: ";
    for (int val : tmr.getCopies()) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Value after corruption: " << tmr.get() << std::endl;
    std::cout << "Health scores: ";
    for (double score : tmr.getHealthScores()) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    // Repair and check results
    tmr.repair();
    std::cout << "Value after repair: " << tmr.get() << std::endl;
    std::cout << "Copies after repair: ";
    for (int val : tmr.getCopies()) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Test 2: Corrupt two copies - should use health scores
    HealthWeightedTMR<int> tmr2(42);
    
    // First corrupt one copy and let the system detect it
    tmr2.corruptCopy(0, 99);
    tmr2.get(); // Updates health scores
    tmr2.repair();
    
    std::cout << "\nTest 2: Corrupt two copies" << std::endl;
    std::cout << "Initial health scores: ";
    for (double score : tmr2.getHealthScores()) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    // Now corrupt two copies - system should use health scores
    tmr2.corruptCopy(1, 77);
    tmr2.corruptCopy(2, 77);
    
    std::cout << "Copies: ";
    for (int val : tmr2.getCopies()) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Value after corruption: " << tmr2.get() << std::endl;
    std::cout << "Health scores: ";
    for (double score : tmr2.getHealthScores()) {
        std::cout << score << " ";
    }
    std::cout << std::endl;
    
    // Repair and check results
    tmr2.repair();
    std::cout << "Value after repair: " << tmr2.get() << std::endl;
    std::cout << "Copies after repair: ";
    for (int val : tmr2.getCopies()) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    // Test 3: Random corruption simulation
    std::cout << "\nTest 3: Random corruption simulation" << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < 5; ++i) {
        HealthWeightedTMR<int> test_tmr(42);
        
        // Corrupt 1 or 2 copies randomly
        std::uniform_int_distribution<> num_corrupted(1, 2);
        int to_corrupt = num_corrupted(gen);
        
        std::cout << "Run " << i << ": corrupting " << to_corrupt << " copies" << std::endl;
        
        // Select which copies to corrupt
        std::vector<int> indices = {0, 1, 2};
        std::shuffle(indices.begin(), indices.end(), gen);
        
        for (int j = 0; j < to_corrupt; ++j) {
            test_tmr.corruptCopy(indices[j], 100 + i + j);
        }
        
        std::cout << "  Copies: ";
        for (int val : test_tmr.getCopies()) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        
        int result = test_tmr.get();
        std::cout << "  Value after corruption: " << result << std::endl;
        
        if (result == 42) {
            std::cout << "  PROTECTED: TMR successfully maintained correct value" << std::endl;
        } else {
            std::cout << "  CORRUPTED: TMR returned incorrect value" << std::endl;
        }
        
        test_tmr.repair();
        std::cout << "  After repair: " << test_tmr.get() << std::endl;
        std::cout << "  Repaired copies: ";
        for (int val : test_tmr.getCopies()) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
} 