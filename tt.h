#include <stdint.h>
#include "types.h"

class TTEntry {
public:
    Move move() const { return (Move)move16; }
    
private:
    uint16_t key16; 
    uint16_t move16; 
    uint16_t ration16; 
    uint8_t value8; 
    uint8_t static_eval8;
    uint8_t gen_bound8;
};
