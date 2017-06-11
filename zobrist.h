#ifndef ZOBRIST_GAURD
#define ZOBRIST_GAURD

#include "types.h"

namespace Zobrist {
    extern Key psqs[SQUARE_NB][PIECE_NB];
    extern Key ep_sqs[SQUARE_NB];
    extern Key side[SIDE_NB];
    extern Key castling[CASTLING_NB];

    void init();
}

#endif
