#include <random>
#include "zobrist.h"

namespace Zobrist {
    Key psqs[SQUARE_NB][PIECE_NB];
    Key ep_sqs[SQUARE_NB];
    Key side[SIDE_NB];
    Key castling[CASTLING_NB];

    void init();
}

void Zobrist::init() {
    std::mt19937_64 gen(53820873);

    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (Piece pt : Pieces) {
            Zobrist::psqs[sq][pt] = gen();
        }

        Zobrist::ep_sqs[sq] = gen();
    }

    Zobrist::side[WHITE] = gen();
    Zobrist::side[BLACK] = gen();

    Zobrist::castling[WHITE_00] = gen();
    Zobrist::castling[WHITE_000] = gen();
    Zobrist::castling[BLACK_00] = gen();
    Zobrist::castling[BLACK_000] = gen();
}
