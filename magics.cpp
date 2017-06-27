#include <iostream>
#include "bb.h"
#include "movecalc.h"
#include "magics.h"

Bitboard MagicMasksBB[PIECETYPE_NB][SQUARE_NB];
Bitboard MagicAttacksBBB[SQUARE_NB][512];
Bitboard MagicAttacksRBB[SQUARE_NB][4096];

// Magic bitboard helpers
Bitboard edge_mask(Square sq) {
    Bitboard edges = 0;
    int rk = static_cast<int>(sq) >> 3;
    int fl = 7 - (static_cast<int>(sq) & 7);
    if (rk != 0) edges |= RankBB[0];
    if (rk != 7) edges |= RankBB[7];
    if (fl != 0) edges |= FileBB[0];
    if (fl != 7) edges |= FileBB[7];
    return edges;
}

Bitboard rook_mask(Square sq) {
    Bitboard attacks = PseudoAttacksBB[ROOK][sq];
    Bitboard edges = edge_mask(sq);
    return attacks & ~edges;
}

Bitboard bishop_mask(Square sq) {
    Bitboard attacks = PseudoAttacksBB[BISHOP][sq];
    Bitboard edges = edge_mask(sq);
    return attacks & ~edges;
}

Bitboard index_to_occupation(Bitboard index, int bits, Bitboard mask) {
    Bitboard j, result;
    result = 0;
    for (int i = 0; i < bits; ++i) {
        j = ls1b(mask);
        mask = reset_ls1b(mask);
        if (index & (1ULL << i)) {
            result |= j;
        }
    }
    return result;
}

void Magics::init() {
    PieceType qpt = QUEEN;
    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (PieceType pt : { BISHOP, ROOK }) {

            Bitboard mask;
            if (pt == BISHOP) {
                mask = bishop_mask(sq);
            } else {
                mask = rook_mask(sq);
            }
            MagicMasksBB[pt][sq] = mask;
            MagicMasksBB[qpt][sq] |= mask;
        }
    }

    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (PieceType pt : { BISHOP, ROOK }) {
            Bitboard magic = MagicNumbers[pt][sq];
            Bitboard mask = MagicMasksBB[pt][sq];
            int bits = MaskBitLength[pt][sq];
            int range = 1ULL << bits;
            for (int index = 0; index < range; ++index) {
                Bitboard occupation = index_to_occupation(index, bits, mask);
                Bitboard free = ~occupation;
                Bitboard index_hash = (occupation * magic) >> (64 - bits);
                if (pt == BISHOP) {
                    MagicAttacksBBB[sq][index_hash] = bishop_attack_calc(SquareBB[sq], free);
                } else {
                    MagicAttacksRBB[sq][index_hash] = rook_attack_calc(SquareBB[sq], free);
                }
            }
        }
    }
}
