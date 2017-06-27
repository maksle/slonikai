#include "material.h"

int game_phase(const int counts[PIECE_NB]) {
    int value = 0;
    for (Piece p = W_PAWN; p < PIECE_NB; ++p) {
        PieceType pt = base_type(p);
        value += counts[p] * PhaseWeight[pt];
    }
    return value;
}

int scale_phase(int mg, int eg, int phase) {
    int delta = mg - eg;
    return eg + delta * phase / MAX_PHASE;
}

Value material_bootstrap(const int counts[PIECE_NB]) {
    int phase = game_phase(counts);
    if (phase > MAX_PHASE)
        phase = MAX_PHASE;
    
    int ret = 0;
    ret = counts[W_PAWN] * scale_phase(PawnValueMg, PawnValueEg, phase);
    ret += counts[W_KNIGHT] * scale_phase(KnightValueMg, KnightValueEg, phase);
    ret += counts[W_BISHOP] * scale_phase(BishopValueMg, BishopValueEg, phase);
    ret += counts[W_ROOK] * scale_phase(RookValueMg, RookValueEg, phase);
    ret += counts[W_QUEEN] * scale_phase(QueenValueMg, QueenValueEg, phase);
    ret -= counts[B_PAWN] * scale_phase(PawnValueMg, PawnValueEg, phase);
    ret -= counts[B_KNIGHT] * scale_phase(KnightValueMg, KnightValueEg, phase);
    ret -= counts[B_BISHOP] * scale_phase(BishopValueMg, BishopValueEg, phase);
    ret -= counts[B_ROOK] * scale_phase(RookValueMg, RookValueEg, phase);
    ret -= counts[B_QUEEN] * scale_phase(QueenValueMg, QueenValueEg, phase);

    return Value(ret);
}
