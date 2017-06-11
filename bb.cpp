#include <string>
#include "types.h"
#include "bb.h"
#include "movecalc.h"
#include "magics.h"

Bitboard RankBB[RANK_NB];
Bitboard FileBB[FILE_NB];
Bitboard SquareBB[SQUARE_NB + 1];
Bitboard BetweenBB[SQUARE_NB][SQUARE_NB];
Bitboard LineBB[SQUARE_NB][SQUARE_NB];
Bitboard AheadBB[SIDE_NB][SQUARE_NB];

Bitboard castlingPathBB[CASTLING_RIGHT_NB];
Bitboard castlingCheckPathBB[CASTLING_RIGHT_NB];

Bitboard StepAttacksBB[PIECETYPE_NB][SQUARE_NB];
Bitboard PseudoAttacksBB[PIECETYPE_NB][SQUARE_NB];

const Bitboard WHITE_SQS = 0xAA55AA55AA55AA55;
const Bitboard DARK_SQS = ~WHITE_SQS;

const Bitboard FULL_BOARD = 0xFFFFFFFFFFFFFFFF;
const Bitboard EMPTY_BOARD = 0ULL;

const Bitboard A_FILE = 0x8080808080808080;
const Bitboard B_FILE = 0x4040404040404040;
const Bitboard G_FILE = 0x0202020202020202;
const Bitboard H_FILE = 0x0101010101010101;

const std::string Bitboards::print_bb(Bitboard b) {
    std::string result = "";
    for (Square s = A8; int(s) >= 0 ; --s) {
        if (b & (1ULL << s)) {
            result += '1';
        } else {
            result += '0';
        }
        if (s % 8 == 0) {
            result += '\n';
        }
    }
    return result;
}

void Bitboards::init() {

    // Bottom up 0 - 7
    for (Rank r = RANK_1; r < RANK_NB; ++r) {
        RankBB[r] = 0xFFULL << 8 * r;
    }

    // A = 0, H = 7
    for (File f = FILE_A; f < FILE_NB; ++f) {
        FileBB[f] = 0x0101010101010101ULL << (7 - f);
    }

    for (Square s = H1; s < SQUARE_NB; ++s) {
        SquareBB[s] = 1ULL << s;
    }
    SquareBB[SQUARE_NONE] = 0;
    
    PieceType qpt = QUEEN;
    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (PieceType pt : { BISHOP, ROOK }) {

            Bitboard b = SquareBB[sq];
            Bitboard attacks;
            if (pt == BISHOP) {
                attacks = bishop_attack_calc(b, FULL_BOARD);
            } else {
                attacks = rook_attack_calc(b, FULL_BOARD);
            }
            PseudoAttacksBB[pt][sq] = attacks;
            PseudoAttacksBB[qpt][sq] |= attacks;
        }
    }

    for (Square sq1 = H1; sq1 < SQUARE_NB; ++sq1) {
        for (Square sq2 = H1; sq2 < SQUARE_NB; ++sq2) {
            for (PieceType pt : { BISHOP, ROOK }) {

                if (!(PseudoAttacksBB[pt][sq1] & SquareBB[sq2])) {
                    continue;
                }

                Bitboard attacks1, attacks2;

                if (pt == BISHOP) {
                    attacks1 = bishop_attack_calc(SquareBB[sq1], ~SquareBB[sq2]);
                    attacks2 = bishop_attack_calc(SquareBB[sq2], ~SquareBB[sq1]);
                } else {
                    attacks1 = rook_attack_calc(SquareBB[sq1], ~SquareBB[sq2]);
                    attacks2 = rook_attack_calc(SquareBB[sq2], ~SquareBB[sq1]);
                }
                BetweenBB[sq1][sq2] = attacks1 & attacks2;

                if (pt == BISHOP) {
                    attacks1 = bishop_attack_calc(SquareBB[sq1], FULL_BOARD);
                    attacks2 = bishop_attack_calc(SquareBB[sq2], FULL_BOARD);
                } else {
                    attacks1 = rook_attack_calc(SquareBB[sq1], FULL_BOARD);
                    attacks2 = rook_attack_calc(SquareBB[sq2], FULL_BOARD);
                }
                LineBB[sq1][sq2] = (attacks1 & attacks2)
                    | SquareBB[sq1] | SquareBB[sq2];
            }
        }
    }

    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (Side side : { WHITE, BLACK }) {
            Rank last_rank = Rank(side == WHITE ? 7 : 0);
            File file = file_of(sq);
            Bitboard last_sq = FileBB[file] & RankBB[last_rank];
            Square last_sq_ind = lsb(last_sq);
            AheadBB[side][sq] = BetweenBB[sq][last_sq_ind] | last_sq;
        }
    }

    for (Square sq = H1; sq < SQUARE_NB; ++sq) {
        for (PieceType pt : { KNIGHT, KING }) {
            if (pt == KNIGHT) {
                StepAttacksBB[pt][sq] = knight_attack_calc(SquareBB[sq]);
            } else {
                StepAttacksBB[pt][sq] = king_attack_calc(SquareBB[sq]);
            }
        }
    }

    castlingPathBB[WHITE_00_RIGHT] = G1 | F1;
    castlingPathBB[BLACK_00_RIGHT] = G8 | F8;
    castlingPathBB[WHITE_000_RIGHT] = B1 | C1 | D1;
    castlingPathBB[BLACK_000_RIGHT] = B8 | C8 | D8;

    castlingCheckPathBB[WHITE_00_RIGHT] = G1 | F1;
    castlingCheckPathBB[BLACK_00_RIGHT] = G8 | F8;
    castlingCheckPathBB[WHITE_000_RIGHT] = C1 | D1;
    castlingCheckPathBB[BLACK_000_RIGHT] = C8 | D8;
}

Bitboard shift_north(Bitboard b, Side side) {
    return side == WHITE ? b << 8 : b >> 8;
}

Bitboard shift_south(Bitboard b, Side side) {
    return side == WHITE ? b >> 8 : b << 8;
}

Bitboard north(Bitboard b) {
    return b << 8;
}

Bitboard south(Bitboard b) {
    return b >> 8;
}

Bitboard east(ULL b) {
    return b >> 1 & ~A_FILE;
}

Bitboard west(ULL b) {
    return b << 1 & ~H_FILE;
}

Bitboard north_east(ULL b) {
    return b << 7 & ~A_FILE;
}

Bitboard north_west(ULL b) {
    return b << 9 & ~H_FILE;
}

Bitboard south_east(ULL b) {
    return b >> 9 & ~A_FILE;
}

Bitboard south_west(ULL b) {
    return b >> 7 & ~H_FILE;
}
