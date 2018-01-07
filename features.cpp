#include "types.h"
#include "bb.h"
#include "position.h"
#include "features.h"
#include <iostream>

namespace {
    
    float normalize_coord(int v) {
        return static_cast<float>(v) * 0.125; // div 8
    }

    void push_coords(Square sq, Side side, std::vector<float>& f) {
        int x = sq == SQUARE_NONE ? 0 : static_cast<int>(file_of(sq)) + 1;
        int y = sq == SQUARE_NONE ? 0 : static_cast<int>(rank_of(sq)) + 1;
        int y_alt = sq == SQUARE_NONE ? 0
            : static_cast<int>(relative_rank(sq, side)) + 1;
        f.push_back(normalize_coord(x));
        f.push_back(normalize_coord(y));
        f.push_back(normalize_coord(y_alt));
    }

    void push_coords(Bitboard b, Side side, std::vector<float>& f) {
        assert(!b || !reset_ls1b(b));
        if (!b)
            return push_coords(SQUARE_NONE, side, f);
        return push_coords(pop_lsb(b), side, f);
    }
} // namespace

void FeatureExtractor::set_position(const Position& position) {
    pos = &position;
}

std::vector<std::vector<float> > FeatureExtractor::extract() {
    std::vector<float> g;       // global
    std::vector<float> p;       // pawn-centric
    std::vector<float> np;      // piece-centric
    std::vector<float> s;       // square-centric
    
    set_attacks();
    
    g.push_back(pos->side_to_move());
    castling_rights(g);
    g.push_back(popcount(pos->checkers()));
    counts_and_values<WHITE>(g);
    counts_and_values<BLACK>(g);
    
    pawn_exists<WHITE>(p);
    pawn_exists<BLACK>(p);
    pawn_counts<WHITE>(p);
    pawn_counts<BLACK>(p);

    BRN_pairs<KNIGHT, WHITE>(np);
    BRN_pairs<KNIGHT, BLACK>(np);
    BRN_pairs<BISHOP, WHITE>(np);
    BRN_pairs<BISHOP, BLACK>(np);
    BRN_pairs<ROOK, WHITE>(np);
    BRN_pairs<ROOK, BLACK>(np);
    queens<WHITE>(np);
    queens<BLACK>(np);
    king<WHITE>(np);
    king<BLACK>(np);
    
    attack_count<KNIGHT, WHITE>(s);
    attack_count<KNIGHT, BLACK>(s);
    attack_count<BISHOP, WHITE>(s);
    attack_count<BISHOP, BLACK>(s);
    attack_count<ROOK, WHITE>(s);
    attack_count<ROOK, BLACK>(s);
    attack_count<QUEEN, WHITE>(s);
    attack_count<QUEEN, BLACK>(s);
    attack_count<KING, WHITE>(s);
    attack_count<KING, BLACK>(s);
    lowest_attackers<WHITE>(s);
    lowest_attackers<BLACK>(s);

    return std::vector<std::vector<float> > { g, p, np, s};
}

void FeatureExtractor::castling_rights(std::vector<float>& f) {
    f.push_back(static_cast<bool>(pos->castling_rights(WHITE_00_RIGHT)));
    f.push_back(static_cast<bool>(pos->castling_rights(WHITE_000_RIGHT)));
    f.push_back(static_cast<bool>(pos->castling_rights(WHITE_CASTLING_ANY)));
    f.push_back(static_cast<bool>(pos->castling_rights(BLACK_00_RIGHT)));
    f.push_back(static_cast<bool>(pos->castling_rights(BLACK_000_RIGHT)));
    f.push_back(static_cast<bool>(pos->castling_rights(BLACK_CASTLING_ANY)));
}

void FeatureExtractor::set_attacks()
{
    Bitboard b;
    for (Square sq = H1; sq <= A8; ++sq) {
        for (Side side : { WHITE, BLACK }) {
            b = pos->attackers(sq, side);
            if (b)
            {
                all_attacks[side] |= sq;
                if (reset_ls1b(b))
                    double_attacks[side] |= sq;

                if (b & pos->pieces(side, PAWN))
                {
                    piece_attacks[make_piece(PAWN, side)] |= b;
                    maybe_unsafe_for[~side][PAWN] |= b;
                    unsafe_for[~side][KNIGHT] |= b;
                    unsafe_for[~side][BISHOP] |= b;
                    unsafe_for[~side][ROOK] |= b;
                    unsafe_for[~side][QUEEN] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = PAWN;
                }
                else if (b & pos->pieces(side, KNIGHT))
                {
                    piece_attacks[make_piece(KNIGHT, side)] |= b;
                    maybe_unsafe_for[~side][KNIGHT] |= b;
                    maybe_unsafe_for[~side][BISHOP] |= b;
                    unsafe_for[~side][ROOK] |= b;
                    unsafe_for[~side][QUEEN] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = KNIGHT;
                }
                else if (b & pos->pieces(side, BISHOP))
                {
                    piece_attacks[make_piece(BISHOP, side)] |= b;
                    maybe_unsafe_for[~side][BISHOP] |= b;
                    maybe_unsafe_for[~side][KNIGHT] |= b;
                    unsafe_for[~side][ROOK] |= b;
                    unsafe_for[~side][QUEEN] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = BISHOP;
                }
                else if (b & pos->pieces(side, ROOK))
                {
                    piece_attacks[make_piece(ROOK, side)] |= b;
                    maybe_unsafe_for[~side][ROOK] |= b;
                    unsafe_for[~side][QUEEN] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = ROOK;
                }
                else if (b & pos->pieces(side, QUEEN))
                {
                    piece_attacks[make_piece(QUEEN, side)] |= b;
                    maybe_unsafe_for[~side][QUEEN] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = QUEEN;
                }
                else if (b & pos->pieces(side, KING))
                {
                    piece_attacks[make_piece(KING, side)] |= b;
                    unsafe_for[~side][KING] |= b;
                    lowest[side][sq] = KING;
                }
                else
                    assert(false);
            }
        }
    }
}

int FeatureExtractor::int_lowest(Side side, Square sq) {
    return static_cast<int>(lowest[side][sq]);
}

template<Side S>
void FeatureExtractor::lowest_attackers(std::vector<float>& f) {
    for (Square sq = H1; sq <= A8; ++sq)
    {
        // lower pt gives higher score
        f.push_back((7 - int_lowest(S, sq)) / 6);
    }
}

template<PieceType PT, Side S>
void FeatureExtractor::attack_count(std::vector<float>& f) {
    int all = piece_attacks[make_piece(PT, S)];
    int safe = all & ~unsafe_for[S][PT];
    // poor-man's SEE
    safe |= all & maybe_unsafe_for[S][PT] & double_attacks[S];
    f.push_back(popcount(all) / 16);
    f.push_back(popcount(safe) / 16);
}

template<Side S>
void FeatureExtractor::counts_and_values(std::vector<float>& f) {
    float max_counts[] = { 0, 8, 2, 2, 2, 1, 0 };
    float values[] = { 0, 1, 3.25, 3.25, 5, 9.75, 0};
    float non_pawn_sum = 0;
    float pawn_sum = 0;
    for (auto bt : PieceTypes) {
        Piece pc = make_piece(bt, S);
        if (bt == KING)
            continue;
        int count = popcount(pos->pieces(S, bt));
        f.push_back(count / max_counts[bt]);
        if (bt == PAWN)
            pawn_sum += count * values[bt];
        else
            non_pawn_sum += count * values[bt];
    }
    float npv = non_pawn_sum /
        ((2 * values[KNIGHT])
         + (2 * values[BISHOP])
         + (2 * values[ROOK])
         + values[QUEEN]);
    f.push_back(npv);
    float pv = pawn_sum / (8 * values[PAWN]);
    f.push_back(pv);
}

template<Side S>
void FeatureExtractor::queens(std::vector<float>& f) {
    Bitboard qs = pos->pieces(S, QUEEN);
    Square q = qs > 0 ? lsb(qs) : SQUARE_NONE;
    push_coords(q, S, f);
    f.push_back(popcount(qs));
}

template<Side S>
void FeatureExtractor::king(std::vector<float>& f) {
    push_coords(pos->pieces(S, KING), S, f);
}

template<Side S>
void FeatureExtractor::pawn_counts(std::vector<float>& f) {
    Bitboard pawns = pos->pieces(S, PAWN);
    f.push_back(popcount(pos->pieces(S, PAWN)) / 8.0);
}

template<PieceType PT, Side S>
void FeatureExtractor::BRN_pairs(std::vector<float>& f)
{
    // B R and N pairs
    // ordered exist flag, coords, count
    
    Piece pc = make_piece(PT, S);
    Bitboard b = pos->pieces(S, PT);
    int count = popcount(b);
    f.push_back(count / 2);
    
    // exist 1, exist 2, 1 coords, 2 coords
    if (count == 1)
    {
        f.push_back(1);
        f.push_back(0);
        push_coords(b, S, f);
        push_coords(SQUARE_NONE, S, f);
    }
    else if (count >= 2)
    {
        f.push_back(1);
        f.push_back(1);
        push_coords(pop_lsb(b), S, f);
        push_coords(pop_lsb(b), S, f);
    }
    else
    {
        f.push_back(0);
        f.push_back(0);
        push_coords(SQUARE_NONE, S, f);
        push_coords(SQUARE_NONE, S, f);
    }
}

template<Side S>
void FeatureExtractor::pawn_exists(std::vector<float>& f) {
    // Rather than filling in empty slots for double pawns, I'm tallying count
    // per row
    Bitboard pawns = pos->pieces(S, PAWN);
    std::vector<float> flags = { 0, 0, 0, 0, 0, 0, 0, 0 };
    while (pawns) {
        Square psq = pop_lsb(pawns);
        flags[static_cast<size_t>(file_of(psq))] += 1;
    }
    std::move(flags.begin(), flags.end(), std::back_inserter(f));
}
