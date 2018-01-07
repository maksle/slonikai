#ifndef FEATURES_H_GUARD
#define FEATURES_H_GUARD

class FeatureExtractor {
private:
    const Position* pos = nullptr;

    Bitboard all_attacks[SIDE_NB];
    Bitboard double_attacks[SIDE_NB];
    Bitboard piece_attacks[PIECE_NB];
    Bitboard unsafe_for[SIDE_NB][PIECETYPE_NB];
    Bitboard maybe_unsafe_for[SIDE_NB][PIECETYPE_NB];
    PieceType lowest[SIDE_NB][SQUARE_NB];

public:
    std::vector<std::vector<float> > extract();
    int int_lowest(Side side, Square sq);
    void set_position(const Position& position);
    void set_attacks();
    template<Side S> void queens(std::vector<float>& f);
    template<Side S> void king(std::vector<float>& f);
    template<Side S> void lowest_attackers(std::vector<float>& f);
    void castling_rights(std::vector<float>& f);
    template<PieceType PT, Side S> void attack_count(std::vector<float>& f);
    template<Side S> void counts_and_values(std::vector<float>& f);
    template<PieceType PT, Side S> void BRN_pairs(std::vector<float>& f);
    template<Side side> void pawn_exists(std::vector<float>& f);
    template<Side S> void pawn_counts(std::vector<float>& f);
};

#endif
