#ifndef MOVEGEN_GAURD
#define MOVEGEN_GAURD

class Position;

enum GenType {
    ALL_PSEUDO,
    ALL_LEGAL,
    EVASIONS,
    QUIESCENCE_TIER1,
    QUIESCENCE_TIER2
};

Bitboard knight_attack(Square sq);
Bitboard king_attack(Square sq);
Bitboard pawn_attack(Square sq, Side side);
Bitboard bishop_attack(Square sq, Bitboard occupied = 0);
Bitboard rook_attack(Square sq, Bitboard occupied = 0);
Bitboard queen_attack(Square sq, Bitboard occupied = 0);
Bitboard piece_attack(Piece pt, Square sq, Bitboard occupied = 0);

template <GenType GT>
std::vector<Move>& generate(const Position&, std::vector<Move>&);

template<CastlingRight cr>
std::vector<Move>& generate_castling(const Position&, std::vector<Move>&);

#endif
