#ifndef POSITION_GUARD_H
#define POSITION_GUARD_H

#include <vector>

struct PositionState {
    Side stm;
    int halfmove_clock;
    int fullmove_clock;
    Square en_pessant_sq;
    Key zkey;
    CastlingRight castling;
    Piece captured_pt;
};

class Position {
private:
    Piece board[SQUARE_NB];
    Bitboard checkersBB;
    Bitboard pieceBB[PIECE_NB];
    Bitboard occupiedBB[SIDE_BOTH_NB];
    
    std::vector<PositionState> states;
    // Move moves[MAX_MOVES];
    // std::map<string, int> three_fold;

    // one slot for each side
    Bitboard pinnedBB[SIDE_NB];
    Bitboard discoverersBB[SIDE_NB];
    Bitboard checkSquaresBB[PIECETYPE_NB];
    
    void init_zobrist(PositionState&);

    void set_check_info();
    void discoverers_and_pinned(Bitboard sliders, Square sq, Bitboard& discoverers, Bitboard& pinned) const;

public:
    std::vector<Move> moves;

    Position();

    bool castling_rights(CastlingRight c) const;
    bool castling_rights(CastlingRight cr, PositionState& ps) const;
    
    void make_move(Move m, bool givesCheck);
    void unmake_move();

    void put_piece(Piece pt, Square sq);
    void remove_piece(Piece pt, Square sq);
    
    Position& set(const std::string& fen);
    const std::string fen() const;

    Side side_to_move() const;
    Key key() const;
    Key key_after(Move m) const;
    Square en_pessant_sq() const;
    Piece piece_on(Square sq) const;
    Bitboard pieces() const;
    Bitboard pieces(Side side) const;
    Bitboard pieces(Side side, PieceType bt) const;
    Bitboard pieces(PieceType bt) const;
    Bitboard pieces(Piece pt) const;
    Bitboard pieces(Piece pt, Piece pt2) const;
    Bitboard pieces(Side side, PieceType bt, PieceType bt2) const;
    Bitboard attackers(Square sq, Side side, Bitboard occupied) const;
    Bitboard attackers(Square sq, Side side) const;
    Bitboard attackers(Square sq) const;
    Bitboard checkers() const;
    Bitboard check_squares(PieceType pt) const;
    Piece captured() const;

    bool gives_check(Move m) const;
    Bitboard discoverers(Side s) const;
    Bitboard pinned(Side s) const;
    bool is_legal(Move move) const;
    bool arbiter_draw() const;
    bool insuficient_material() const;

    friend std::ostream& operator<<(std::ostream& os, const Position& pos);
};

inline Piece Position::piece_on(Square sq) const {
    return board[sq];
}

inline Bitboard Position::pieces() const {
    return occupiedBB[SIDE_BOTH];
}

inline Bitboard Position::pieces(Side side) const {
    return occupiedBB[side];
}

inline Bitboard Position::pieces(Side side, PieceType bt) const {
    return pieceBB[make_piece(bt, side)];
}

inline Bitboard Position::pieces(PieceType bt) const {
  return pieceBB[make_piece(bt, WHITE)] | pieceBB[make_piece(bt, BLACK)];
}

inline Bitboard Position::pieces(Piece pt) const {
    return pieceBB[pt];
}

inline Bitboard Position::pieces(Piece pt, Piece pt2) const {
    return pieceBB[pt] | pieceBB[pt2];
}

inline Bitboard Position::pieces(Side side, PieceType bt, PieceType bt2) const {
    return pieceBB[make_piece(bt, side)] | pieceBB[make_piece(bt2, side)];
}

inline Bitboard Position::checkers() const {
    return checkersBB;
}

inline Bitboard Position::check_squares(PieceType pt) const {
  return checkSquaresBB[pt];
}

inline Piece Position::captured() const {
  return states.back().captured_pt;
}

inline Key Position::key() const {
  return states.back().zkey;
}

inline Square Position::en_pessant_sq() const {
  return states.back().en_pessant_sq;
}

inline Bitboard Position::discoverers(Side s) const {
  return discoverersBB[s];
}

inline Bitboard Position::pinned(Side s) const {
  return pinnedBB[s];
}

#endif
