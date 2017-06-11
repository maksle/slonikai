#ifndef TYPES_GAURD
#define TYPES_GAURD

#include <sstream>
#include <cassert>
#include <stdint.h>
#include <string>

typedef uint64_t ULL;
typedef uint64_t Key;
typedef uint64_t Bitboard;

enum Square {
  H1, G1, F1, E1, D1, C1, B1, A1,
  H2, G2, F2, E2, D2, C2, B2, A2,
  H3, G3, F3, E3, D3, C3, B3, A3,
  H4, G4, F4, E4, D4, C4, B4, A4,
  H5, G5, F5, E5, D5, C5, B5, A5,
  H6, G6, F6, E6, D6, C6, B6, A6,
  H7, G7, F7, E7, D7, C7, B7, A7,
  H8, G8, F8, E8, D8, C8, B8, A8,
  SQUARE_NONE,
  SQUARE_NB = 64
};

enum Direction {
  NORTH = 8,
  SOUTH = -8,
  EAST = -1,
  WEST = 1
};

enum File {
  FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_NB
};

enum Rank {
  RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_NB
};

enum BoardSide {
  QUEENSIDE, KINGSIDE
};

enum Side {
  WHITE, BLACK, SIDE_NB = 2,
  SIDE_BOTH = 2, SIDE_BOTH_NB = 3
};

enum Piece { 
  PIECE_NONE = 0,
  W_PAWN = 1, W_KNIGHT = 2, W_BISHOP = 3, W_ROOK = 4, W_QUEEN = 5, W_KING = 6,
  B_PAWN = 7, B_KNIGHT = 8, B_BISHOP = 9, B_ROOK = 10, B_QUEEN = 11, B_KING = 12,
  PIECE_ALL,
  PIECE_NB,
};

enum PieceType {
  PIECETYPE_NONE = 0,
  PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6,
  PIECETYPE_ALL = 7,
  PIECETYPE_NB = 8
};

const Piece Pieces[] = { W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                         B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING };

const std::string PieceToChar(" PNBRQKpnbrqk");
const std::string PicturePieces[] {
  "·", "♙", "♘", "♗", "♖", "♕", "♔", "♟", "♞", "♝", "♜", "♛", "♚"
};

enum Castling {
  WHITE_00, WHITE_000,
  BLACK_00, BLACK_000,
  CASTLING_NB
};

enum CastlingRight {
  WHITE_00_RIGHT = 1,
  WHITE_000_RIGHT = 2,
  WHITE_CASTLING_ANY = 3,
  BLACK_00_RIGHT = 4,
  BLACK_000_RIGHT = 8,
  BLACK_CASTLING_ANY = 12,
  CASTLING_RIGHT_NB
};

// Using Stockfish's more compact representation of moves, not including
// piecetype, so that it can fit in 16 bits and TTEntry can fit into 64 bits.
// 0-5 from square
// 6-11 to square
// 12-13 promo piece (KNIGHT-2) to (QUEEN-2)
// 14-15 promotion(1), en pessant (2), castling (3)
enum Move : int {
  MOVE_NONE, MOVE_NULL
};

enum MoveType {
  NORMAL,
  PROMOTION = 1 << 14,
  ENPESSANT = 2 << 14,
  CASTLING = 3 << 14
};

const int MAX_MOVES = 128;

// Values From stockfish
enum Value {
  VALUE_ZERO = 0,
  VALUE_DRAW = 0,
  
  PawnValueMg   = 188,   PawnValueEg   = 248,
  KnightValueMg = 753,   KnightValueEg = 832,
  BishopValueMg = 826,   BishopValueEg = 897,
  RookValueMg   = 1285,  RookValueEg   = 1371,
  QueenValueMg  = 2513,  QueenValueEg  = 2650,

  MidgameLimit  = 15258, EndgameLimit  = 3915
};

enum Phase {
  MG, EG, PHASE_NB
};

inline PieceType& operator++(PieceType& pt) { return pt = PieceType(int(pt) + 1); }
inline Square& operator++(Square& sq) { return sq = Square(int(sq) + 1); }
inline Square& operator--(Square& sq) { return sq = Square(int(sq) - 1); }
inline Rank& operator++(Rank& r) { return r = Rank(int(r) + 1); }
inline File& operator--(File& f) { return f = File(int(f) - 1); }
inline File& operator++(File& f) { return f = File(int(f) + 1); }
inline Piece& operator++(Piece& p) { return p = Piece(int(p) + 1); }

inline Side operator~(Side s) {
  return Side(s ^ 1);
}

inline PieceType base_type(Piece pt) {
  return PieceType(pt < 7 ? pt : pt - 6);
}

inline bool is_white(Piece pt) {
  return pt >= 1 && pt <= 6;
}

inline bool is_black(Piece pt) {
  return pt >= 7 && pt <= 12;
}

inline Side get_side(Piece pt) {
  return is_white(pt) ? WHITE : BLACK;
}

inline Piece make_piece(PieceType pt, Side side) {
  return Piece(side == WHITE ? pt : pt + 6);
}

inline Square make_square(File f, Rank r) {
  return Square((r << 3) + (7 - f));
}

inline Rank rank_of(Square s) {
  return Rank(s >> 3);
}

inline File file_of(Square s) {
  return File(7 - (s & 7));
}

inline Rank relative_rank(Square sq, Side side) {
  return Rank(rank_of(sq) ^ (side * 7));
}

inline Rank relative_rank(Rank r, Side side) {
  return Rank(r ^ (side * 7));
}

inline Move make_move(Square from_sq, Square to_sq) {
  return Move(from_sq + (to_sq << 6));
}

/* #include <iostream> */
template<MoveType T>
inline Move make_move(Square from_sq, Square to_sq, PieceType pt = KNIGHT) {
  /* std::cout << "movetype T is " << T << std::endl; */
  /* std::cout << "parameters are " << T << make_move(from_sq, to_sq) << " " << pt << std::endl; */
  return Move(T + ((pt - KNIGHT) << 12) + from_sq + (to_sq << 6));
}

inline MoveType type_of(Move m) {
  return MoveType(m & (3 << 14));
}

inline PieceType promo_piece(Move m) {
  return PieceType(((m >> 12) & 3) + KNIGHT);
}

inline Square from_sq(Move m) {
  return Square(m & 0x3F);
}

inline Square to_sq(Move m) {
  return Square((m >> 6) & 0x3F);
}

inline std::ostream& operator<<(std::ostream& os, const Move& move) {
  Square from = from_sq(move);
  Square to = to_sq(move);
  return os << char(file_of(from) + 'a')
            << char(rank_of(from) + '1')
            << char(file_of(to) + 'a')
            << char(rank_of(to) + '1');
}

#endif
