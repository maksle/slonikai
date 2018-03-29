#include <vector>
#include <string>
#include <iostream>
// #include <iomanip>
#include <sstream>
#include "types.h"
#include "bb.h"
#include "zobrist.h"
#include "movegen.h"
#include "position.h"

Position::Position() {
    set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Position::Position(std::string fen) {
    set(fen);
}

bool Position::castling_rights(CastlingRight cr) const {
    return states.back().castling & cr;
}

bool Position::castling_rights(CastlingRight cr, PositionState& ps) const {
    return ps.castling & cr;
}

void Position::init_zobrist(PositionState& ps) {
    ps.zkey = 0;

    for (auto pt : Pieces) {
        for (Bitboard b = pieceBB[pt]; b; ) {
            ps.zkey ^= Zobrist::psqs[pop_lsb(b)][pt];
        }
    }

    ps.zkey ^= Zobrist::side[ps.stm];

    if (castling_rights(WHITE_00_RIGHT))
        ps.zkey ^= Zobrist::castling[0];
    if (castling_rights(WHITE_000_RIGHT))
        ps.zkey ^= Zobrist::castling[1];
    if (castling_rights(BLACK_00_RIGHT))
        ps.zkey ^= Zobrist::castling[2];
    if (castling_rights(BLACK_000_RIGHT))
        ps.zkey ^= Zobrist::castling[3];
}

void Position::set_check_info() {
    // Return squares of singular pieces between sliders and the king of side `side`.
    // Also calculates squares to help when checking if a move gives check.
    // This is called on every move..

    Side us = side_to_move();
    Side them = ~us;
    Bitboard occupied = pieces();
    
    // a) calculate moves that would check the other side's king
    Square ksq = lsb(pieces(them, KING));

    // std::cout << "ksq " << char(file_of(ksq) + 'a')
    //           << char(rank_of(ksq) + '1') << std::endl;
    
    checkSquaresBB[PAWN] = pawn_attack(ksq, them);
    checkSquaresBB[KNIGHT] = knight_attack(ksq);
    checkSquaresBB[BISHOP] = bishop_attack(ksq, occupied);
    checkSquaresBB[ROOK] = rook_attack(ksq, occupied);
    checkSquaresBB[QUEEN] = checkSquaresBB[BISHOP] | checkSquaresBB[ROOK];
    checkSquaresBB[KING] = 0ULL;
    checkSquaresBB[PIECETYPE_ALL] = checkSquaresBB[PAWN]
        | checkSquaresBB[KNIGHT]
        | checkSquaresBB[BISHOP]
        | checkSquaresBB[ROOK];
    
    // b) calculate for both sides the blockers to the king (pinned & disc check pieces)
    Bitboard sliders = (pseudo_attacks(BISHOP, ksq) & pieces(us, BISHOP, QUEEN))
        | (pseudo_attacks(ROOK, ksq) & pieces(us, ROOK, QUEEN));
    discoverers_and_pinned(sliders, ksq, discoverersBB[them], pinnedBB[them]);
 
    if (!pieces(us, KING)) {
        // int tid = omp_get_thread_num();
        std::cout
            // << "(" << tid << ")\n"
            << "\n" << *this << std::endl;
        for (auto m : moves)
            std::cout << m << " ";
        for (std::vector<Move>::const_reverse_iterator it = moves.rbegin(); it != moves.rend(); it += 1) {
            std::cout << "undoing " << *it << "\n";
            unmake_move();
            std::cout << *this;
        }
        ksq = lsb((Bitboard)0);
    }
    ksq = lsb(pieces(us, KING));

    // std::cout << char(file_of(ksq) + 'a') << char(rank_of(ksq) + '1') << std::endl;
    sliders = (pseudo_attacks(BISHOP, ksq) & pieces(them, BISHOP, QUEEN))
        | (pseudo_attacks(ROOK, ksq) & pieces(them, ROOK, QUEEN));
    discoverers_and_pinned(sliders, ksq, discoverersBB[us], pinnedBB[us]);
}

void Position::discoverers_and_pinned(Bitboard sliders, Square sq, Bitboard& discoverers, Bitboard& pinned) const {
    // Sets discoverers or pinned (based on color compared to the piece on sq), which if moved cause an attack on sq
    pinned = discoverers = 0;
    Side side = get_side(piece_on(sq));
    
    while (sliders) {
        Square sniper_sq = pop_lsb(sliders);
        Bitboard b = between_sqs(sq, sniper_sq) & pieces();
        if (b && reset_ls1b(b) == 0) {
            if (pieces(side) & b) {
                pinned |= b;
            } else if (pieces(~side) & b) {
                discoverers |= b;
            }
        }
    }
}

void Position::make_move(Move m) {
    make_move(m, gives_check(m));
}

void Position::make_move(Move m, bool givesCheck) {
    Square from = from_sq(m);
    Square to = to_sq(m);

    Piece pt = board[from];
    PieceType bt = base_type(pt);
    Side side = states.back().stm;

    Piece captured_pt = board[to];

    PositionState ps(states.back());
    
    // save captured_pt for unmake_move
    ps.captured_pt = captured_pt;

    // toggle side to move
    ps.stm = ~side;
    ps.zkey ^= Zobrist::side[WHITE];
    ps.zkey ^= Zobrist::side[BLACK];

    // update pieces
    if (captured_pt != PIECE_NONE)
        remove_piece(captured_pt, to);
    remove_piece(pt, from);
    put_piece(pt, to);

    ps.zkey ^= Zobrist::psqs[to][captured_pt];
    ps.zkey ^= Zobrist::psqs[from][pt];
    ps.zkey ^= Zobrist::psqs[to][pt];

    // removing previous ep square
    Square ep = states.back().en_pessant_sq;
    if (ep != SQUARE_NONE) {
        ps.zkey ^= Zobrist::ep_sqs[ep];
        ps.en_pessant_sq = SQUARE_NONE;
    }
    
    // creating ep square
    if (bt == PAWN
        && relative_rank(from, side) == RANK_2
        && relative_rank(to, side) == RANK_4) {
        Square n = side == WHITE
            ? Square(from + 8)
            : Square(from - 8);
        ps.en_pessant_sq = n;
        ps.zkey ^= Zobrist::ep_sqs[n];
    }

    // ep capture
    bool ep_capture = bt == PAWN and to == ep;
    if (ep_capture) {
        Square s = Square(side == WHITE ? ep - 8 : ep + 8);
        Piece p_them = make_piece(PAWN, ~side);
        remove_piece(p_them, s);
        ps.zkey ^= Zobrist::psqs[s][p_them];
    }
    
    if (bt == KING) {
        // castling
        if (side == WHITE) {
            ps.castling = CastlingRight(int(ps.castling) & ~WHITE_CASTLING_ANY);

            if (castling_rights(WHITE_00_RIGHT, states.back())) {
                ps.zkey ^= Zobrist::castling[WHITE_00];
            }
            if (castling_rights(WHITE_000_RIGHT, states.back())) {
                ps.zkey ^= Zobrist::castling[WHITE_000];
            }

            if (from == E1 && to == G1) {
                remove_piece(W_ROOK, H1);
                put_piece(W_ROOK, F1);
                ps.zkey ^= Zobrist::psqs[H1][W_ROOK];
                ps.zkey ^= Zobrist::psqs[F1][W_ROOK];
            } else if ((from == E1 && to == C1)) {
                remove_piece(W_ROOK, A1);
                put_piece(W_ROOK, D1);
                ps.zkey ^= Zobrist::psqs[A1][W_ROOK];
                ps.zkey ^= Zobrist::psqs[D1][W_ROOK];
            }
        } else { // black
            ps.castling = CastlingRight(int(ps.castling) & ~BLACK_CASTLING_ANY);

            if (castling_rights(BLACK_00_RIGHT, states.back())) {
                ps.zkey ^= Zobrist::castling[BLACK_00];
            }
            if (castling_rights(BLACK_000_RIGHT, states.back())) {
                ps.zkey ^= Zobrist::castling[BLACK_000];
            }

            if (from == E8 && to == G8) {
                remove_piece(B_ROOK, H8);
                put_piece(B_ROOK, F8);
                ps.zkey ^= Zobrist::psqs[H8][B_ROOK];
                ps.zkey ^= Zobrist::psqs[F8][B_ROOK];
            } else if ((from == E8 && to == C8)) {
                remove_piece(B_ROOK, A8);
                put_piece(B_ROOK, D8);
                ps.zkey ^= Zobrist::psqs[A8][B_ROOK];
                ps.zkey ^= Zobrist::psqs[D8][B_ROOK];
            }
        }
    }

    // moving rook causes side to lose castling rights
    if (bt == ROOK) {
        if (side == WHITE) {
            if (from == H1
                && castling_rights(WHITE_00_RIGHT, states.back())) {
                ps.castling = CastlingRight(int(ps.castling) & ~WHITE_00_RIGHT);
                ps.zkey ^= Zobrist::castling[WHITE_00];
            }
            if (from == A1
                && castling_rights(WHITE_000_RIGHT, states.back())) {
                ps.castling = CastlingRight(int(ps.castling) & ~WHITE_000_RIGHT);
                ps.zkey ^= Zobrist::castling[WHITE_000];
            }
        } else { // black
            if (from == H8
                && castling_rights(BLACK_00_RIGHT, states.back())) {
                ps.castling = CastlingRight(int(ps.castling) & ~BLACK_00_RIGHT);
                ps.zkey ^= Zobrist::castling[BLACK_00];
            }
            if (from == A8
                && castling_rights(BLACK_000_RIGHT, states.back())) {
                ps.castling = CastlingRight(int(ps.castling) & ~BLACK_000_RIGHT);
                ps.zkey ^= Zobrist::castling[BLACK_000];
            }
        }
    }
    
    // capture of rook causes other side to lose castling rights
    if (side == BLACK && (to == A1 || to == H1)) {
        if (to == H1 &&
            castling_rights(WHITE_00_RIGHT, states.back()))
        {
            ps.castling = CastlingRight(int(ps.castling) & ~WHITE_00_RIGHT);
            ps.zkey ^= Zobrist::castling[WHITE_00];
        }
        if (to == A1 &&
            castling_rights(WHITE_000_RIGHT, states.back()))
        {
            ps.castling = CastlingRight(int(ps.castling) & ~WHITE_000_RIGHT);
            ps.zkey ^= Zobrist::castling[WHITE_000];
        }
    }
    if (side == WHITE && (to == A8 || to == H8)) {
        if (to == H8 &&
            castling_rights(BLACK_00_RIGHT, states.back()))
        {
            ps.castling = CastlingRight(int(ps.castling) & ~BLACK_00_RIGHT);
            ps.zkey ^= Zobrist::castling[BLACK_00];
        }
        if (to == A8 &&
            castling_rights(BLACK_000_RIGHT, states.back()))
        {
            ps.castling = CastlingRight(int(ps.castling) & ~BLACK_000_RIGHT);
            ps.zkey ^= Zobrist::castling[BLACK_000];
        }
    }
    
    // promotions
    if (bt == PAWN && relative_rank(to, side) == RANK_8)
    {
        Piece promo_pt = make_piece(promo_piece(m), side);
        remove_piece(pt, to);
        put_piece(promo_pt, to);
        ps.zkey ^= Zobrist::psqs[to][pt];
        ps.zkey ^= Zobrist::psqs[to][promo_pt];
    }
    
    // if (from == B5 && to == E8 && int(pt) == 3) {
    //     std::cout << "move made: " << pt << m << std::endl;
    //     std::cout << *this << std::endl;
    //     std::cout << moves.size() << std::endl;
    //     std::cout << moves.rbegin()[1] << std::endl;  
    // }
    
    if (captured_pt || bt == PAWN) {
        ps.halfmove_clock = 0;
    } else {
        ps.halfmove_clock += 1;
    }

    if (side == BLACK) {
        ps.fullmove_clock += 1;
    }

    moves.push_back(m);
    // three_fold.add(fen(board_only=true)) += 1;
    states.push_back(ps);

    if (givesCheck)
        checkersBB = attackers(lsb(pieces(~side, KING)), side);
    else 
        checkersBB = 0;
    
    set_check_info();
}

void Position::unmake_move() {
    PositionState ps = states.back();
    Move m = moves.back();

    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece pt = board[to];
    
    // remove piece from 'to' square and put on 'from' square
    remove_piece(pt, to);

    if (type_of(m) == PROMOTION)
        pt = make_piece(PAWN, ~ps.stm);

    put_piece(pt, from);
    
    // put captured piece back
    if (ps.captured_pt)
        put_piece(ps.captured_pt, to);
    
    MoveType move_type = type_of(m);
    
    // put captured ep pawn back
    if (move_type == ENPESSANT) {
        Piece p = make_piece(PAWN, ps.stm);
        // std::cout << "undoing move " << m << " , removing " << p << " from " << Square(~ps.stm == WHITE ? to + 8 : to - 8) << std::endl;
        remove_piece(p, Square(~ps.stm == WHITE ? to + 8 : to - 8));
        // std::cout << "undoing move " << m << " , placing " << p << " to " << Square(~ps.stm == WHITE ? to - 8 : to + 8) << std::endl;
        put_piece(p, Square(~ps.stm == WHITE ? to - 8 : to + 8));
    }
    // put castled rook back
    else if (move_type == CASTLING) {
        if (~ps.stm == WHITE) {
            if (to == G1) {
                remove_piece(W_ROOK, F1);
                put_piece(W_ROOK, H1);
            } else if (to == C1) {
                remove_piece(W_ROOK, D1);
                put_piece(W_ROOK, A1);
            }
        } else { // black
            if (to == G8) {
                remove_piece(B_ROOK, F8);
                put_piece(B_ROOK, H8);
            } else if (to == C8) {
                remove_piece(B_ROOK, D8);
                put_piece(B_ROOK, A8);
            }
        }
    }
    // // unpromote piece
    // else if (move_type == PROMOTION) {
    //     PieceType bt = promo_piece(m);
    //     Piece pt = make_piece(bt, ~ps.stm);
    //     remove_piece(pt, to);
    // }
    
    // update the stack
    states.pop_back();
    moves.pop_back();

    // recompute pinned / discoverers / checkers
    checkersBB = attackers(lsb(pieces(side_to_move(), KING)), ~side_to_move());
    set_check_info();
}

void Position::put_piece(Piece pt, Square sq) {
    if (pt == PIECE_NONE)
        return;
    Side side = get_side(pt);
    Bitboard b = SquareBB[sq];
    pieceBB[pt] |= b;
    occupiedBB[side] |= b;
    occupiedBB[SIDE_BOTH] |= b;
    board[sq] = pt;
}

void Position::remove_piece(Piece pt, Square sq) {
    if (pt == PIECE_NONE)
        return;
    Side side = get_side(pt);
    Bitboard b = SquareBB[sq];
    pieceBB[pt] &= ~b;
    occupiedBB[side] &= ~b;
    occupiedBB[SIDE_BOTH] &= ~b;
    board[sq] = PIECE_NONE;
}

Side Position::side_to_move() const {
    return states.back().stm;
}

std::ostream& operator<<(std::ostream& os, const Position& pos) {
    for (Square s = A8; s >= H1; --s) {
        Piece pt = pos.board[s];
        os << ' ' << PicturePieces[pt] << ' ';
        if (s % 8 == 0) {
            os << '\n';
        }
    }
    os << (pos.side_to_move() == WHITE ? 'W' : 'B') << " to move\n";
    return os;
}

Bitboard Position::attackers(Square sq, Side side, Bitboard occupied) const {
    return (pawn_attack(sq, ~side) & pieces(side, PAWN))
        | (knight_attack(sq) & pieces(side, KNIGHT))
        | (bishop_attack(sq, occupied) & pieces(side, BISHOP, QUEEN))
        | (rook_attack(sq, occupied) & pieces(side, ROOK, QUEEN))
        | (king_attack(sq) & pieces(side, KING));
}

Bitboard Position::attackers(Square sq, Side side) const {
    Bitboard occupied = occupiedBB[SIDE_BOTH];
    return attackers(sq, side, occupied);
}

Bitboard Position::attackers(Square sq) const {
    return attackers(sq, WHITE) | attackers(sq, BLACK);
}

bool Position::gives_check(Move m) const {
    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece pt = piece_on(from);
    PieceType bt = base_type(pt);
    
    // direct check
    if (checkSquaresBB[bt] & SquareBB[to])
        return true;
    
    Side us = get_side(pt);
    Side them = ~us;
    Square kthem = lsb(pieces(them, KING));

    // std::cout << "......" << std::endl;
    
    // std::cout << Bitboards::print_bb(discoverers(them)) << std::endl;
    
    // discovered check
    if ((discoverers(them) & from) && !(between_sqs(kthem, from) & to)) {
        return true;
    }
        
    Bitboard occ, rook;
    
    switch (type_of(m)) {
    case NORMAL:
        return false;

    case PROMOTION:
        return piece_attack(make_piece(promo_piece(m), get_side(pt)), to, pieces() ^ SquareBB[from]) & kthem;

    case ENPESSANT:
        // can only be check by a slider due to clearance of the taken pawn
        occ = (pieces() ^ shift_south(SquareBB[to], get_side(pt)) ^ from) | to;
        return (bishop_attack(kthem, occ) | rook_attack(kthem, occ)) & (pieces(us, ROOK, BISHOP) | pieces(us, QUEEN));

    case CASTLING:
        occ = pieces() ^ from ^ to;
        if (us == WHITE && to == G1) {
            occ ^= SquareBB[H1] ^ SquareBB[F1];
            rook = SquareBB[F1];
        } else if (us == WHITE && to == C1) {
            occ ^= SquareBB[A1] ^ SquareBB[D1];
            rook = SquareBB[D1];
        } else if (us == BLACK && to == G8) {
            occ ^= SquareBB[H8] ^ SquareBB[F8];
            rook = SquareBB[F8];
        } else if (us == BLACK && to == G8) {
            occ ^= SquareBB[A8] ^ SquareBB[D8];
            rook = SquareBB[D8];
        }
        return rook_attack(kthem, occ) & rook;
    default:
        assert(false);
        return false;
    }
}

bool Position::is_legal(Move move) const {
    Side us = side_to_move();
    Side them = ~us;
    Square from = from_sq(move);
    Square to = to_sq(move);
    Piece pt = piece_on(from);
    PieceType bt = base_type(pt);
    
    // King can't step into check. Castling checks are taken care of in move
    // generation.
    if (bt == KING) {
        // std::cout << move << std::endl;
        // std::cout << Bitboards::print_bb(pieces() ^ SquareBB[from] ^ SquareBB[to]) << std::endl;
        // std::cout << attackers(to, them, pieces() ^ SquareBB[from] ^ SquareBB[to]) << std::endl;
        return (type_of(move) == CASTLING) || !attackers(to, them, pieces() ^ SquareBB[from] ^ SquareBB[to]);
    }
        
    Square ksq = lsb(pieces(us, KING));

    // en pessant
    if ((bt == PAWN)
        && en_pessant_sq() != SQUARE_NONE
        && (SquareBB[en_pessant_sq()] & to)) {
        // std::cout << "this happened for move " << move << std::endl;
        // std::cout << en_pessant_sq() << move << std::endl;
        Bitboard occ = pieces();
        occ ^= SquareBB[from];
        occ |= to;
        occ ^= shift_south(SquareBB[to], us);
        return !(rook_attack(ksq, occ) & pieces(them, ROOK, QUEEN))
            && !(bishop_attack(ksq, occ) & pieces(them, BISHOP, QUEEN));
    }
    
    // pinned pieces can only move in line with the king
    if ((pinned(us) & from) && !(line_sqs(from, ksq) & to))
        return false;

    return true;
}

Position& Position::set(const std::string& fen) {
    std::istringstream ss(fen);
    unsigned char token;

    states.clear();
    moves.clear();

    std::fill(std::begin(occupiedBB), std::end(occupiedBB), 0);
    std::fill(std::begin(pieceBB), std::end(pieceBB), 0);
    std::fill(std::begin(board), std::end(board), PIECE_NONE);
    std::fill(std::begin(checkSquaresBB), std::end(checkSquaresBB), PIECE_NONE);
    checkersBB = 0;
    
    PositionState ps = PositionState();

    ss >> std::noskipws;
    
    // pieces
    Square sq = A8;
    while ((ss >> token) && (!isspace(token))) {
        size_t idx;
        Piece pt;
        if (isdigit(token)) {
            sq = Square(int(sq) - (token - '0') + 1);
            --sq;
        } else if (token != '/') {
            if ((idx = PieceToChar.find(token)) != std::string::npos) {
                pt = Piece(idx);
                put_piece(pt, sq);
            }
            --sq;
        }
    }
    
    // color
    ss >> token;
    ps.stm = token == 'w' ? WHITE : BLACK;
    ss >> token;
    
    // castling
    Bitboard cr = 0;
    while ((ss >> token) && !isspace(token)) {
        if (token == 'K') cr |= WHITE_00_RIGHT;
        else if (token == 'Q') cr |= WHITE_000_RIGHT;
        else if (token == 'k') cr |= BLACK_00_RIGHT;
        else if (token == 'q') cr |= BLACK_000_RIGHT;
    }
    ps.castling = CastlingRight(cr);
    
    // en_pessant
    unsigned char col, rank;
    if (   ((ss >> col) && (col >= 'a' && col <= 'h'))
        && ((ss >> rank) && (rank == '3' || rank == '6'))) {
        ps.en_pessant_sq = make_square(File(col - 'a'), Rank(rank - '1'));
    } else {
        ps.en_pessant_sq = SQUARE_NONE;
        ss >> token;
    }
    
    ss >> std::skipws >> ps.halfmove_clock >> ps.fullmove_clock;
    
    states.push_back(ps);
    
    checkersBB = attackers(lsb(pieces(ps.stm, KING)), ~ps.stm);
    set_check_info();
    
    init_zobrist(ps);
    
    return *this;
}

const std::string Position::fen() const {
    int empty = 0;
    std::ostringstream ss;
    const PositionState& ps = states.back();

    for (Square sq = A8; sq >= H1; --sq) {
        Piece pt = piece_on(sq);
        if (pt == PIECE_NONE) {
            empty++;
        } else {
            if (empty > 0) {
                ss << empty;
                empty = 0;
            }
            ss << PieceToChar.at(pt);
        }
        if (file_of(sq) == FILE_H) {
            if (empty > 0) {
                ss << empty;
                empty = 0;
            }
            if (sq > H1) {
                ss << '/';
            }
        }
    }

    if (side_to_move() == WHITE)
        ss << " w ";
    else
        ss << " b ";

    if (ps.castling & WHITE_00_RIGHT)
        ss << 'K';
    if (ps.castling & WHITE_000_RIGHT)
        ss << 'Q';
    if (ps.castling & BLACK_00_RIGHT)
        ss << 'k';
    if (ps.castling & BLACK_000_RIGHT)
        ss << 'q';
    if (!(ps.castling & (WHITE_CASTLING_ANY | BLACK_CASTLING_ANY)))
        ss << '-';

    ss << ' ';

    if (ps.en_pessant_sq != SQUARE_NONE) {
        Square ep = ps.en_pessant_sq;
        ss << std::string { char(file_of(ep) + 'a'), char(rank_of(ep) + '1') };
    } else {
        ss << '-';
    }

    ss << ' ' << ps.halfmove_clock << ' ' << ps.fullmove_clock;
    
    return ss.str();
}

bool Position::arbiter_draw() const {
    const PositionState& ps = states.back();

    if (ps.halfmove_clock >= 100)
        return true;
    
    for (std::vector<PositionState>::const_reverse_iterator it = states.rbegin();
         it + 1 != states.rend() && it + 2 != states.rend();
         it += 2)
    {
        if ((it + 2)->zkey == ps.zkey)
            return true;
    }

    return false;
}

// Some very basic detection
bool Position::insufficient_material() const {
    if (pieces(QUEEN) || pieces(ROOK) || pieces(PAWN))
        return false;
    
    int nb_pcs = popcount(pieces());

    if (nb_pcs == 2)
        return true;
    
    if (nb_pcs == 3 && (pieces(KNIGHT) || pieces(BISHOP)))
        return true;
    
    return false;
}

// For speculative prefetch
Key Position::key_after(Move m) const
{
    Square from = from_sq(m);
    Square to = to_sq(m);
    Piece pc = piece_on(from);
    Piece captured = piece_on(to);
    Key k = key() ^ Zobrist::side[0] ^ Zobrist::side[1];

    if (captured)
        k ^= Zobrist::psqs[to][captured];

    return k ^= Zobrist::psqs[from][pc] ^ Zobrist::psqs[to][pc];
}
