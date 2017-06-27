#include <vector>
#include <iostream>
#include <algorithm>
#include "types.h"
#include "bb.h"
#include "movecalc.h"
#include "magics.h"
#include "movegen.h"
#include "position.h"

Bitboard knight_attack(Square sq) {
    return StepAttacksBB[KNIGHT][sq];
}

Bitboard king_attack(Square sq) {
    return StepAttacksBB[KING][sq];
}

Bitboard pawn_attack(Square sq, Side side) {
    return pawn_attack_calc(SquareBB[sq], side);
}

Bitboard bishop_attack(Square sq, Bitboard occupied) {
    int hash_index;
    Bitboard occ;
    occ = occupied & MagicMasksBB[BISHOP][sq];
    occ *= MagicNumbers[BISHOP][sq];
    hash_index = occ >> (64 - MaskBitLength[BISHOP][sq]);
    return MagicAttacksBBB[sq][hash_index];
}

Bitboard rook_attack(Square sq, Bitboard occupied) {
    int hash_index;
    Bitboard occ;
    occ = occupied & MagicMasksBB[ROOK][sq];
    occ *= MagicNumbers[ROOK][sq];
    hash_index = occ >> (64 - MaskBitLength[ROOK][sq]);
    return MagicAttacksRBB[sq][hash_index];
}

Bitboard queen_attack(Square sq, Bitboard occupied) {
    return bishop_attack(sq, occupied) | rook_attack(sq, occupied);
}

Bitboard piece_attack(Piece pt, Square sq, Bitboard occupied) {
    PieceType bt = base_type(pt);
    if (bt == PAWN) {
        Side side = get_side(pt);
        return pawn_attack(sq, side);
    } else if (bt == KNIGHT) {
        return knight_attack(sq);
    } else if (bt == BISHOP) {
        return bishop_attack(sq, occupied);
    } else if (bt == ROOK) {
        return rook_attack(sq, occupied);
    } else if (bt == QUEEN) {
        return queen_attack(sq, occupied);
    } else if (bt == KING) {
        return king_attack(sq);
    }
    assert(false);
    return 0ULL;
}

// template <PieceType PT>
// Bitboard attacks_bb(Bitboard b, Bitboard occupied) {
//     if (PT == KNIGHT) {
//         return knight_attack(sq);
//     } else if (PT == BISHOP) {
//         return bishop_attack(sq, occupied);
//     } else if (PT == ROOK) {
//         return rook_attack(sq, occupied);
//     } else if (PT == QUEEN) {
//         return queen_attack(sq, occupied);
//     } else if (PT == KING) {
//         return king_attack(sq);
//     }
// }

template<PieceType PT, bool Checks>
std::vector<Move>&
generate_attacks(const Position& pos, Bitboard valid, std::vector<Move>& moves)
{
    if (Checks)
        valid |= pos.check_squares(QUEEN);
    
    Side stm = pos.side_to_move();
    Bitboard pcs = pos.pieces(stm, PT);
    while (pcs) {
        Square sq = pop_lsb(pcs);
        Bitboard attacks = piece_attack(make_piece(PT, stm), sq, pos.pieces());
        if (attacks) {
            attacks &= valid;
            while (attacks) {
                moves.push_back(make_move(sq, pop_lsb(attacks)));
            }
        }
    }
    return moves;
}

template<CastlingRight CR, bool Checks>
std::vector<Move>& generate_castling(const Position& pos, std::vector<Move>& moves)
{
    if (pos.checkers())
        return moves;
    
    Side stm = pos.side_to_move();
    Bitboard occ = pos.pieces();
    Bitboard path = castlingPathBB[CR];
    
    // std::cout << castlingPathBB[CR] << std::endl;
    // std::cout << "path " << Bitboards::print_bb(path) << std::endl;
    
    if (!pos.castling_rights(CR) || (occ & path))
        return moves;
    
    Square from = lsb(pos.pieces(stm, KING));
    Square to = (CR == WHITE_00_RIGHT || CR == BLACK_00_RIGHT) ?
        Square(from - 2) : Square(from + 2);
    
    assert(from == E1 || from == E8);

    Bitboard cpath = castlingCheckPathBB[CR];
    while (cpath)
        if (pos.attackers(pop_lsb(cpath), ~stm))
            return moves;
    
    Move move = make_move<CASTLING>(from, to);
    if (!Checks || pos.gives_check(move))
        moves.push_back(move);
    
    return moves;
}

std::vector<Move>& generate_pawn_moves(const Position& pos, Bitboard valid,
                                       std::vector<Move>& moves,
                                       const std::vector<PieceType>& promo_types)
{
    Side us = pos.side_to_move();
    Side them = ~us;
    
    Rank r3 = relative_rank(RANK_3, us);
    Rank r7 = relative_rank(RANK_7, us);
    
    Bitboard allp = pos.pieces(us, PAWN);
    Bitboard r7p = allp & rank(r7); 
    
    // single and double push (non-captures)
    Bitboard single_candidates = allp & (FULL_BOARD ^ rank(RANK_1) ^ rank(RANK_8));
    Bitboard single = shift_north(single_candidates, us) & ~pos.pieces();
    Bitboard doubles = shift_north(single & rank(r3), us) & ~pos.pieces() & valid;
    single &= valid;
    
    while (single) {
        Square to = pop_lsb(single);
        Square from = Square(us == WHITE ? to - 8 : to + 8);
        if (r7p & from)
            for (auto bt : promo_types)
                moves.push_back(make_move<PROMOTION>(from, to, bt));
        else
            moves.push_back(make_move(from, to));
    }
    while (doubles) {
        Square to = pop_lsb(doubles);
        Square from = Square(us == WHITE ? to - 16 : to + 16);
        moves.push_back(make_move(from, to));
    }
    
    // captures
    Square ep = pos.en_pessant_sq();
    if (ep != SQUARE_NONE)
        valid |= ep;
    while (allp) {
        Square from = pop_lsb(allp);
        Bitboard attacks = pawn_attack(from, us)
            & (pos.pieces(them) | ep)
            & valid;
        while (attacks) {
            Square to = pop_lsb(attacks);
            // std::cout << " e3 ? " << (to == E3) << std::endl;
            if (r7p & from)
                for (auto bt : promo_types)
                    moves.push_back(make_move<PROMOTION>(from, to, bt));
            else {
                if (to & ep) 
                    moves.push_back(make_move<ENPESSANT>(from, to));
                else
                    moves.push_back(make_move(from, to));
            }
        }
    }
    
    return moves;
}

template <GenType GT>
std::vector<Move>&
generate(const Position& pos, std::vector<Move>& moves)
{
    if (pos.checkers())
        return generate<EVASIONS>(pos, moves);
    
    Side us = pos.side_to_move();
    Side them = ~us;
    
    const bool Checks = GT == ALL_PSEUDO ? false : GT == QUIESCENCE_TIER1;

    Bitboard valid = GT == ALL_PSEUDO ? ~pos.pieces(us) : pos.pieces(them);
    
    Bitboard b = Checks ? valid | pos.check_squares(PAWN) : valid;
    std::vector<PieceType> promo_types { QUEEN, KNIGHT, ROOK, BISHOP };
    generate_pawn_moves(pos, b, moves, promo_types);
    
    generate_attacks<KNIGHT, Checks>(pos, valid, moves);
    generate_attacks<BISHOP, Checks>(pos, valid, moves);
    generate_attacks<ROOK, Checks>(pos, valid, moves);
    generate_attacks<QUEEN, Checks>(pos, valid, moves);
    generate_attacks<KING, false>(pos, valid, moves);
    
    if (us == WHITE) {
        generate_castling<WHITE_00_RIGHT, Checks>(pos, moves);
        generate_castling<WHITE_000_RIGHT, Checks>(pos, moves);
    } else {
        generate_castling<BLACK_00_RIGHT, Checks>(pos, moves);
        generate_castling<BLACK_000_RIGHT, Checks>(pos, moves);
    }
    return moves;
}

// explict instantiations
template std::vector<Move>&
generate<ALL_PSEUDO>(const Position&, std::vector<Move>&);


template<>
std::vector<Move>&
generate<EVASIONS>(const Position& pos, std::vector<Move>& moves)
{
    assert(pos.checkers());

    Side us = pos.side_to_move();
    Side them = ~us;
    
    generate_attacks<KING, false>(pos, ~pos.pieces(us), moves);

    Square ksq = lsb(pos.pieces(us, KING));
    
    Bitboard checkers = pos.checkers();

    if (!reset_ls1b(checkers)) {
        // There's only one checker, we can try capture, and block if checker is
        // a slider
        Bitboard valid = checkers;
        Bitboard slider = checkers & ~pos.pieces(them, PAWN, KNIGHT);
        if (slider)
            valid |= between_sqs(lsb(slider), ksq);
        
        std::vector<PieceType> promo_types { QUEEN, KNIGHT, ROOK, BISHOP };
        generate_pawn_moves(pos, valid, moves, promo_types);
        generate_attacks<KNIGHT, false>(pos, valid, moves);
        generate_attacks<BISHOP, false>(pos, valid, moves);
        generate_attacks<ROOK, false>(pos, valid, moves);
        generate_attacks<QUEEN, false>(pos, valid, moves);
    }

    return moves;
}

template<> 
std::vector<Move>&
generate<ALL_LEGAL>(const Position& pos, std::vector<Move>& moves)
{
    generate<ALL_PSEUDO>(pos, moves);
        
    moves.erase(std::remove_if(moves.begin(), moves.end(), [&pos](const Move& move) { return !pos.is_legal(move); }),
                moves.end());

    return moves;
}

std::vector<ProbMove> evaluate_moves(const Position& pos, const std::vector<Move>& moves)
{
    std::vector<ProbMove> pmoves;
    for (const auto& m : moves) {
        if (pos.gives_check(m))
            pmoves.emplace_back(m, .3);
        else if (pos.piece_on(to_sq(m)))
            pmoves.emplace_back(m, .25);
        else if (type_of(m) == PROMOTION && promo_piece(m) == QUEEN)
            pmoves.emplace_back(m, .25);
        else
            pmoves.emplace_back(m, .1);
    }
    std::sort(pmoves.rbegin(), pmoves.rend());
    return pmoves;
}
