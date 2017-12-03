// #ifndef SEARCH_CPP
// #define SEARCH_CPP

#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include "types.h"
#include "bb.h"
#include "movegen.h"
#include "position.h"
#include "search.h"
#include "tt.h"

using namespace Search;

namespace {

    int nodecnt = 0;
    int tthitcnt = 0;
    
    // int statically_evaluate(const Position& pos);
    void print_uci_moves(const std::vector<Move>&);
    std::vector<Move> tt_find_pv(Position& pos);

    template<bool PVNode>
    int search(Context& context, Position& pos, SearchInfo* si, Score alpha, Score beta, uint64_t allowance);

    template<bool PVNode>
    int qsearch(Context& context, Position& pos, SearchInfo* si, Score alpha, Score beta);
}

template<bool Root>
int Search::perft(Position& pos, int depth) {
    int cnt = 0;
    int nodes = 0;
    bool leaf = depth == 2;

    std::vector<Move> moves;
    moves.reserve(50);
    generate<ALL_LEGAL>(pos, moves);

    for (const auto& move : moves)
    {
        if (Root and depth <= 1) {
            cnt = 1;
            nodes++;
        } else {
            pos.make_move(move, pos.gives_check(move));
            if (leaf) {
                std::vector<Move> moves2;
                moves2.reserve(50);
                generate<ALL_LEGAL>(pos, moves2);
                cnt = moves2.size();
            } else {
                cnt = perft<false>(pos, depth - 1);
            }
            pos.unmake_move();
            nodes += cnt;
        }
        if (Root)
            std::cout << move << ": " << cnt << std::endl;
    }
    return nodes;
}

SearchOutput Search::iterative_deepening(Context& context)
{
    int depth = 0;
    uint64_t allowance = std::pow(4, depth);
    
    Search::SearchInfo si[256] {};
    auto root_si = si + 1;
    
    Signals signals = context.signals;

    std::vector<Move> pv;

    Score value = context.evaluator(context.root_position);
    while (!signals.stop && depth < context.limits.max_depth)
    {
        depth += 1;
        allowance = std::pow(4, depth);
        bool finished = false;
        Score fail_factor = 400;

        Score alpha = value - fail_factor;
        Score beta = value + fail_factor;

        if (alpha >= beta)
        {
            alpha = NEGATIVE_INF;
            beta = POSITIVE_INF;
        }
            
        std::string bound = "";
        
        std::cout << "depth " << depth << " allowance " << allowance << std::endl;
        
        while (!finished)
        {
            nodecnt = 0;
            tthitcnt = 0;

            if (signals.stop) break;
            root_si->pv.clear();
            value = search<true>(context, context.root_position, root_si, alpha, beta, allowance);
            if (signals.stop) break;

            std::cout << "val: " << value
                      << " alpha: " << alpha
                      << " beta: " << beta
                      << (value <= alpha ? " faillow " : value >= beta ? " failhigh " : " exact ")
                      << " ;; nodecnt: " << nodecnt
                      << " tthitcnt: " << tthitcnt << std::endl;
            
            if (value <= alpha) {
                bound = " upperbound";
                beta = (alpha + beta) / 2;
                if (value - fail_factor < value)
                    alpha = static_cast<Score>(value - fail_factor);
                else
                    alpha = NEGATIVE_INF;
            } else if (value >= beta ) {
                bound = " lowerbound";
                alpha = (alpha + beta) / 2;
                if (value + fail_factor > value)
                    beta = static_cast<Score>(value + fail_factor);
                else
                    beta = POSITIVE_INF;
                pv = root_si->pv;
            } else {
                bound = "";
                pv = root_si->pv;
                finished = true;
            }

            fail_factor *= 8;

            if (bound == "")
            {
                std::cout << "info depth " << depth
                          << " score cp " << value << " " << bound
                          << " pv ";
                print_uci_moves(pv);
                std::cout << std::endl;
            }
        }
    }

    if (pv.size() == 0)
        pv = tt_find_pv(context.root_position);

    if (pv.size() > 1)
        std::cout << "bestmove " << pv[0] << " ponder " << pv[1] << std::endl;
    else if (pv.size() > 0)
        std::cout << "bestmove " << pv[0] << std::endl;

    SearchOutput output = SearchOutput();
    output.pv = pv;
    output.value = value;
    return output;
}

template int Search::perft<true>(Position& pos, int depth);
// template int Search::search<true>(Position& pos, SearchInfo* si, int alpha, int beta, double allowance);
// template int Search::search<false>(Position& pos, SearchInfo* si, int alpha, int beta, double allowance);

namespace {

void print_uci_moves(const std::vector<Move>& moves) {
    for (const auto& m : moves)
        std::cout << m << " ";
}

std::vector<Move> tt_find_pv(Position& pos) {
    std::vector<Move> result;

    auto find_next_move = [&pos]() {
        TTEntry* tte = TT.probe(pos.key());
        if (tte && (tte->bound() == LOW_BOUND || tte->bound() == EXACT_BOUND)) 
            return tte->best_move();
        return MOVE_NONE;
    };

    int moves_made = 0;

    Move m = find_next_move();

    while (m != MOVE_NONE) {
        result.push_back(m);
        pos.make_move(m, pos.gives_check(m));
        m = find_next_move();
        moves_made++;
    }

    while (moves_made--)
        pos.unmake_move();

    return result;
}

template<bool PVNode>
int search(Context& context, Position& pos, SearchInfo* si, Score alpha, Score beta, uint64_t allowance)
{
    assert(PVNode || (alpha == beta - 1));

    nodecnt += 1;
    tthitcnt += 1;
    
    bool is_root = PVNode && si->ply == 0;
    
    si->ply = (si-1)->ply + 1;
    if (!is_root)
        si->pv.clear();
    (si+1)->pv.clear();

    si->is_QS = false;
    si->total_allowance = allowance;

    if (alpha > MATE_WIN - (si->ply + 1))
        return alpha;
    
    if (pos.arbiter_draw() || (si->ply >= MAX_PLY))
        return DRAW_VALUE;
    
    if (allowance < 1)
        return qsearch<PVNode>(context, pos, si, alpha, beta);
    
    TTEntry* tte = TT.probe(pos.key());
    if (tte && tte->allowance >= allowance)
    {
        if (!PVNode
            && tte->value != VALUE_NONE
            && (tte->value >= beta
                ? tte->bound() == LOW_BOUND
                : tte->bound() == HIGH_BOUND))
        {
            tthitcnt += 1;
            return tte->value;
        }
    }

    allowance *= .9999;
    
    // Move tt_move = tte ? tte->best_move() : MOVE_NONE;
    
    // int static_eval;
    // si->static_eval = static_eval = statically_evaluate(pos);

    // bool improving = (si->ply < 2)
    //     || (si->ply.static_eval >= (si-2)->ply.static_eval);
    
    Move best_move = MOVE_NONE;
    Score best_val = NEGATIVE_INF;
    int move_count = 0;

    std::vector<Move> moves;
    generate<ALL_PSEUDO>(pos, moves);
    std::vector<ProbMove> pmoves = evaluate_moves(pos, moves);

    for (const auto& pm : pmoves)
    {
        Move m = pm.move;
        double prob = pm.probability;

        Score val = 0;

        TT.prefetch(pos.key_after(m));
        
        if (!pos.is_legal(m))
            continue;
        
        bool gives_check = pos.gives_check(m);
        pos.make_move(m, gives_check);
        ++move_count;
        
        uint64_t child_allowance = allowance * prob;
        if (PVNode && (move_count > 1) && (allowance > MIN_PVS_ALLOWANCE))
        {
            val = -search<false>(context, pos, si+1, -(alpha+1), -alpha, child_allowance);
            if (alpha < val && val < beta)
                val = -search<true>(context, pos, si+1, -beta, -alpha, child_allowance);
        }
        else
            val = -search<PVNode>(context, pos, si+1, -beta, -alpha, child_allowance);

        pos.unmake_move();
        
        if (val > best_val)
        {
            best_val = val;
            best_move = m;

            if (val > alpha)
            {
                alpha = val;

                if (PVNode) {
                    si->pv.clear();
                    si->pv.push_back(m);
                    si->pv.insert(si->pv.end(), (si+1)->pv.begin(), (si+1)->pv.end());
                }

                if (val >= beta)
                    break;
            }
                
        }
    }

    if (move_count == 0)
    {
        if (pos.checkers())
            return MATE_LOSE + si->ply;
        else
            return DRAW_VALUE;
    }

    TTBound&& bound = (best_val >= beta ? LOW_BOUND
                       : best_val <= alpha ? HIGH_BOUND
                       : EXACT_BOUND);
    // TT.save(pos, best_move, bound, best_val, allowance);
    
    return best_val;
}

template<bool PVNode>
int qsearch(Context& context, Position& pos, SearchInfo* si, Score alpha, Score beta)
{

    nodecnt += 1;
    
    si->pv.clear();
    (si+1)->pv.clear();
    si->ply = (si-1)->ply + 1;
    si->is_QS = true;
    
    // if (pos.insuficient_material())
    if (pos.arbiter_draw() || (si->ply >= MAX_PLY))
        return DRAW_VALUE;

    // Try to avoid calling evaluate
    TTEntry* tte = TT.probe(pos.key());
    if (tte
        && !PVNode
        && (tte->value >= beta
            ? tte->bound() == LOW_BOUND
            : tte->bound() == HIGH_BOUND)) {
        tthitcnt += 1;
        return tte->value;
    }
        
    Score alpha_orig = alpha;
    Score best_value = NEGATIVE_INF;
    
    bool in_check = pos.checkers();
    if (!in_check)
    {
        // We can't stand pat if in check, because standing pat assumes that
        // there is at least some quiet move at least as good as alpha. The
        // assumption isn't safe when in check.
        Score static_eval = context.evaluator(pos);
        best_value = static_eval;

        if (static_eval >= beta)
            return static_eval;

        if (static_eval > alpha)
            alpha = static_eval;
    }
    
    Move best_move = MOVE_NONE;
    int move_count = 0;

    std::vector<Move> moves;
    generate<QUIESCENCE_TIER2>(pos, moves);
    std::vector<ProbMove> pmoves = evaluate_moves(pos, moves);

    for (const auto& pm : pmoves)
    {
        Move m = pm.move;
        Score val = 0;

        TT.prefetch(pos.key_after(m));
        
        if (!pos.is_legal(m))
            continue;

        ++move_count;
        pos.make_move(m, pos.gives_check(m));
        val = -qsearch<PVNode>(context, pos, si+1, -beta, -alpha);
        pos.unmake_move();
        
        if (val > best_value)
        {
            best_value = val;
            best_move = m;

            if (val > alpha)
            {
                alpha = val;
                
                if (PVNode)
                {
                    si->pv.clear();
                    si->pv.push_back(m);
                    si->pv.insert(si->pv.end(), (si+1)->pv.begin(), (si+1)->pv.end());
                }

                if (val >= beta)
                    break;
            }
        }
    }

    if (in_check && move_count == 0)
        return MATE_LOSE + si->ply;

    TTBound&& bound = (best_value >= beta ? LOW_BOUND
                       : best_value <= alpha_orig ? HIGH_BOUND
                       : EXACT_BOUND);
    // TT.save(pos, best_move, bound, best_value, 0);
        
    return best_value;
}

} // namespace

// #endif
