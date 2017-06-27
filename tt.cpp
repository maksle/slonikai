#include <stdint.h>
#include "types.h"
#include "position.h"
#include "tt.h"

TTable TT; // global


void TTable::resize(size_t mb_size) {
    size_t curr_mb_size = sizeof(TTEntry) * entries.size() / (1024 * 1024);
    if (curr_mb_size == mb_size)
        return;

    entries.resize(mb_size * 1024 * 1024 / sizeof(TTEntry));
}

TTEntry* TTable::probe(Key zkey)
{
    size_t idx = (zkey >> 48) % entries.size();
    TTEntry *entry = &entries[idx];
    if (entry->key == (zkey >> 48))
        return entry;
    return nullptr;
}

void TTable::prefetch(Key zkey)
{
    __builtin_prefetch(&entries[(zkey >> 48) % entries.size()]);
}

void TTable::save(const Position& pos, Move move, TTBound bound, Score val, uint64_t allowance)
{
    Key zkey = pos.key();
    TTEntry* entry = &entries[(zkey >> 48) % entries.size()];
    if ((zkey >> 48) != entry->key
        || allowance >= entry->allowance
        || entry->gen() < current_generation)
    {
        entry->key = zkey >> 48;
        entry->move = move;
        entry->value = val;
        entry->allowance = allowance;
        entry->gen_bound = current_generation + bound;
    }
}

void TTable::age()
{
    current_generation += 4;
    if (current_generation == 4)
        clear();
};

void TTable::clear()
{
    for (auto& e : entries)
        e.key = 0;
};
