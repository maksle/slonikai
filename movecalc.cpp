#include "bb.h"

Bitboard west_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~H_FILE;
    pr1 = pr0 & (pr0 << 1);
    pr2 = pr1 & (pr1 << 2);
    g |= pr0 & g << 1;
    g |= pr1 & g << 2;
    g |= pr2 & g << 4;
    return (g << 1) & ~H_FILE;
}

Bitboard east_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~A_FILE;
    pr1 = pr0 & (pr0 >> 1);
    pr2 = pr1 & (pr1 >> 2);
    g |= pr0 & g >> 1;
    g |= pr1 & g >> 2;
    g |= pr2 & g >> 4;
    return (g >> 1) & ~A_FILE;
}

Bitboard north_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    g |= p & (g << 8);
    p &= (p << 8);
    g |= p & (g << 16);
    p &= (p << 16);
    g |= p & (g << 32);
    return (g << 8);
}

Bitboard south_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    g |= p & (g >> 8);
    p &= p >> 8;
    g |= p & (g >> 16);
    p &= p >> 16;
    g |= p & (g >> 32);
    return g >> 8;
}

Bitboard ne_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~A_FILE;
    pr1 = pr0 & (pr0 << 7);
    pr2 = pr1 & (pr1 << 14);
    g |= pr0 & g << 7;
    g |= pr1 & g << 14;
    g |= pr2 & g << 28;
    return (g << 7) & ~A_FILE;
}

Bitboard nw_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~H_FILE;
    pr1 = pr0 & (pr0 << 9);
    pr2 = pr1 & (pr1 << 18);
    g |= pr0 & g << 9;
    g |= pr1 & g << 18;
    g |= pr2 & g << 36;
    return (g << 9) & ~H_FILE;
}

Bitboard se_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~A_FILE;
    pr1 = pr0 & (pr0 >> 9);
    pr2 = pr1 & (pr1 >> 18);
    g |= pr0 & g >> 9;
    g |= pr1 & g >> 18;
    g |= pr2 & g >> 36;
    return g >> 9 & ~A_FILE;
}

Bitboard sw_attack(ULL g, ULL p) {
    // g: attacker
    // p: free squares
    ULL pr0, pr1, pr2;
    pr0 = p & ~H_FILE;
    pr1 = pr0 & (pr0 >> 7);
    pr2 = pr1 & (pr1 >> 14);
    g |= pr0 & g >> 7;
    g |= pr1 & g >> 14;
    g |= pr2 & g >> 28;
    return g >> 7 & ~H_FILE;
}

Bitboard rook_attack_calc(ULL g, ULL p) {
    return north_attack(g, p)
        | east_attack(g, p)
        | south_attack(g, p)
        | west_attack(g, p);
}

Bitboard bishop_attack_calc(ULL g, ULL p) {
    return nw_attack(g, p)
        | ne_attack(g, p)
        | se_attack(g, p)
        | sw_attack(g, p);
}

Bitboard queen_attack_calc(ULL g, ULL p) {
    return rook_attack_calc(g, p) | bishop_attack_calc(g, p);
}

Bitboard knight_attack_calc(ULL g) {
    ULL attacks;
    attacks = ((g << 6) & ~A_FILE & ~B_FILE)
        | ((g >> 10) & ~A_FILE & ~B_FILE)
        | ((g >> 17) & ~A_FILE)
        | ((g >> 15) & ~H_FILE)
        | ((g >> 6) & ~G_FILE & ~H_FILE)
        | ((g << 10) & ~G_FILE & ~H_FILE)
        | ((g << 15) & ~A_FILE)
        | ((g << 17) & ~H_FILE);
    return attacks;
}

Bitboard king_attack_calc(ULL g) {
    return ((g << 9) & ~H_FILE)
        | g << 8
        | ((g << 7) & ~A_FILE)
        | ((g << 1) & ~H_FILE)
        | ((g >> 1) & ~A_FILE)
        | ((g >> 7) & ~H_FILE)
        | g >> 8
        | ((g >> 9) & ~A_FILE);
}

Bitboard pawn_attack_calc(ULL pawn, Side stm) {
    if (stm == WHITE) {
        return ((pawn << 9) & ~H_FILE) | ((pawn << 7) & ~A_FILE);
    } else {
        return ((pawn >> 9) & ~A_FILE) | ((pawn >> 7) & ~H_FILE);
    }
}
