#ifndef CALC_GAURD
#define CALC_GAURD

Bitboard west_attack(ULL g, ULL p);
Bitboard east_attack(ULL g, ULL p);
Bitboard north_attack(ULL g, ULL p);
Bitboard south_attack(ULL g, ULL p);
Bitboard ne_attack(ULL g, ULL p);
Bitboard nw_attack(ULL g, ULL p);
Bitboard se_attack(ULL g, ULL p);
Bitboard sw_attack(ULL g, ULL p);
Bitboard rook_attack_calc(ULL g, ULL p);
Bitboard bishop_attack_calc(ULL g, ULL p);
Bitboard queen_attack_calc(ULL g, ULL p);
Bitboard knight_attack_calc(ULL g);
Bitboard king_attack_calc(ULL g);
Bitboard pawn_attack_calc(ULL pawn, Side stm);

#endif
