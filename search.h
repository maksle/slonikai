#ifndef SEARCH_GAURD_H
#define SEARCH_GAURD_H

class Position;

namespace Search { 
  
  template<bool Root = true> int perft(Position& pos, int depth);
  
}

#endif
