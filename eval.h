#ifndef EVALUATOR_H_GUARD
#define EVALUATOR_H_GUARD

class Position;

class IEvaluator
{
 public:
  virtual Score evaluate(const Position&) = 0;
};

#endif
