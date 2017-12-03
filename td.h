#ifndef TD_H_GUARD
#define TD_H_GUARD

namespace TD {

  void initialize(int valid_offset, int valid_num,
                  int train_offset, int train_num,
                  int batch_size, int valid_frequency);
  void play();
}

#endif
