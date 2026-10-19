#include <array>
#include <memory>
#include <string>
#include <vector>
#define private public
#include "cyber1_edu_sdk/mnn_model.h"
#undef private
#include "/home/lyz/桌面/cyberstartrainanddeploy/Cyberstar_rl_deploy/src/cyber1_edu_sdk/src/mnn_model.cpp"
#include "/space2/Project/rl_sar/src/rl_sar/library/core/observation_buffer/observation_buffer.hpp"
#include "/space2/Project/rl_sar/src/rl_sar/library/core/rl_sdk/stop_phase_clock.hpp"
#include <fstream>
int main(int argc, char **argv) {
  cyber1::MNNModel::Impl model;
  if (!model.init(argv[1]))
    return 1;
  std::ifstream in(argv[2], std::ios::binary);
  std::ofstream out(argv[3], std::ios::binary);
  StopPhaseClock clock;
  unsigned long long step = 0;
  // Rows: reset, command3, quaternion4, gyro3, jointpos12, jointvel12.
  float row[35];
  while (in.read((char *)row, sizeof(row))) {
    if (row[0]) {
      model.resetState();
      clock.Reset();
      step = 0;
    }
    RobotStatePacket state{};
    state.timestamp = 1000000 + (step + 1) * 20000;
    state.motion_mode = 2;
    std::copy(row + 4, row + 8, state.quaternion);
    std::copy(row + 8, row + 11, state.gyroscope);
    std::copy(row + 11, row + 23, state.joint_pos);
    std::copy(row + 23, row + 35, state.joint_vel);
    model.setCommand(row[1], row[2], row[3]);
    auto result = model.compute(state);
    float phase = clock.Update(++step, {row[1], row[2], row[3]}, .02f, .5f, .5f,
                               .5f, .05f, .05f);
    out.write((char *)model.history.data(), 282 * sizeof(float));
    out.write((char *)model.last_diagnostics.raw_action.data(),
              12 * sizeof(float));
    out.write((char *)result.joint_pos, 12 * sizeof(float));
    out.write((char *)&phase, sizeof(float));
  }
}
