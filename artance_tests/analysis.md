# close drawer

## /home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/part_analysis/variation0_episode3_frame051_wrist

- context 里面包含了 drawer handle 这种可以算作交互部件的内容。
- object 和 interaction part 和 suport part 中的 mask 都主要是多个 drawer，尽管这些类别使用了不同的 prompt。
- VLMs 的交互点不够理想，点在了拉开的抽屉的内部区域，而不是抽屉表面，然后导致 interaction part 选择了没有交互功能的抽屉的内部区域（一团空气！），suport part 和 object 分别选择了不同的抽屉。

## /home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/part_analysis/variation1_episode0_frame031_wrist

- 物体选择正确，object 是抽屉，interaction part 是 handle，但是 suport part 是抽屉的内部区域（一团空气！）

## /home/fubin/projects/artance/cap-x/artance_tests/close_drawer/outputs/part_analysis/variation2_episode0_frame071_wrist

- VLMs 的交互点 很理想，但是 interaction part 和 suport part 选择了错误的抽屉
- 在 SAM3 的原始预测中，interaction part 和 suport part 的最高得分选择是正确的，但是不知道为什么后续计算结果偏移到了另外的抽屉上

## 总结

- 4 个层级对于 close drawer 来说过于冗余了，而且有时候 SAM3 原本正确的 mask 被后续计算阶段搞错了，错误的合并、或者错误的互斥操作等等。
- 这个任务核心是要找到合适的接触点，然后以垂直 drawer 表面的方向推动即可。

# push button

## /home/fubin/projects/artance/cap-x/artance_tests/push_button/outputs/part_analysis/variation0_episode2_frame036_wrist

- interaction part 和 object 找的很对，分别是 button 和 button panel，但是 suport part 找成了桌面（prompt）
- 最终各个 mask 混合分析后，结果更加错误了
- VLMs 接触点稍微偏了一点，点在了 button panel 上

## /home/fubin/projects/artance/cap-x/artance_tests/push_button/outputs/part_analysis/variation2_episode2_frame032_wrist

- 三个主要 part 的 SAM3 初始结果中，最高分的结果都是正确的，但是最终计算的时候不知道为什么把 object 调整成为了 一个远处的机械臂夹爪，压根没有和 button 和 button panel 重叠。

## /home/fubin/projects/artance/cap-x/artance_tests/push_button/outputs/part_analysis/variation4_episode2_frame000_wrist

- 不知道为什么 object 找成了整个桌面
- 原始的 SAM3 结果没问题，单纯是后面分析错误

## 总结

- VLMs 基本能理解要找圆形的 button 和方形的 button panel 作为主要部件（足够用于交互分析）
- 选择 3 个部件似乎冗余了
- 引入 context 部件，可能会导致 mask 推理失败，总是有 SAM3 预测正确的 mask 在推理之后被错误的转变成全图 mask 或者某个错误 mask

# close fridge

## /home/fubin/projects/artance/cap-x/artance_tests/close_fridge/outputs/part_analysis/variation0_episode0_frame035_wrist

- 结果正确

## /home/fubin/projects/artance/cap-x/artance_tests/close_fridge/outputs/part_analysis/variation0_episode2_frame048_wrist

- 这个任务本身视角不好

## /home/fubin/projects/artance/cap-x/artance_tests/close_fridge/outputs/part_analysis/variation0_episode3_frame029_wrist

- interaction part 和 suport part 没有问题，object mask 也没问题，但是经过分析之后变成了空白

## 总结

基本上 interaction part，suport part 和 object 分别会找 handle、冰箱门、冰箱，但是某些视角下，object mask 会退化消失（可能是算作和 suport part 重叠了），实际操作中，知道 handle 和冰箱门的关系即可，因为冰箱门和冰箱的铰链部分本来就很难建模到。

# close microwave

## /home/fubin/projects/artance/cap-x/artance_tests/close_microwave/outputs/part_analysis/variation0_episode2_frame054_wrist

- 这个 case 非常完美，即便 VLMs 将接触点点在了桌面上，仍然准确找到了 handle 和微波炉门

## /home/fubin/projects/artance/cap-x/artance_tests/close_microwave/outputs/part_analysis/variation0_episode3_frame047_wrist

- suport part 明明找的是 door（SAM3 也找对了），但是最终分析完变成了 handle 周围的一圈离散点
- interaction part 找的是 handle（SAM3 也找对了），但是分析完之后面积扩大了（可能是合并了某个分数不佳的更差的 handle mask），然后就错了，包含了 handle 和门上的区域

## /home/fubin/projects/artance/cap-x/artance_tests/close_microwave/outputs/part_analysis/variation0_episode4_frame042_wrist

- 和第二个微波炉案例的结果类似

## 总结

- 对于这个任务 suport part 和 object 可能会重叠，因为目前的任务视角原因只会看到微波炉门，不会看到微波炉，如果后面真的可以看到微波炉和微波炉门，这样的三层分割才有意义
- 经常出现的问题是 SAM3 原本的分割（top 1）是准确的，但是分析完之后 suport part 和 interaction part 就错了

# toilet seat down

## /home/fubin/projects/artance/cap-x/artance_tests/toilet_seat_down/outputs/part_analysis/variation0_episode1_frame000_wrist

- 非常准确，尽管 VLMs 接触点点在了不够理想的位置

## /home/fubin/projects/artance/cap-x/artance_tests/toilet_seat_down/outputs/part_analysis/variation0_episode2_frame015_wrist

- suport part 被认为是马桶水箱
- 影响后续的推理

## /home/fubin/projects/artance/cap-x/artance_tests/toilet_seat_down/outputs/part_analysis/variation0_episode4_frame006_wrist

- 非常准确

## 总结

存在隐患，因为关闭马桶盖这个任务和前面的关门任务不太一样，那两个任务可能看不见冰箱和微波炉本体，也就看不见门和本体的夹角处铰链（打开的门通常是锐角角度），关马桶盖的这个任务，马桶盖向上打开90度，很容易看到马桶盖和马桶的夹角处铰链，因此选择了不同的旋转关节分析方式。但是对于关马桶这个任务来说，重点在于 suport part 必须选择为 toilet bowl 的这个部分，而不能是马桶水箱，否则就无法通过拟合铰链处旋转轴来进行交互。