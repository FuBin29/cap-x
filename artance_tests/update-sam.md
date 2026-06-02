你的调整是合理的。之前方案里有不少面向 fridge / microwave / handle 的 task prior，这对具体任务有效，但如果目标是做一个通用的 **Code-as-Policy 工具 API**，确实应该把这些强先验移出底层 mask selection 模块，让 VLM 负责语义层面的任务理解，底层模块只做**通用的一致性约束**。

下面是修改后的版本。

---

# 1. 总体设计原则修改

新的模块应该遵循这个原则：

```text
VLM 负责：任务理解、目标物体、交互部件、SAM3 prompts、接触点预测

Mask selection 模块负责：根据 VLM 输出和 SAM3 结果，做通用的候选整合、mask 层级约束、重叠消解、contact point 软引导

不在 mask selection 中写死 fridge / microwave / drawer / handle 等对象先验
```

也就是说，底层模块不应该知道：

```text
冰箱把手一般在门侧边
微波炉门可能是侧开或下拉
handle 应该细长
door 应该大面积
抽屉面板应该是平面
```

这些都属于 task/object prior，应该由 VLM 的结构化输出或后续 motion module 处理，而不是写进 mask selection 的通用规则里。

新的目标是：

```text
给定 VLM 结构化输出 + SAM3 多 prompt 候选 masks + contact point，
选择一组语义一致、层级一致、尽量不重叠、受 contact point 软引导的 object / part masks。
```

---

# 2. VLM 结构化输出应该怎么设计

你之前的 prompt 只让 VLM 输出一个 `[y, x]`。这个非常干净，适合单独做 contact point 预测。但是现在你希望 VLM 同时生成 SAM3 层级 prompts，因此建议把 VLM 输出分成两个阶段，而不是一次性全部输出。

## 2.1 阶段一：输出结构化交互感知结果

这个阶段让 VLM 输出：

```json
{
  "target_object": "...",
  "interaction_part": "...",
  "support_part": "...",
  "contact_pixel": [y, x],
  "sam3_prompts": {
    "object": ["..."],
    "interaction_part": ["..."],
    "support_part": ["..."],
    "context": ["..."]
  }
}
```

其中：

* `target_object`：当前任务主要操作的物体；
* `interaction_part`：机器人应该直接接触的部件；
* `support_part`：与交互部件相关、用于建立层级关系的较大部件；
* `contact_pixel`：归一化坐标 `[y, x]`；
* `sam3_prompts.object`：整体物体 prompt；
* `sam3_prompts.interaction_part`：直接交互部件 prompt；
* `sam3_prompts.support_part`：承载或包含 interaction part 的部件 prompt；
* `sam3_prompts.context`：可选，用于辅助排除或理解周围结构，但底层模块不强依赖。

例如关闭冰箱门可能输出：

```json
{
  "target_object": "fridge",
  "interaction_part": "fridge door handle",
  "support_part": "fridge door",
  "contact_pixel": [527, 320],
  "sam3_prompts": {
    "object": ["fridge", "refrigerator"],
    "interaction_part": ["fridge door handle", "refrigerator door handle", "handle"],
    "support_part": ["fridge door", "refrigerator door"],
    "context": ["fridge body", "cabinet frame"]
  }
}
```

打开微波炉门可能输出：

```json
{
  "target_object": "microwave",
  "interaction_part": "microwave door handle",
  "support_part": "microwave door",
  "contact_pixel": [527, 224],
  "sam3_prompts": {
    "object": ["microwave"],
    "interaction_part": ["microwave door handle", "door handle", "handle"],
    "support_part": ["microwave door", "door"],
    "context": ["microwave body", "microwave front panel"]
  }
}
```

注意，这里并不是底层模块知道“handle 应该细长”，而是 VLM 看到图像和任务后，自己判断当前应该查哪些 prompt。

---

# 3. 推荐的 VLM prompt 模板

可以在你原来的 prompt 基础上扩展，但仍然要求输出严格 JSON。

## 3.1 结构化输出 prompt

```text
You are a robotic vision and manipulation perception module.

Task:
Analyze the provided image and prepare semantic grounding information for a robot manipulation task:
"{TASK_DESCRIPTION}"

Your job:
1. Identify the main target object involved in the task.
2. Identify the interaction part that the robot should directly touch.
3. Identify the support part that contains, supports, or is physically associated with the interaction part.
4. Predict one best contact point for the robot end-effector.
5. Generate hierarchical SAM3 text prompts for segmenting the object and its relevant parts.

Definitions:
- target_object: the main object being manipulated.
- interaction_part: the specific visible part that the robot should directly contact.
- support_part: the larger part that contains or supports the interaction_part.
- contact_pixel: the best point for robot contact, using normalized integer coordinates [y, x].
- sam3_prompts.object: prompts for the whole target object.
- sam3_prompts.interaction_part: prompts for the direct contact part.
- sam3_prompts.support_part: prompts for the larger related part.
- sam3_prompts.context: optional prompts for nearby relevant structures.

Contact Point Rules:
- Choose exactly one point that is physically suitable for the robot to touch.
- The point should lie on the visible interaction part if it is visible.
- If the interaction part is not clearly visible, choose a visible rigid area of the target object suitable for the task.
- The point must not lie on the robot, floor, background, or unrelated objects.
- If multiple valid points exist, choose the most stable and central visible point on the interaction part.

SAM3 Prompt Rules:
- Prompts should be short noun phrases.
- Prompts should be image-specific and task-relevant.
- Include 1 to 3 prompts for each category.
- Use more specific prompts before generic prompts.
- Do not include action verbs such as open, close, push, or pull in SAM3 prompts.
- Do not include coordinate descriptions in SAM3 prompts.
- Avoid prompts for irrelevant background objects unless they are useful context.

Coordinate System:
- Use normalized integer coordinates [y, x].
- [0, 0] is the top-left corner of the image.
- [1000, 1000] is the bottom-right corner of the image.

Output Format:
Output ONLY a valid JSON object in the following schema:

{
  "target_object": "string",
  "interaction_part": "string",
  "support_part": "string",
  "contact_pixel": [int, int],
  "sam3_prompts": {
    "object": ["string"],
    "interaction_part": ["string"],
    "support_part": ["string"],
    "context": ["string"]
  }
}

Do NOT include any explanation, markdown, comments, or additional text.
```

这个 prompt 的特点是：
它把 task-specific 语义交给 VLM，但输出仍然足够结构化，方便后面的程序调用。

---

# 4. 是否还需要单独的 contact point prompt？

建议保留。

你现在的 contact point prompt 很干净，输出稳定。我的建议是使用两种模式：

## 模式 A：快速模式

只需要 contact point 时，使用你原来的 prompt：

```text
Output ONLY one coordinate in this exact format: [y, x]
```

用于快速测试、对比、消融。

## 模式 B：结构化模式

用于完整 pipeline：

```text
Output target_object / interaction_part / support_part / contact_pixel / SAM3 prompts
```

这两个模式可以做一致性检查。

例如：

```python
contact_only = vlm_predict_contact_point(image, task)
structured = vlm_predict_structured_grounding(image, task)

if distance(contact_only, structured["contact_pixel"]) > threshold:
    mark_as_uncertain()
```

如果两个 VLM 调用给出的 contact point 差异很大，说明该图像/任务本身存在不确定性，可以触发重试或人工检查。

---

# 5. SAM3 层级 prompt 由 VLM 生成

你提出的第 2 点很重要：
不希望为每个 task 手写 SAM3 prompt。

因此新的 pipeline 应该是：

```text
image + task
  ↓
VLM outputs:
  target_object
  interaction_part
  support_part
  contact_pixel
  hierarchical SAM3 prompts
  ↓
SAM3 runs all prompts
  ↓
universal mask selection module
```

也就是说，SAM3 prompt generation 是 VLM 的职责。

底层模块只知道这些 prompt 的角色：

```python
sam3_prompts = {
    "object": [...],
    "interaction_part": [...],
    "support_part": [...],
    "context": [...]
}
```

而不需要知道“fridge”“microwave”“handle”的具体含义。

---

# 6. 修改后的 mask selection 原则

根据你的第 3 点，mask selection 中只保留以下通用因素：

1. **object-part 层级约束**
   interaction_part / support_part 应该尽量位于 object mask 内部或附近。

2. **part-support 层级约束**
   interaction_part 应该尽量位于 support_part 内部、边界附近，或与 support_part 有空间重叠/邻接。

3. **mask 不重叠约束**
   多 prompt 得到的 mask 需要做冲突消解，避免同一像素被多个语义占用。

4. **SAM3 score**
   使用 SAM3 自身分数，但不要完全依赖，因为不同 prompt 分数可能不完全可比。

5. **contact point 软引导**
   contact point 用于提高附近候选的分数，但不是硬约束。

6. **不使用具体对象和部件形状先验**
   不写死 handle 细长、door 大平面、fridge 侧边等规则。

---

# 7. 修改后的通用 scoring

对于每个候选 mask，保存：

```python
candidate = {
    "mask": mask,
    "prompt": prompt,
    "role": role,  # object / interaction_part / support_part / context
    "sam_score": score,
    "area": area,
    "bbox": bbox
}
```

然后计算通用分数：

```python
score(mask) =
    w_sam * sam_score
  + w_role * role_score
  + w_contact * contact_score
  + w_hierarchy * hierarchy_score
  - w_overlap * conflict_penalty
```

这里没有任何 object-specific prior。

---

## 7.1 role score

role score 来自 VLM 输出的角色，而不是人工 task prior。

例如：

```python
role_score = {
    "interaction_part": 1.0,
    "support_part": 0.7,
    "object": 0.5,
    "context": 0.2
}
```

但注意，这个分数不是说 interaction part 一定最好，而是表示当前目标是选择直接交互区域，所以 interaction part 角色天然更接近最终 target。

---

## 7.2 contact score

contact point 只作为 2D 软引导。

不用强依赖 3D contact，因为你提到的情况很关键：

```text
contact pixel 可能落在物体镂空区域上；
例如微波炉把手中间是空的；
反投影 3D 点可能落到背景或内部错误深度。
```

因此新的设计中：

```text
contact_3d 不用于 mask selection 的核心评分。
```

只使用 2D 距离：

```python
d2d = distance_to_mask(contact_pixel, mask)
contact_score = exp(-d2d / sigma)
```

其中 `sigma` 可以是通用的自适应尺度：

```python
sigma = alpha * sqrt(mask_area)
```

或者：

```python
sigma = alpha * max(mask_bbox_width, mask_bbox_height)
```

这样不需要知道这个 mask 是 handle 还是 door。

如果 contact point 在 mask 内：

```python
contact_score += inside_bonus
```

但 inside 也只是 bonus，不是硬条件。

---

## 7.3 hierarchy score

这是通用性的核心。

你不希望写死 handle / door 规则，但可以写通用层级关系：

```text
interaction_part 应该属于 support_part
support_part 应该属于 object
interaction_part 也应该与 object 有较高关联
```

这里的“属于”不是严格包含，可以用 2D mask 关系表示：

```python
containment(A, B) = area(A ∩ B) / area(A)
```

表示 A 有多少比例落在 B 内。

对于 interaction_part candidate `p`、support candidate `s`、object candidate `o`：

```python
hierarchy_score =
    containment(p, s)
  + containment(s, o)
  + containment(p, o)
```

但是要允许边界误差，所以可以先 dilate 上层 mask：

```python
containment(p, dilate(s))
containment(s, dilate(o))
containment(p, dilate(o))
```

这样把手稍微凸出门板、门板稍微超出整体物体 mask 的情况不会被过度惩罚。

---

## 7.4 overlap conflict penalty

如果一个 candidate 与同层其他候选严重重叠，但分数较低，可以降权。

例如同属于 `interaction_part` 的多个 prompt：

```text
"microwave door handle"
"door handle"
"handle"
```

它们可能指向同一个区域。这个不一定是坏事，反而是证据增强。

所以要区分两类 overlap：

### 同角色 overlap

同角色重叠可以合并或互相增强：

```text
同样都是 interaction_part，多个 prompt 命中同一区域 → 说明该区域更可信
```

### 不同角色 overlap

不同角色重叠需要冲突解决：

```text
interaction_part 和 support_part 重叠 → 可接受，但最终像素分配时需要互斥
support_part 和 object 重叠 → 可接受
interaction_part 和 context 大面积重叠 → 需要降权或重新分配
```

因此不建议一开始直接惩罚所有重叠，而是先做 candidate grouping。

---

# 8. 修改后的 candidate grouping

多个 prompt 经常会返回几乎相同的 mask。比如：

```text
"microwave door handle"
"door handle"
"handle"
```

可能都返回同一个把手区域。

建议先把候选按 IoU 聚类：

```python
if IoU(mask_i, mask_j) > 0.7:
    group them
```

每个 group 保存：

```python
group = {
    "merged_mask": union or best mask,
    "roles": [...],
    "prompts": [...],
    "score": max or weighted average score,
    "support_count": number of matched prompts
}
```

同角色/近似同语义的多个 prompt 支持同一个区域时，给一个 support bonus：

```python
group_score += lambda_support * log(1 + support_count)
```

这个是通用的，不依赖具体物体。

---

# 9. 修改后的 target-support-object 联合选择

不要单独选最近 mask，而是选一个三元组：

```text
object mask O
support mask S
interaction mask P
```

目标是最大化：

```python
Score(O, S, P) =
    base(O) + base(S) + base(P)
  + λ1 * containment(P, dilate(S))
  + λ2 * containment(S, dilate(O))
  + λ3 * containment(P, dilate(O))
  + λ4 * contact_score(P)
```

这里：

* `P` 是 interaction_part；
* `S` 是 support_part；
* `O` 是 object；
* contact point 主要引导 `P`，也可以弱引导 `S`；
* 没有使用任何 handle/door/fridge/microwave 先验。

如果 object mask 不存在，允许退化为：

```python
Score(S, P) =
    base(S) + base(P)
  + λ1 * containment(P, dilate(S))
  + λ2 * contact_score(P)
```

如果 support mask 也不存在，则退化为：

```python
Score(P) =
    base(P) + λ * contact_score(P)
```

这个退化机制很重要，因为 SAM3 可能分不到完整 object 或 support。

---

# 10. 修改后的 overlap resolution

在选择出 `O, S, P` 后，再做互斥化。

这里不要用 object-specific priority，而用 role priority：

```python
role_priority = {
    "interaction_part": 3.0,
    "support_part": 2.0,
    "object": 1.0,
    "context": 0.0
}
```

像素级分数：

```python
pixel_score_i(y, x) =
    sam_score_i
  + λ_role * role_priority_i
  + λ_contact * contact_heatmap(y, x)
```

但是 contact heatmap 主要应该加给 `interaction_part`，否则整个 object mask 靠近 contact point 的区域也会被抬高。

可以这样写：

```python
if role == "interaction_part":
    pixel_score += λ_contact * contact_heatmap
elif role == "support_part":
    pixel_score += 0.3 * λ_contact * contact_heatmap
else:
    pixel_score += 0.0
```

这样重叠区域会优先分给 interaction_part，但不会让 contact point 完全支配结果。

---

# 11. 关于 contact pixel 落在镂空区域的问题

这是你指出的第 4 点，非常重要。

例如微波炉把手是环形或有空洞结构：

```text
VLM contact point 可能落在把手内部空洞；
该像素的 depth 来自背景；
contact point 不在 handle mask 内；
如果强制 inside mask 或 contact_3d，会错误筛掉正确 handle。
```

所以新的规则应该是：

```text
contact point 只表示“交互区域附近”，不要求它必须落在 mask 内部。
```

具体设计：

1. 不使用 3D contact 作为核心筛选；
2. 不要求 contact point inside mask；
3. 使用 distance-to-mask，而不是 point-in-mask；
4. 对 contact score 使用平滑函数；
5. 允许接触点附近的一组 mask 竞争，而不是最近 mask 胜出。

例如：

```python
d = distance_transform(~mask)[contact_y, contact_x]
contact_score = exp(-d / sigma)
```

其中 `sigma` 取候选 mask 尺度：

```python
sigma = 0.15 * max(mask_bbox_width, mask_bbox_height)
```

或者：

```python
sigma = 0.25 * sqrt(mask_area)
```

这样如果 contact point 落在 handle 空洞中，只要离 handle mask 边界很近，handle 仍然会得到较高分数。

---

# 12. 修改后的完整 pipeline

最终通用 pipeline 可以写成：

```text
Input:
  image, depth, task_description

Step 1:
  VLM outputs structured grounding:
    target_object
    interaction_part
    support_part
    contact_pixel
    hierarchical SAM3 prompts

Step 2:
  Run SAM3 for all prompts:
    object prompts
    interaction_part prompts
    support_part prompts
    context prompts

Step 3:
  Build candidate masks:
    record prompt, role, SAM score, mask

Step 4:
  Group near-duplicate masks:
    group masks by IoU
    aggregate scores and prompt support

Step 5:
  Select object-support-interaction tuple:
    maximize generic score based on:
      SAM score
      role score
      contact-to-mask distance
      containment hierarchy
      prompt support count

Step 6:
  Resolve overlap:
    use role priority:
      interaction_part > support_part > object > context
    add contact heatmap only mainly to interaction_part
    assign each pixel to only one selected role

Step 7:
  Output:
    interaction mask
    support mask
    object mask
    confidence
    contact pixel
```

---

# 13. 伪代码版本

```python
def select_masks_general(
    image,
    depth,
    task_description,
    vlm_output,
    sam3_results,
):
    """
    Generic task-conditioned mask selection.
    No object-specific or part-shape prior is used.
    """

    contact_px = vlm_output["contact_pixel"]
    prompts = vlm_output["sam3_prompts"]

    # 1. Build candidates
    candidates = []
    for result in sam3_results:
        c = Candidate(
            mask=result.mask,
            sam_score=result.score,
            prompt=result.prompt,
            role=result.role,  # object / interaction_part / support_part / context
        )
        c.area = mask_area(c.mask)
        c.bbox = mask_bbox(c.mask)
        c.contact_dist = distance_to_mask(contact_px, c.mask)
        c.contact_score = compute_soft_contact_score(
            c.contact_dist,
            c.area,
            c.bbox
        )
        candidates.append(c)

    # 2. Group near-duplicate candidates
    groups = group_by_iou(candidates, iou_thr=0.7)

    for g in groups:
        g.mask = aggregate_masks(g.members)
        g.sam_score = aggregate_sam_score(g.members)
        g.roles = aggregate_roles(g.members)
        g.primary_role = choose_primary_role(g.roles)
        g.prompt_support = len(g.members)
        g.contact_dist = distance_to_mask(contact_px, g.mask)
        g.contact_score = compute_soft_contact_score(
            g.contact_dist,
            mask_area(g.mask),
            mask_bbox(g.mask)
        )
        g.base_score = (
            1.0 * normalize(g.sam_score)
          + 0.5 * np.log(1 + g.prompt_support)
          + 1.0 * role_score(g.primary_role)
        )

    # 3. Split by role
    O_candidates = [g for g in groups if has_role(g, "object")]
    S_candidates = [g for g in groups if has_role(g, "support_part")]
    P_candidates = [g for g in groups if has_role(g, "interaction_part")]

    # 4. Select best O-S-P tuple
    best = None
    best_score = -float("inf")

    for P in P_candidates:
        for S in S_candidates or [None]:
            for O in O_candidates or [None]:

                score = P.base_score + 2.0 * P.contact_score

                if S is not None:
                    score += S.base_score
                    score += 1.5 * containment(P.mask, dilate(S.mask))

                if O is not None:
                    score += O.base_score
                    if S is not None:
                        score += 1.0 * containment(S.mask, dilate(O.mask))
                    score += 1.0 * containment(P.mask, dilate(O.mask))

                if score > best_score:
                    best_score = score
                    best = (O, S, P)

    O, S, P = best

    # 5. Resolve overlap among selected masks
    selected = []
    if O is not None:
        selected.append(O)
    if S is not None:
        selected.append(S)
    selected.append(P)

    final_masks = resolve_overlap_by_role_and_contact(
        selected,
        contact_px
    )

    return {
        "object_mask": final_masks.get("object", None),
        "support_mask": final_masks.get("support_part", None),
        "interaction_mask": final_masks["interaction_part"],
        "contact_pixel": contact_px,
        "confidence": best_score,
        "selected_prompts": {
            "object": O.prompts if O else [],
            "support_part": S.prompts if S else [],
            "interaction_part": P.prompts
        }
    }
```

---

# 14. overlap resolution 的通用伪代码

```python
def resolve_overlap_by_role_and_contact(selected_groups, contact_px):
    H, W = selected_groups[0].mask.shape
    score_stack = []
    ids = []

    contact_heatmap = make_contact_heatmap(H, W, contact_px)

    role_priority = {
        "interaction_part": 3.0,
        "support_part": 2.0,
        "object": 1.0,
        "context": 0.0,
    }

    for g in selected_groups:
        score = np.full((H, W), -np.inf)
        inside = g.mask > 0

        base = (
            1.0 * normalize(g.sam_score)
          + 1.0 * role_priority[g.primary_role]
        )

        score[inside] = base

        if g.primary_role == "interaction_part":
            score[inside] += 1.0 * contact_heatmap[inside]
        elif g.primary_role == "support_part":
            score[inside] += 0.3 * contact_heatmap[inside]

        score_stack.append(score)
        ids.append(g.primary_role)

    score_stack = np.stack(score_stack, axis=0)
    owner = np.argmax(score_stack, axis=0)
    valid = np.max(score_stack, axis=0) > -np.inf

    final_masks = {}
    for i, role in enumerate(ids):
        final_masks[role] = (owner == i) & valid

    return final_masks
```

---

# 15. 为什么这个版本更通用

修改后的版本不再使用这些规则：

```text
handle 应该细长
door 应该大
fridge handle 应该在门侧边
microwave hinge 在左边或下边
drawer front 是平面
```

而只使用这些通用规则：

```text
interaction part 应该接近 contact point
interaction part 应该和 support part 有层级关系
support part 应该和 object 有层级关系
多个 prompt 支持同一区域时置信度提高
SAM3 score 较高的候选更可信
最终 masks 应该尽量互斥
```

这更符合你希望的通用模块设计。

---

# 16. 修改后的 API 设计

可以把整个模块拆成三个 API：

```python
vlm_ground_interaction(image, task_description)
```

输出：

```python
{
    "target_object": ...,
    "interaction_part": ...,
    "support_part": ...,
    "contact_pixel": ...,
    "sam3_prompts": ...
}
```

---

```python
run_sam3_hierarchical(image, sam3_prompts)
```

输出：

```python
[
    {
        "mask": ...,
        "score": ...,
        "prompt": ...,
        "role": "interaction_part"
    },
    ...
]
```

---

```python
select_interaction_masks(sam3_results, vlm_output)
```

输出：

```python
{
    "object_mask": ...,
    "support_mask": ...,
    "interaction_mask": ...,
    "contact_pixel": ...,
    "confidence": ...,
    "selected_prompts": ...
}
```

其中第三个 API 是完全通用的，不包含具体 task/object 规则。

---

# 17. 更推荐的最终模块定位

你可以把该模块命名为：

```text
Task-Conditioned Hierarchical Mask Selection
```

或者：

```text
Contact-Guided Hierarchical Part Grounding
```

它的输入输出关系是：

```text
VLM task grounding
+ SAM3 open-vocabulary candidate masks
+ contact point soft guidance
↓
consistent object / support / interaction masks
```

这个模块解决的是：

```text
多 prompt 查询结果不统一、mask 重叠、最近 contact mask 错选、接触点落在空洞区域、SAM3 score 跨 prompt 不稳定
```

但它不解决：

```text
该任务应该抓哪里
该物体是什么关节
handle 应该是什么形状
旋转轴在哪里
轨迹如何生成
```

这些留给 VLM grounding、joint inference 和 motion primitive 模块。

---

# 18. 修改后的核心结论

根据你的要求，最终方案应该从之前的：

```text
task-aware target-support pair selection
+ object/part shape priors
+ 2D/3D geometric consistency
```

修改为：

```text
VLM-generated hierarchical prompts
+ universal object-support-interaction hierarchy
+ SAM3 score aggregation
+ contact-point soft guidance
+ role-based overlap resolution
```

新的 mask selection 不再写死任何关于 fridge、microwave、drawer、handle 的几何或位置先验。

最关键的变化是：

```text
VLM 负责产生任务相关语义；
SAM3 负责产生开放词汇候选 mask；
通用 mask selection 只负责把这些候选整理成层级一致、互斥、接触点附近合理的部件 mask。
```

这样更符合你的 code-as-policy 框架：
VLM 负责“看图理解任务和生成工具调用参数”，底层 API 负责“尽可能通用、可组合、可解释地执行几何处理”。


---

# 2026-05-29 实现落地记录

本轮已在保留原版 `contact_point -> sam3 -> sam3_point_selection` 最近 mask 方案的基础上，新增一套完整的 **Contact-Guided Hierarchical Part Grounding** 实现。旧方案仍可用于快速对比和消融；新方案负责完整部件分析。

## 已完成步骤

1. 新增结构化 VLM grounding：`common/vlm_interaction_grounding.py`

   输入为图像和 task prompt spec，输出严格 JSON：

   ```json
   {
     "target_object": "...",
     "interaction_part": "...",
     "support_part": "...",
     "contact_pixel": [y, x],
     "sam3_prompts": {
       "object": ["..."],
       "interaction_part": ["..."],
       "support_part": ["..."],
       "context": ["..."]
     }
   }
   ```

   模块会保存 `structured_grounding_summary.json`，并同时记录：

   - VLM 原始响应；
   - normalized `[y, x]` 接触点；
   - 内部统一使用的 normalized `[x, y]`；
   - 像素坐标 `pixel_xy`；
   - 按角色组织的 SAM3 prompts。

2. 新增通用层级部件分析：`common/hierarchical_part_analysis.py`

   该模块执行：

   ```text
   VLM structured grounding
     -> flatten role prompts
     -> SAM3 text prompts
     -> build role-aware candidates
     -> IoU grouping
     -> object/support/interaction tuple scoring
     -> role-based overlap resolution
     -> final object/support/interaction masks
   ```

   scoring 只包含通用约束：

   - SAM3 score；
   - role score；
   - prompt support count；
   - contact point 到 mask 的 soft distance；
   - `interaction_part -> support_part -> object` 的 dilated containment；
   - role priority overlap resolution。

   没有写入 fridge、microwave、drawer、handle 的形状、大小、位置或关节先验。

3. 新增 CLI 模块：`structured_grounding` 和 `part_analysis`

   统一入口：

   ```bash
   cd /home/fubin/projects/artance/cap-x/artance_tests

   uv run --no-sync --active python run_task_module.py structured_grounding \
     --task close_fridge \
     --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_fridge/variation0/episodes/episode0/wrist_rgb/35.png \
     --model gemini-2.5-pro \
     --server-url http://127.0.0.1:8110/chat/completions

   uv run --no-sync --active python run_task_module.py part_analysis \
     --task close_fridge \
     --episode-line /home/fubin/projects/artance/RLBench/data/RLBench-data/close_fridge/variation0/episodes/episode0/wrist_rgb/35.png \
     --model gemini-2.5-pro \
     --vlm-server-url http://127.0.0.1:8110/chat/completions \
     --sam3-service-url http://127.0.0.1:8114 \
     --top-k 5 \
     --no-show
   ```

   如果已经有结构化 grounding summary，可以跳过 VLM：

   ```bash
   uv run --no-sync --active python run_task_module.py part_analysis \
     --task close_fridge \
     --episode-line /path/to/frame.png \
     --grounding-summary close_fridge/outputs/structured_grounding/.../structured_grounding_summary.json \
     --sam3-service-url http://127.0.0.1:8114 \
     --top-k 5 \
     --no-show
   ```

4. 新增 batch 支持

   `run_batch_module_tests.sh` 现在支持：

   ```bash
   TASKS="close_fridge close_microwave" \
   MODULES="structured_grounding part_analysis" \
   bash run_batch_module_tests.sh
   ```

   默认 batch 仍保持旧链路：`contact_point sam3 sam3_point_selection pointcloud`。

5. 点云模块已支持新输出

   `common/pointcloud_reconstruction.py` 新增：

   ```python
   load_part_masks_from_part_analysis_summary(...)
   ```

   可直接读取 `part_analysis_summary.json` 的 final masks。现在推荐用 `--mask-source part_analysis`，episode 输入会自动推导 summary 路径：

   ```bash
   uv run --no-sync --active python run_task_module.py pointcloud \
     --task close_fridge \
     --episode-line /path/to/frame.png \
     --mask-source part_analysis \
     --mask-only
   ```

   `--mask-source auto` 为默认值，自动优先级是：

   ```text
   part_analysis_summary.json -> point_selection_summary.json -> sam3/summary.json
   ```

   批量脚本可用 `POINTCLOUD_MASK_SOURCE=part_analysis` 强制新版，或用 `POINTCLOUD_MASK_SOURCE=point_selection` 强制旧版。

## 新输出文件

`part_analysis` 默认输出到：

```text
<task>/outputs/part_analysis/<episode_key>/<image_stem>/
```

主要文件：

- `structured_grounding_summary.json`: VLM 结构化 grounding；
- `hierarchical_sam3_summary.json`: 按 role/prompt 保存的 SAM3 top-k 结果和 mask 路径；
- `part_analysis_summary.json`: 候选、group、评分、选中部件和 final mask 索引；
- `part_analysis_overlay.png`: object/support/interaction final masks 叠加可视化；
- `selected_masks/*_interaction_part_mask.npy|png`；
- `selected_masks/*_support_part_mask.npy|png`；
- `selected_masks/*_object_mask.npy|png`。

## 与旧方案的关系

旧方案保留：

```text
VLM contact point
  -> 手写 task sam3_prompts
  -> 每个 prompt 选择离接触点最近的 mask
```

新方案并行新增：

```text
VLM structured grounding
  -> VLM-generated hierarchical SAM3 prompts
  -> generic candidate grouping
  -> object/support/interaction joint selection
  -> role-based mutually exclusive masks
```

后续实验建议先对同一帧同时跑旧方案和新方案，对比：

- interaction mask 是否稳定落在可交互部件；
- support/object mask 是否层级一致；
- 接触点落在空洞或 mask 外时，新方案是否仍能选中合理部件；
- 点云/运动模块读取 final masks 后的几何质量。
