"""合格切件错缝拼板预览：把选中的合格切件拼成若干等长行（纯计算）。

规则
----
* 切件长度取自批次冻结方案；等宽木条、接头无长度损耗、不裁短；
* 每行从左至右拼到目标行长（累计长度恰好等于目标行长），每件最多用一次，
  允许余件；
* 接缝位置为除末件外的累计长度；相邻行任意两接缝位置差的绝对值须不小于
  最小错缝距离；无接缝行（单件行）不限制邻行；
* 精确求解用件数最少的可行排布；并列时逐行比较切件标识序列取字典序最小
  （标识先比板号再比切件号，短前缀在前）。

选件 1～10 件、行数 2～4。求解按行深度优先枚举：子集和预计算 +
「剩余行最少件数」松弛下界（忽略错缝距离）做件数剪枝，当前行的接缝在
拼件追加时即与上一行接缝核对，字典序前缀剪枝保证并列时取到字典序最小者。
本模块不读写数据库、不修改批次。
"""


def plan_joint(pieces, target_length, row_count, min_stagger):
    """精确求解错缝拼板排布。

    参数
    ----
    pieces        : list  每项含 ``board_no``、``cut_no``、``length``（正整数）；
                    调用方须保证 1～10 件且标识无重复
    target_length : int   目标行长（正整数）
    row_count     : int   行数（2～4）
    min_stagger   : int   最小错缝距离（非负整数）

    返回
    ----
    dict
        可行时 ``feasible=True``，含 ``used_count``（用件数）、逐行 ``rows``
        （``row_no``、按拼接顺序的 ``pieces``、``joints`` 接缝坐标、
        ``total_length``）与未用切件 ``unused``；不可行时
        ``feasible=False`` 并给出 ``reason``。
    """
    ordered = sorted(pieces, key=lambda p: (p["board_no"], p["cut_no"]))
    n = len(ordered)
    ids = [(p["board_no"], p["cut_no"]) for p in ordered]
    lens = [p["length"] for p in ordered]
    target = target_length
    rows_total = row_count
    gap = min_stagger
    full = (1 << n) - 1
    inf = float("inf")

    # 全部子集的件长和。
    sub_sum = [0] * (1 << n)
    for mask in range(1, 1 << n):
        lsb = mask & (-mask)
        sub_sum[mask] = sub_sum[mask ^ lsb] + lens[lsb.bit_length() - 1]

    if sub_sum[full] < rows_total * target:
        return {
            "feasible": False,
            "reason": (
                f"所选切件总长度 {sub_sum[full]} 不足 {rows_total} 行目标行长的"
                f"合计 {rows_total * target}，无解。"
            ),
        }

    # g[mask][k]：忽略错缝距离，用 mask 中的切件拼出 k 行的最少件数
    # （每行累计长度恰为目标行长）；不可行为 inf。它是真实问题的件数下界。
    g = [[inf] * (rows_total + 1) for _ in range(1 << n)]
    for mask in range(1 << n):
        g[mask][0] = 0
    for k in range(1, rows_total + 1):
        for mask in range(1 << n):
            best = inf
            sub = mask
            while sub:
                if sub_sum[sub] == target:
                    cand = sub.bit_count() + g[mask ^ sub][k - 1]
                    if cand < best:
                        best = cand
                sub = (sub - 1) & mask
            g[mask][k] = best

    if g[full][rows_total] == inf:
        return {
            "feasible": False,
            "reason": (
                f"所选切件无法分出 {rows_total} 组累计长度均恰为 {target} 的"
                "切件（尚未考虑错缝距离），无解。"
            ),
        }

    # min_rest(mask, need, rows_left)：先用 mask 中的切件把当前行补齐
    # （还差 need），再拼 rows_left 整行，所需的最少件数（忽略错缝距离）。
    rest_cache = {}

    def min_rest(mask, need, rows_left):
        key = (mask, need, rows_left)
        cached = rest_cache.get(key)
        if cached is not None:
            return cached
        best = inf
        sub = mask
        while sub:
            if sub_sum[sub] == need:
                tail = g[mask ^ sub][rows_left]
                if tail != inf:
                    cand = sub.bit_count() + tail
                    if cand < best:
                        best = cand
            sub = (sub - 1) & mask
        rest_cache[key] = best
        return best

    best_count = None
    best_key = None     # 逐行切件标识元组；Python 元组比较即题目字典序
    best_rows = None    # 逐行切件下标

    rows_acc = []       # 已完成行（切件下标列表）
    cur_row = []        # 当前行（切件下标列表）

    def lex_worse():
        """当前部分解的首个不同行已大于最优解对应行时为 True。"""
        for r, row in enumerate(rows_acc):
            a = tuple(ids[i] for i in row)
            b = best_key[r]
            if a != b:
                return a > b
        if cur_row:
            a = tuple(ids[i] for i in cur_row)
            b = best_key[len(rows_acc)]
            for x, y in zip(a, b):
                if x != y:
                    return x > y
            # 当前行尚未拼完、必然继续追加：已不短于最优对应行时，
            # 追加后只会更长（短前缀在前），字典序必败。
            if len(a) >= len(b):
                return True
        return False

    def dfs(row_idx, cur_sum, used_mask, prev_joints, used_count):
        nonlocal best_count, best_key, best_rows

        if row_idx == rows_total:
            key = tuple(tuple(ids[i] for i in row) for row in rows_acc)
            if (
                best_count is None
                or used_count < best_count
                or (used_count == best_count and key < best_key)
            ):
                best_count = used_count
                best_key = key
                best_rows = [row[:] for row in rows_acc]
            return

        remaining = full ^ used_mask
        lb = min_rest(remaining, target - cur_sum, rows_total - row_idx - 1)
        if lb == inf:
            return
        if best_count is not None:
            if used_count + lb > best_count:
                return  # 件数必败
            if used_count + lb >= best_count and lex_worse():
                return  # 件数无法严格更少且字典序已败

        need = target - cur_sum
        for i in range(n):
            bit = 1 << i
            if used_mask & bit or lens[i] > need:
                continue
            new_sum = cur_sum + lens[i]
            if new_sum < target:
                # 新接缝位于 new_sum：与上一行全部接缝核对错缝距离。
                if any(abs(new_sum - j) < gap for j in prev_joints):
                    continue
                cur_row.append(i)
                dfs(row_idx, new_sum, used_mask | bit, prev_joints, used_count + 1)
                cur_row.pop()
            else:
                # 本行完成：接缝为除末件外的累计长度。
                cur_row.append(i)
                joints = []
                acc = 0
                for p in cur_row[:-1]:
                    acc += lens[p]
                    joints.append(acc)
                completed = cur_row[:]
                rows_acc.append(completed)
                cur_row.clear()  # 下一行从空行重新拼
                dfs(row_idx + 1, 0, used_mask | bit, joints, used_count + 1)
                cur_row.extend(completed)
                rows_acc.pop()
                cur_row.pop()

    dfs(0, 0, 0, (), 0)

    if best_rows is None:
        return {
            "feasible": False,
            "reason": (
                f"所选切件虽能拼出 {rows_total} 行长度 {target} 的拼板，"
                f"但无法满足相邻行接缝错开至少 {gap} 的要求，无解。"
            ),
        }

    used_mask = 0
    rows_out = []
    for r, row in enumerate(best_rows, start=1):
        joints = []
        acc = 0
        for p in row[:-1]:
            acc += lens[p]
            joints.append(acc)
        for p in row:
            used_mask |= 1 << p
        rows_out.append(
            {
                "row_no": r,
                "pieces": [
                    {
                        "board_no": ids[p][0],
                        "cut_no": ids[p][1],
                        "length": lens[p],
                    }
                    for p in row
                ],
                "joints": joints,
                "total_length": target,
            }
        )

    return {
        "feasible": True,
        "used_count": best_count,
        "rows": rows_out,
        "unused": [
            {"board_no": ids[i][0], "cut_no": ids[i][1], "length": lens[i]}
            for i in range(n)
            if not (used_mask >> i) & 1
        ],
    }
