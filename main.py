def select_col(matrix, col, start=0, end=float('inf')):
    # 取出matrix的第col列的行数为从start到end的元素
    # 默认col列全部取出
    # input: matrix = [[1,2,3],[1,2,3]], col = 1
    # output: res = [[2], [2]]
    res = []
    row = len(matrix)
    if row == 0:
        return res
    for i in range(start, min(row, end+1)):
        res.append([matrix[i][col]])
    return res


def transpose(matrix):
    # 列表转置
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return
    m, n = len(matrix), len(matrix[0])
    res = [[0] * m for i in range(n)]
    for i in range(m):
        for j in range(n):
            res[j][i] = matrix[i][j]
    return res


def list_join(m1, m2):
    # 将m2连接到m1后面
    if len(m1) == 0:
        return m2
    if len(m2) == 0:
        return m1
    row1, col2 = len(m1), len(m2[0])
    for i in range(row1):
        for j in range(col2):
            m1[i].append(m2[i][j])
    return m1


def permute(nums):
    # 计算nums的全排列
    res = []
    n = len(nums)
    def helper(nums, start, end):
        if start >= n:
            res.append(nums[:])
            return
        for i in range(start, end):
            nums[start], nums[i] = nums[i], nums[start]
            helper(nums, start+1, end)
            nums[start], nums[i] = nums[i], nums[start]
    helper(nums, 0, n)
    return res


def select_y_from_n(n, y):
    # 从[0,1,2,..,n]中取出长度为y的子序列且不重复
    res = []
    nums = [i for i in range(n)]
    def helper(nums, tmp, pos):
        if len(tmp) == y:
            res.append(tmp[:])
            return
        for i in range(pos, n):
            tmp.append(nums[i])
            helper(nums, tmp, i+1)
            tmp.pop()
    helper(nums, [], 0)
    return res


# 舞蹈链算法
def solve(X, Y, solution=[]):

    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def bool2num(line):
    # 将只含有0,1的01矩阵转换为序列编码
    # line:[1,0,0,1]
    # res:[1,4]
    res = []
    for i in range(len(line)):
        if line[i] == 1:
            res.append(i+1)
    return res


if __name__ == '__main__':
    TraceMatrix = [
        ['b', 'c', 'a', 'a'],
        ['a', 'b', 'b', 'c'],
        ['a', 'c', 'a', 'c'],
        ['b', 'b', 'b', 'a']
    ]
    Library = {
        'L1': [['c', 'a', 'b'], ['c', 'a', 'c'], ['a', 'b', 'b']],
        'L2': [['a'], ['b']],
        'L3': [['c', 'b']],
        'L4': [['a']]
    }
    times = len(TraceMatrix)               # T
    robot_number = len(TraceMatrix[0])     # n
    library_number = len(Library)          # plan library
    BooleanMatrix = []                     # E
    boolmat_plan_index = []                # 记录E中每一行对应plan的序号

    for i in range(len(Library)):
        key = 'L' + str(i+1)
        plan = Library[key]
        x, y = len(plan), len(plan[0])
        c_n_ys = select_y_from_n(robot_number, y)  # 所有可能的排列,permute_res = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3], [3, 2]]
        permute_res = []
        for cny in c_n_ys:
            tmp = permute(cny)
            permute_res.extend(tmp)

        for r in range(times-x+1):
            u_matrix = []
            for c in range(robot_number):
                u_matrix = list_join(u_matrix, select_col(TraceMatrix, c, r, r+x-1))

            for pers in permute_res:
                tmp = []
                for per in pers:
                    tmp = list_join(tmp, select_col(u_matrix, per))
                if tmp == plan:
                    A = [0] * (times * robot_number)
                    for i in pers:
                        for j in range(x):
                            A[(r+j)*robot_number+i] = 1
                    BooleanMatrix.append(A)
                    boolmat_plan_index.append(key)

    bool_mat_row_num = len(BooleanMatrix)
    bool_mat_col_num = times * robot_number
    X = {i+1 for i in range(bool_mat_col_num)}
    Y = {}
    for i in range(bool_mat_row_num):
        Y[i+1] = bool2num(BooleanMatrix[i])

    X = {j: set() for j in X}
    for i in Y:
        for j in Y[i]:
            X[j].add(i)

    bool_index_result = list(solve(X, Y, []))[0]    # 舞蹈链计算结果

    plan_reg_result = {}           # 最终规划识别结果，{plan：{time：[robots]}}
    for i in range(times):
        plan_reg_result['L'+str(i+1)] = {}
    for b_i_res in bool_index_result:
        plan_name = boolmat_plan_index[b_i_res-1]
        boolmat_row = BooleanMatrix[b_i_res-1]
        for i in range(bool_mat_col_num):
            if boolmat_row[i] == 1:
                time = i // robot_number + 1
                robot = i % robot_number
                if time not in plan_reg_result[plan_name]:
                    plan_reg_result[plan_name][time] = [robot]
                else:
                    plan_reg_result[plan_name][time].append(robot)
    print(plan_reg_result)